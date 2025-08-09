# from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.calculators.calculator import Calculator
import numpy as np
from scipy.spatial.transform import Rotation
from typing import ClassVar, Literal
import os

from pathlib import Path
from functools import partial
from ase.calculators.gaussian import Gaussian
from kinbot.ase_sella_util import get_valid_multiplicity

#from aimnet import load_AIMNetMT_ens, load_AIMNetSMD_ens, AIMNetCalculator
from aimnet2calc import AIMNet2ASE
from ase.atoms import Atoms
from ase.vibrations import Vibrations
from fairchem.core.models.model_registry import model_name_to_local_file
import warnings
import logging
import tempfile
import shutil

ELEMENT_ENERGIES = {
    # energies in Hartree at coupled cluster cc-pCVTZ level
    # from https://cccbdb.nist.gov/
    "H": -0.497449,
    "C": -37.827545,
    "N": -54.571306,  # composite method G4
    "O": -75.028319,
    "S": -397.983905,
}

ROOT_PATH = Path("/hpcwork/zo122003/BA/")
CUTOFF = 15.0
MAX_NEIGHBORS = 60
HARTREE_TO_EV = 27.211386245988


class Nn_surr(Calculator):
    """Neural Network calculator implementing both cheap energy and expensive force calculations."""
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        energy_calculator_name: str | None = None,
        force_calculator_name: str | None = None,
        unit: Literal["hartree", "ev"] = "ev",
    ) -> None:
        Calculator.__init__(self)

        warnings.filterwarnings('ignore', category=FutureWarning,
            message='You are using `torch.load` with `weights_only=False`')

        if force_calculator_name is None:
            self.force_calculator_name = os.environ.get(
                "KINBOT_FORCE_CALCULATOR_NAME", "new_12_layers"
            )
        else:
            self.force_calculator_name = force_calculator_name

        if energy_calculator_name is None:
            self.energy_calculator_name = os.environ.get(
                "KINBOT_ENERGY_CALCULATOR_NAME", "aimnet2ens"
            )
        else:
            self.energy_calculator_name = energy_calculator_name

        print(f"Using energy calculator name: {self.energy_calculator_name}")
        print(f"Using force calculator name: {self.force_calculator_name}")

        self._energy_calculator = None
        self._force_calculator = None

        if unit == "hartree":
            self.unit = "hartree"
        else:
            self.unit = "ev"

    @property
    def energy_calculator(self):
        """Lazy loading of energy calculator."""
        if self._energy_calculator is None:
            self._energy_calculator = self.get_energy_calculator(
                self.energy_calculator_name)
        return self._energy_calculator

    @property
    def force_calculator(self):
        """Lazy loading of force calculator."""
        if self._force_calculator is None:
            self._force_calculator = self.get_force_calculator(
                self.force_calculator_name)
        return self._force_calculator

    def get_energy_calculator(
        self,
        calc_type: str = "aimnet2ens",
    ) -> Calculator:
        """
        Returns a calculator optimized for quick energy calculations.

        Temporarily disables logging to suppress log outputs from
        the AIMNetCalculator initialization.

        :return: An instance of AIMNetCalculator
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message="You are using `torch.load` with `weights_only=False`",
                )
                logging.disable(logging.CRITICAL)
                if calc_type == "ocp":
                    calc = self.get_force_calculator()
                elif calc_type == "aimnet2":
                    calc = AIMNet2ASE('aimnet2',
                                      mult=get_valid_multiplicity(self.atoms))
                elif calc_type == "aimnet2ens":
                    calc = AIMNet2ASE_ENS()
                elif calc_type.startswith("gaussian"):
                    temp_dir = tempfile.mkdtemp(prefix="gaussian_calc_")
                    params = {
                        "mem": "12GB",
                        "nprocshared": 4,
                        "mult": get_valid_multiplicity(self.atoms),
                        "directory": temp_dir,
                        "extra": "scf=(tight, xqc) int=ultrafine guess=mix",  # "XQC,Conver=6,Direct"
                        "force": "",
                        "geom": "NoCrowd",
                    }
                    if calc_type in ("gaussian_wb97x_6_31g", "gaussian"):
                        params["method"] = "wb97x"
                        params["basis"] = "6-31G(d)"
                    elif calc_type == "gaussian_pm6":
                        params["method"] = "pm6"
                    else:
                        logging.disable(logging.NOTSET)
                        raise ValueError(
                            f"Unsupported Gaussian method: {calc_type}"
                        )
                    calc = Gaussian(**params)
                else:
                    logging.disable(logging.NOTSET)
                    raise ValueError(
                        f"Unsupported energy calculator: {calc_type}"
                    )
        finally:
            # Re-enable logging after calculator creation
            logging.disable(logging.NOTSET)
        return calc

    def get_force_calculator(
        self, force_calculator_name: str = "new_12_layers"
    ) -> Calculator:
        """
        Returns a calculator optimized for force calculations without extraneous
        log outputs.

        Temporarily disables logging to suppress the OCPCalculator messages.

        :return: Instance of the force calculator created via OCPCalculator.
        """

        if self._force_calculator is not None:
            return self._force_calculator

        if force_calculator_name.startswith("gaussian"):
            # Defer to get_energy_calculator for Gaussian instance creation
            # to avoid code duplication.
            # We can safely assume self.atoms is populated before this is called.
            calc = self.get_energy_calculator(calc_type=force_calculator_name)
            self._force_calculator = calc
            return self._force_calculator

        get_model = partial(
            model_name_to_local_file, local_cache="/hpcwork/zo122003/BA/models"
        )
        calculators = {
            "main": ROOT_PATH / "rundir/out/checkpoints/2024-12-06-11-20-48/best_freq_rmse_checkpoint.pt",
            "main_2": ROOT_PATH / "/hpcwork/zo122003/BA/rundir/out/checkpoints/2024-12-09-13-03-12/best_freq_rmse_checkpoint.pt",
            "small": ROOT_PATH / "rundir/out/checkpoints/2024-12-09-16-49-20/best_freq_rmse_checkpoint.pt",
            "small2": ROOT_PATH / "rundir/out/checkpoints/2024-12-11-14-09-20/best_freq_rmse_checkpoint.pt",
            "small3": ROOT_PATH / "rundir/out/checkpoints/2024-12-10-17-36-16/best_freq_rmse_checkpoint.pt",
            "mini": ROOT_PATH / "rundir_small/out/checkpoints/2024-12-12-12-44-00/best_freq_rmse_checkpoint.pt",
            "minier": ROOT_PATH / "rundir_smaller/out/checkpoints/2024-12-12-15-45-20/best_freq_rmse_checkpoint.pt",
            "small_success": ROOT_PATH / "rundir_small/out/checkpoints/2024-12-16-19-25-04/best_freq_rmse_checkpoint.pt",
            "smaller_success": ROOT_PATH / "rundir_smaller/out/checkpoints/2024-12-16-19-22-56/best_freq_rmse_checkpoint.pt",
            "best_no_dens": ROOT_PATH / "rundir_smaller/out/checkpoints/2024-12-26-10-42-24/best_freq_rmse_checkpoint.pt",
            "n3_best_chkpt": ROOT_PATH / "rundir_small/out/checkpoints/2025-01-27-13-33-04/best_checkpoint.pt",
            "n8_chkpt": ROOT_PATH / "rundir_small/out/checkpoints/2025-01-28-12-35-28/checkpoint.pt",
            "latest": ROOT_PATH / "rundir_small/out/checkpoints/2025-01-31-23-34-40/checkpoint.pt",
            "8_fin": ROOT_PATH / "rundir_small/out/checkpoints/2025-01-31-23-34-40/checkpoint.pt",
            "12_layers": ROOT_PATH / "rundir_12/out/checkpoints/2025-02-17-16-19-28/best_freq_rmse_checkpoint.pt",
            "new_8_layers":"/hpcwork/thes1749/rundir_8/out/checkpoints/2025-05-20-12-01-36/best_checkpoint.pt", #Nr1
            "new_12_layers":"/hpcwork/thes1749/rundir_12/out/checkpoints/2025-05-20-11-57-20/best_checkpoint.pt", #Nr2
            "new_7_layers":"/hpcwork/thes1749/rundir_7/out/checkpoints/2025-06-30-05-31-12/best_checkpoint.pt"
        }

        checkpoint_path = calculators[force_calculator_name]

        # Temporarily disable logging to suppress any log messages
        logging.disable(logging.CRITICAL)
        try:
            calc = OCPCalculator(
                checkpoint_path=checkpoint_path,
                #cutoff=CUTOFF,
                #max_neighbors=MAX_NEIGHBORS,
                cpu=True,
                seed=42
            )
        except Exception as e:
            raise ValueError(f"Error creating force calculator: {e}")
        finally:
            # Re-enable logging after creating the calculator
            logging.disable(logging.NOTSET)
        return calc

    def calculate(
        self,
        atoms: Atoms,
        properties,
        system_changes,
    ) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results["energy"] = self.get_potential_energy(atoms=atoms)
        self.results["forces"] = self.get_forces(atoms=atoms)

    def _get_energy_from_calculator(self, use_force_fallback: bool = False) -> float:
        """
        Helper method to get energy from the appropriate calculator and handle units.

        :param use_force_fallback: If True, use the force calculator as a fallback.
        :return: Energy in the correct unit (eV or Hartree).
        """
        # Determine which calculator to use
        calculator = self.force_calculator if use_force_fallback else self.energy_calculator
        calculator_name = self.force_calculator_name if use_force_fallback else self.energy_calculator_name

        raw_energy = calculator.get_potential_energy(self.atoms)

        # Ensure energy is a float
        if hasattr(raw_energy, "item"):
            raw_energy = raw_energy.item()

        # Handle unit conversions
        # OCP models are in Hartree, others are in eV
        if isinstance(calculator, OCPCalculator):
            return (
                raw_energy
                if self.unit == "hartree"
                else raw_energy * HARTREE_TO_EV
            )
        else:  # AIMNet, Gaussian, etc. are in eV
            return (
                raw_energy / HARTREE_TO_EV
                if self.unit == "hartree"
                else raw_energy
            )

    def get_potential_energy(
        self,
        atoms=None,
        **kwargs,
    ):
        """
        Calculates the potential energy, handling different calculators and units.
        """
        if atoms is not None:
            self.atoms = atoms

        if len(self.atoms) == 1:
            energy_hartree = ELEMENT_ENERGIES[self.atoms.get_chemical_symbols()[0]]
            return energy_hartree if self.unit == "hartree" else energy_hartree * HARTREE_TO_EV

        try:
            return self._get_energy_from_calculator(use_force_fallback=False)
        except Exception as error:
            logging.error(
                f"Error in energy calculation with {self.energy_calculator_name}: {error}"
            )
            return self._get_energy_from_calculator(use_force_fallback=True)


    def get_forces(self, atoms=None):
        if atoms is not None:
            self.atoms = atoms

        if len(self.atoms) == 1:
            return np.zeros((1, 3), dtype=np.float64)

        raw_forces = self.force_calculator.get_forces(self.atoms).astype(np.float64)

        # OCP forces are in Hartree/Angstrom, others (Gaussian) are in eV/Angstrom
        if isinstance(self.force_calculator, OCPCalculator):
            if self.unit == "ev":
                return raw_forces * HARTREE_TO_EV
            return raw_forces  # Already in Hartree/A
        else:
            if self.unit == "hartree":
                return raw_forces / HARTREE_TO_EV
            return raw_forces  # Already in eV/A


def get_calculator():
    """
    Get a Neural Network Surrogate calculator instance.

    This is a wrapper/modification of an ASE calculator implementation that provides
    a neural network surrogate model for potential energy surface predictions.

    Returns
    -------
    Nn_surr
        Instance of the Neural Network Surrogate calculator class
    """

    return Nn_surr()

def calculate_vibrational_properties(calculator, work_dir, atoms):
    """
    Calculate vibrational frequencies, zero-point energy, and Hessian, then
    clean up the temporary working directory files.

    If the work_dir parameter is the string "random", a random temporary
    directory is created (ensuring it does not already exist) and deleted
    after the analysis.

    This function performs vibrational analysis on the given atoms using the
    provided ASE calculator. Temporary files are written to the specified
    working directory during the analysis and are removed afterwards using the
    Vibrations.clean() method (and the directory is deleted if created randomly).

    :param calculator: ASE calculator for force and energy evaluations.
    :type calculator: ase.calculators.calculator.Calculator
    :param work_dir: Directory path where vibrational analysis files are stored,
                     or the string "random" to use a temporary directory.
    :type work_dir: str or pathlib.Path
    :param atoms: ASE Atoms object representing the system.
    :type atoms: ase.atoms.Atoms
    :raises Exception: Propagates any error encountered during analysis.
    :return: Tuple of vibrational frequencies (numpy.ndarray), zero-point energy
             (float), and Hessian matrix (numpy.ndarray) or None if not available.
    :rtype: tuple
    """

    temporary_dir_created = False
    # Check if a random directory should be used
    if isinstance(work_dir, str) and work_dir.lower() == "random":
        # Create a random temporary directory (it is guaranteed not to exist)
        work_dir = Path(tempfile.mkdtemp(prefix="ase_vib_"))
        temporary_dir_created = True
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Attach the provided calculator to the atoms object.
        atoms.calc = calculator

        # Create a prefix for the vibration output files inside work_dir.
        vib_prefix = str(work_dir / "vibration")

        # Initialize the Vibrations object.
        vib = Vibrations(atoms, name=vib_prefix)

        # Run the vibrational analysis.
        vib.run()

        # Retrieve vibrational frequencies and zero-point energy.
        frequencies = vib.get_frequencies()
        zpe = vib.get_zero_point_energy()

        # Try to obtain the Hessian matrix; if retrieval fails, set to None.
        try:
            hessian = vib.get_hessian()
        except Exception:
            hessian = None

        # Clean up the temporary files created during the analysis.
        vib.clean()
        return frequencies, zpe, hessian

    except Exception as error:
        logging.error("Error in vibrational properties calculation: %s", error)
        raise

    finally:
        # If a random temporary directory was created, remove it.
        if temporary_dir_created:
            shutil.rmtree(work_dir)


def make_lowest_energy_callback(mol, energy_container):
    """
    Create stateful callback that tracks the lowest energy conformer.

    This callback updates a mutable dictionary with the lowest energy
    found and corresponding Atoms object.

    This is used to help combat divergence issues. If we hit max steps during
    optimization, we either:
    1. Did not allow for enough steps and slow convergence.
    2. The optimization is diverging.

    In the former case, the lowest energy conformer will simply be the
    structure from the latest iteration. In the latter case, the most optimal
    structure will be the one from the previous iterations.

    In either case we can use the lowest energy structure from a previous
    iteration as a starting point for the next round of optimization and try
    again.

    :param mol: Target molecule.
    :type mol: ase.atoms.Atoms
    :param energy_container: Mutable dictionary to store lowest energy and
                             corresponding Atoms object.
    :return: Configured callback function.
    :rtype: callable
    """
    def callback():
        """
        Evaluate current energy and update lowest energy container if
        improved.
        """
        energy_container["step"] += 1
        try:
            current_energy = mol.get_potential_energy()
            if current_energy < energy_container.get('min_energy', np.inf):
                energy_container['min_energy'] = current_energy
                energy_container['positions'] = mol.get_positions().copy()
        except Exception as e:
            logging.error("Lowest energy callback error: %s", e)
        return
    return callback


def make_rms_callback(
    mol: Atoms, threshold_container: list, window_size: int = 5
) -> callable:
    """
    Create stateful RMS displacement callback with proper initialization.

    This callback calculates the running average of rotation-invariant RMS
    displacement over the last N steps using the Kabsch algorithm. It raises
    an RMSConvergence exception if the average falls below a specified
    threshold.

    :param mol: Target molecule
    :type mol: ase.atoms.Atoms
    :param threshold_container: Mutable container holding the RMS threshold
    :type threshold_container: list
    :param window_size: Number of steps to consider for the running average
    :type window_size: int
    :return: Configured callback function
    :rtype: callable
    """
    state = {
        'prev_pos': None,  # Will be initialized on first call
        'initialized': False,
        'window_size': window_size,
        'rms_history': [float('inf')] * window_size,  # initialize with inf
    }

    def check_rms():
        """
        Calculate running average of RMS displacement over the last N steps.

        Checking the rmsd using the Kabsch algorithm alone is sufficient since
        geometries dont change much with each step. So no reordering of atoms
        using the hungarian algorithm is needed here.
        """
        nonlocal state
        current_pos = mol.get_positions()

        if not state['initialized']:
            # Initialize on first call (after first step)
            state['prev_pos'] = current_pos.copy()
            state['initialized'] = True
            return

        # Center the coordinates
        prev_pos_centered = state['prev_pos'] - state['prev_pos'].mean(axis=0)
        current_pos_centered = current_pos - current_pos.mean(axis=0)

        # Find the optimal rotation using the Kabsch algorithm.
        _, rssd = Rotation.align_vectors(current_pos_centered, prev_pos_centered)

        # The RMSD is the square root of the mean of the squared differences.
        rms = np.sqrt(rssd / len(mol))
        state['prev_pos'] = current_pos.copy()

        state['rms_history'].pop(0)  # Remove oldest RMS
        state['rms_history'].append(rms) # add newest RMS

        # Calculate running average of RMS
        running_avg_rms = np.mean(state['rms_history'])

        if running_avg_rms < threshold_container[0]:
            raise RMSConvergence(
                f"Running average RMS convergence {running_avg_rms:.5f} "
                f"Å reached over {state['window_size']} steps."
            )

    return check_rms


# Custom exception for RMS convergence
class RMSConvergence(Exception):
    """Exception raised when geometry converges based on RMS displacement."""
    pass


class AIMNet2ASE_ENS(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.calc1 = AIMNet2ASE(base_calc="aimnet2/aimnet2_wb97m_0.jpt")
        self.calc2 = AIMNet2ASE(base_calc="aimnet2/aimnet2_wb97m_1.jpt")
        self.calc3 = AIMNet2ASE(base_calc="aimnet2/aimnet2_wb97m_2.jpt")
        self.calc4 = AIMNet2ASE(base_calc="aimnet2/aimnet2_wb97m_3.jpt")

    def get_potential_energy(self, atoms):
        if atoms is not None:
            self.atoms = atoms

        e1 = self.calc1.get_potential_energy(self.atoms).item()
        e2 = self.calc2.get_potential_energy(self.atoms).item()
        e3 = self.calc3.get_potential_energy(self.atoms).item()
        e4 = self.calc4.get_potential_energy(self.atoms).item()
        e = (e1 + e2 + e3 + e4) / 4
        return e

    def get_forces(self, atoms):
        raise NotImplementedError(
            ("Forces generated by AIMNet2 are not accurate enough."))
