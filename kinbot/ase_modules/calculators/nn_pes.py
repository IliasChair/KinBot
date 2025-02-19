# from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.calculators.calculator import Calculator
import numpy as np
from typing import ClassVar, Literal

from pathlib import Path
from functools import partial
from ase.calculators.gaussian import Gaussian
from kinbot.ase_sella_util import get_valid_multiplicity

from aimnet import load_AIMNetMT_ens, load_AIMNetSMD_ens, AIMNetCalculator
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
    def __init__(self, model_name: str = "best_no_dens", unit: Literal["hartree", "ev"] = "ev") -> None:
        Calculator.__init__(self)

        warnings.filterwarnings('ignore', category=FutureWarning,
            message='You are using `torch.load` with `weights_only=False`')

        self.model_name = model_name
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
            self._energy_calculator = self.get_energy_calculator()
        return self._energy_calculator

    @property
    def force_calculator(self):
        """Lazy loading of force calculator."""
        if self._force_calculator is None:
            self._force_calculator = self.get_force_calculator()
        return self._force_calculator

    def get_energy_calculator(
        self,
        calc_type: Literal["AIMNetGas", "AIMNetSMD", "gaussian"] = "gaussian",
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
                if calc_type == "AIMNetGas":
                    calc = AIMNetCalculator(load_AIMNetMT_ens())
                elif calc_type == "AIMNetSMD":
                    calc = AIMNetCalculator(load_AIMNetSMD_ens())
                elif calc_type == "gaussian":
                    temp_dir = tempfile.mkdtemp(prefix='gaussian_calc_')
                    calc = Gaussian(
                            mem='12GB',
                            nprocshared=4,
                            method='wb97xd', #M05, tpss
                            mult=get_valid_multiplicity(self.atoms),
                            basis="6-31G(d)", #"6-31G(d)"  # '6-311++g(d,p)',
                            directory=temp_dir,
                            SCF="XQC,Conver=6,Direct"
                        )
        finally:
            # Re-enable logging after calculator creation
            logging.disable(logging.NOTSET)
        return calc

    def get_force_calculator(self) -> Calculator:
        """
        Returns a calculator optimized for force calculations without extraneous
        log outputs.

        Temporarily disables logging to suppress the OCPCalculator messages.

        :return: Instance of the force calculator created via OCPCalculator.
        """
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
            "8_2": ROOT_PATH / "rundir_small/out/checkpoints/2025-01-22-20-41-52/checkpoint.pt",
        }

        checkpoint_path = calculators["8_2"]

        # Temporarily disable logging to suppress any log messages
        logging.disable(logging.CRITICAL)
        try:
            calc = OCPCalculator(
                checkpoint_path=checkpoint_path,
                cutoff=CUTOFF,
                max_neighbors=MAX_NEIGHBORS,
                cpu=True,
                seed=42
            )
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

    def get_potential_energy(
        self,
        atoms=None,
        calc_type: Literal["gaussian", "default"] = "default",
        **kwargs,
    ):
        """
        In some cases we want to switch between the default cheap energy and
        a higher level dft method. By passing calc_type="gaussian" we can use
        the gaussian calculator to get the energy on a dft level.
        """

        # Update atoms object if it's provided and changed
        if atoms is not None:
            self.atoms = atoms

        # treat the case in which atoms only consists of one element explicitly
        if len(self.atoms) == 1:
            energy_hartree = ELEMENT_ENERGIES[self.atoms.get_chemical_symbols()[0]]
            if self.unit == "hartree":
                return energy_hartree
            else:
                return energy_hartree * HARTREE_TO_EV

        if calc_type == "gaussian":
            calc = self.get_energy_calculator(calc_type="gaussian")
            if self.unit == "hartree":
                # ASE returns energy in Ev
                return calc.get_potential_energy(self.atoms) / HARTREE_TO_EV
            else:
                return calc.get_potential_energy(self.atoms)

        # for some systems the AIMNet model is not able to predict the energy
        # in this case, we use the EquiformerV2 model to predict the energy
        try:
            # the calculator returns an array, we have to convert it to a float using .item()
            energy_EV = self.energy_calculator.get_potential_energy(
                self.atoms
            )

            if type(energy_EV) is np.ndarray:
                energy_EV = energy_EV.item()

            if self.unit == "hartree":
                return energy_EV / HARTREE_TO_EV
            else:
                return energy_EV
        except Exception as error:
            # Log the error that caused the exception.
            logging.error((
                f"Error in energy_calculation using energy_calculator: {error}"
            ))
            energy_hartree = self.force_calculator.get_potential_energy(self.atoms)
            # equiformerv2 was trained on the Hartree unit
            if self.unit == "hartree":
                return energy_hartree
            else:
                return energy_hartree * HARTREE_TO_EV

    def get_forces(self, atoms=None):
        # Update atoms object if it's provided and changed
        if atoms is not None:
            self.atoms = atoms

        # treat the case in which atoms only consists of one element explicitly
        if len(self.atoms) == 1:
            return np.zeros((1, 3), dtype=np.float64)

        forces = self.force_calculator.get_forces(self.atoms).astype(np.float64)
        if self.unit == 'hartree':
            return forces  # Already in Hartree/Å
        else:
            return forces * HARTREE_TO_EV  # Convert to eV/Å


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
        atoms.set_calculator(calculator)

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


def make_rms_callback(mol: Atoms, threshold_container: list) -> callable:
    """
    Create stateful RMS displacement callback with proper initialization.

    :param mol: Target molecule
    :param threshold_container: Mutable container holding the RMS threshold
    :return: Configured callback function
    """
    state = {
        'prev_pos': None,  # Will be initialized on first call
        'initialized': False,
        'consecutive_below_thresh': 0  # Counter for consecutive steps
    }

    def check_rms():
        """Calculate RMS displacement between consecutive steps."""
        nonlocal state
        current_pos = mol.get_positions()

        if not state['initialized']:
            # Initialize on first call (after first step)
            state['prev_pos'] = current_pos.copy()
            state['initialized'] = True
            return

        rms = np.sqrt(np.mean((current_pos - state['prev_pos'])**2))
        state['prev_pos'] = current_pos.copy()

        if rms < threshold_container[0]:
            state['consecutive_below_thresh'] += 1
            if state['consecutive_below_thresh'] >= 2:
                raise RMSConvergence(
                    f"RMS convergence {rms:.5f} Å reached for two consecutive steps."
                )
        else:
            state['consecutive_below_thresh'] = 0  # Reset counter if condition not met

    return check_rms


# Custom exception for RMS convergence
class RMSConvergence(Exception):
    """Exception raised when geometry converges based on RMS displacement."""
    pass