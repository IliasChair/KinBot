# from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.calculators.calculator import Calculator
import numpy as np
import os
from typing import ClassVar, Literal
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from functools import partial
from typing import List, Optional, Dict

from aimnet import load_AIMNetMT_ens, load_AIMNetSMD_ens, AIMNetCalculator
from ase.atoms import Atoms
from fairchem.core.models.model_registry import model_name_to_local_file
import warnings

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
            self._energy_calculator = self.get_cheap_energy_calculator()
        return self._energy_calculator

    @property
    def force_calculator(self):
        """Lazy loading of force calculator."""
        if self._force_calculator is None:
            self._force_calculator = self.get_force_calculator()
        return self._force_calculator

    def get_cheap_energy_calculator(self) -> 'Calculator':
        """Returns a calculator optimized for quick energy calculations."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning,
                          message='You are using `torch.load` with `weights_only=False`')
            energy_calcs = {"AIMNetGas": load_AIMNetMT_ens(),
                            "AIMNetSMD": load_AIMNetSMD_ens()}

        return AIMNetCalculator(energy_calcs["AIMNetGas"])

    def get_force_calculator(self) -> 'Calculator':
        """Returns a calculator optimized for force calculations."""
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
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                calc = OCPCalculator(
                    checkpoint_path=checkpoint_path,
                    cutoff=CUTOFF,
                    max_neighbors=MAX_NEIGHBORS,
                    cpu=True,
                    seed=42
                )
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

    def get_potential_energy(self, atoms=None, **kwargs):
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

        # for some systems the AIMNet model is not able to predict the energy
        # in this case, we use the EquiformerV2 model to predict the energy
        try:
            # the calculator returns an array, we have to convert it to a float using .item()
            energy_EV = self.energy_calculator.get_potential_energy(
                self.atoms
            ).item()

            if self.unit == "hartree":
                return energy_EV / HARTREE_TO_EV
            else:
                return energy_EV
        except:
            energy_hartree = self.force_calculator.get_potential_energy(self.atoms)
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