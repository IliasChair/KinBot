# from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.calculators.calculator import Calculator, all_changes
import numpy as np

from aimnet import load_AIMNetMT_ens, load_AIMNetSMD_ens, AIMNetCalculator

element_specific_energies = {
    # energies in Hartree at coupled cluster cc-pCVTZ level
    # from https://cccbdb.nist.gov/
    "H": -0.497449,
    "C": -37.827545,
    "N": -54.571306,  # composite method G4
    "O": -75.028319,
    "S": -397.983905,
}


class Nn_surr(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        fname="NNPES",
        restart=None,
        ignore_bad_restart_file=False,
        label="surrogate",
        atoms=None,
        tnsr=False,
        **kwargs,
    ):
        Calculator.__init__(
            self, restart=restart, label=label, atoms=atoms, tnsr=tnsr, **kwargs
        )

        # expensive calculator for forces and energy only instantiated when needed
        self.expensive_force_calculator = None
        # cheap calculator for energy only
        self.cheap_energy_calculator = self.get_cheap_energy_calculator()

    def get_cheap_energy_calculator(self):
        model_gas = load_AIMNetMT_ens()
        model_smd = load_AIMNetSMD_ens()

        energy_calc = AIMNetCalculator(model_gas)

        return energy_calc

    def get_expensive_force_calculator(self):
        root = "/hpcwork/zo122003/BA/"
        # calculators = {"2M":               f"{root}rundir_2M/DeNS_output/checkpoints/2024-08-29-21-58-56/best_checkpoint.pt",
        #                "Main":             f"{root}rundir/DeNS_output/checkpoints/2024-08-27-23-34-56/best_checkpoint.pt",
        #                "energy":           f"{root}rundir_energy_forces/energy/DeNS_output/checkpoints/2024-09-02-14-58-40/best_checkpoint.pt",
        #                "ANI-1x_energy":    f"{root}rundir_energy_forces/true_energy/DeNS_output/checkpoints/2024-09-14-17-17-20/best_checkpoint.pt",
        #                "forces":           f"{root}rundir_energy_forces/forces/DeNS_output/checkpoints/2024-09-02-02-10-40/best_checkpoint.pt",
        #                "ANI-2x":           f"{root}rundir_ANI-2x/DeNS_output/checkpoints/2024-09-08-22-54-24/best_checkpoint.pt",
        #                "ANI-2x_restart":   f"{root}rundir_ANI-2x/DeNS_output/checkpoints/2024-09-12-13-31-12/best_checkpoint.pt",
        #                "ANI-2x_energy":    f"{root}rundir_ANI-2x/energy_only/DeNS_output/checkpoints/2024-09-14-17-10-56/best_checkpoint.pt",}
        calculators = {
            "main": f"{root}rundir__no_pbc/out/checkpoints/2024-11-14-10-10-24/best_checkpoint.pt",
        }

        force_energy_calc = OCPCalculator(
            checkpoint_path=calculators["main"],
            cutoff=12.0,
            max_neighbors=20,
            cpu=True,
            seed=42,
        )
        return force_energy_calc

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
        loaddb=None,
        args=None,
        xid=None,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        # Update atoms object if it's provided and changed
        if atoms is not None and self.atoms is not atoms:
            self.set_atoms(atoms)

        # Invoke surrogate models to predict energy and forces
        if "energy" in properties:
            self.results["energy"] = self.get_potential_energy()

        if "forces" in properties:
            self.results["forces"] = self.get_forces()

    def get_potential_energy(self, atoms=None):
        # Update atoms object if it's provided and changed
        if atoms is not None:
            self.atoms = atoms

        # treat the case in which atoms only consists of one element explicitly
        if len(self.atoms) == 1:
            energy = element_specific_energies[self.atoms.get_chemical_symbols()[0]]
            # convert hartree to eV
            energy = energy * 27.211386245988
            return energy

        # for some systems the AIMNet model is not able to predict the energy
        # in this case, we use the EquiformerV2 model to predict the energy
        try:
            # if the calculator returns an array, we have to convert it to a float using .item()
            energy = self.cheap_energy_calculator.get_potential_energy(
                self.atoms
            ).item()

            return energy
        except:
            if self.expensive_force_calculator is None:
                self.expensive_force_calculator = self.get_expensive_force_calculator()

            energy = self.expensive_force_calculator.get_potential_energy(self.atoms)

            # convert to float64
            energy = energy.astype(np.float64)
            # convert to eV
            energy = energy * 27.211386245988
            return energy

    def get_forces(self, atoms=None):
        # Update atoms object if it's provided and changed
        if atoms is not None:
            self.atoms = atoms

        # treat the case in which atoms only consists of one element explicitly
        if len(self.atoms) == 1:
            forces = np.zeros((1, 3), dtype=np.float64)
            return forces

        # if expensive_force_calculator has not been initialized, initialize it
        if self.expensive_force_calculator is None:
            self.expensive_force_calculator = self.get_expensive_force_calculator()

        forces = self.expensive_force_calculator.get_forces(self.atoms).astype(
            np.float64
        )

        # convert Hartree/Angstrom to eV/Angstrom
        forces = forces * 27.211386245988
        return forces
