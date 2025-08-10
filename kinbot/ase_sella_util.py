from ase import Atoms
from ase.data import atomic_numbers
import numpy as np
from typing import Optional
from ase.calculators.gaussian import Gaussian
import os
import logging
import subprocess
import cclib
from sella import Sella


def get_valid_multiplicity(atoms: Atoms, default_mult: int = 1) -> int:
    """
    Determine an alternative spin multiplicity for Gaussian calculations.

    This function heuristically computes the total electron count from an ASE
    Atoms object and then adjusts the provided default multiplicity to match
    the expected parity based on the electron count. In standard electronic
    structure theory, the spin multiplicity, defined as 2S+1, has parity in
    relation to the number of electrons:

    - A system with an even number of electrons (closed-shell or singlet-like)
      should have an odd multiplicity.
    - A system with an odd number of electrons should have an even
      multiplicity.

    The algorithm proceeds as follows:

    1. Compute the total number of electrons by summing the atomic numbers of
       all atoms in the system.
    2. If the total electron count is even:
       - If the provided default multiplicity is even, add 1 so that the
         resulting multiplicity is odd.
       - Otherwise, return the default multiplicity as it is already odd.
    3. If the total electron count is odd:
       - If the default multiplicity is odd, add 1 so that the resulting
         multiplicity is even.
       - Otherwise, return the default multiplicity as it is already even.

    This computation is solely intended as a workaround for cases where
    Gaussian electronic structure calculations fail due to an inappropriate
    multiplicity assignment. It does not guarantee that the adjusted
    multiplicity reflects the true ground state of the system.

    :param atoms: ASE Atoms object representing the molecular system.
    :type atoms: ase.Atoms
    :param default_mult: The initial multiplicity attempted (which may have
                         led to a failure), defaults to 1.
    :type default_mult: int, optional
    :returns: A heuristically adjusted multiplicity that is more likely to
              satisfy Gaussian's requirements.
    :rtype: int

    .. warning::
       This function is a technical workaround and should not be used as a
       substitute for a rigorous determination of a system's electronic state.
       The returned multiplicity may not correspond to the true physical spin
       state of the molecule and is only used to help Gaussian complete
       calculations that would otherwise fail.

    .. note::
       Within KinBot, this function is part of the error recovery mechanism
       for quantum chemistry calculations. When Gaussian fails with the initial
       multiplicity, this function is called to generate an alternative value,
       after which a new calculation is attempted.

    .. seealso::
       Related error handling functions within KinBot's quantum chemistry
       workflow that interface with Gaussian.
    """
    # Sum the nuclear charges of all atoms to get the total number
    # of electrons (a rough approximation).
    total_electrons = sum(atomic_numbers[atom.symbol] for atom in atoms)

    if total_electrons % 2 == 0:
        # For even-electron systems, valid spin multiplicity (2S+1) must be odd.
        return default_mult + 1 if default_mult % 2 == 0 else default_mult
    else:
        # For odd-electron systems, valid spin multiplicity must be even.
        return default_mult + 1 if default_mult % 2 == 1 else default_mult


def validate_frequencies(freqs: np.ndarray, order: int) -> bool:
    """
    Validate frequency calculation results for geometry optimization
    convergence.

    This function validates vibrational frequencies to ensure proper
    convergence to either a minimum or a transition state. It distinguishes
    between small imaginary frequencies arising from numerical noise and
    significantly imaginary ones (indicative of a real instability or a true
    reaction coordinate).

    :param freqs: Array of vibrational frequencies in cm^-1
    :type freqs: numpy.ndarray
    :param order: Optimization order (0 for minimum, 1 for transition state)
    :type order: int
    :returns: True if frequencies match expected pattern for the given order,
              False otherwise.
    :rtype: bool

    For minima (order=0):
        * Allows at most one small imaginary frequency
          (-70 < freq < 0) to account for numerical noise.
        * Does not allow any significantly imaginary frequencies
          (freq < -70).
        * In other words, returns False if either:
            - Two or more frequencies are less than 0 (n_imag >= 2), or
            - Any frequency is less than -70 (n_large_imag >= 1).

    For transition states (order=1):
        * Generally expects one dominant mode corresponding to the reaction
          coordinate. Here it allows:
            - Either one or two imaginary frequencies overall, but no more
              than one is allowed to be significantly imaginary.
        * Returns False if any of the following conditions are met:
            - More than two frequencies are imaginary.
            - Two or more frequencies are significantly imaginary.
            - There are no imaginary frequencies.

    .. note::
        Small imaginary frequencies (between 0 and -70 cm^-1) often appear
        in quantum chemistry calculations due to numerical imprecision,
        particularly for large, floppy molecules or loose convergence
        criteria. These are typically not chemically meaningful and can be
        safely ignored. However, frequencies below -70 cm^-1 usually
        indicate real transition vectors that should be investigated.
        When validation fails, KinBot's optimization workflow typically
        reduces the force criterion (fmax) by a factor and retries the
        optimization with tighter criteria.

    .. seealso::
        This function is used in both well and saddle point optimization
        templates to ensure that only properly converged geometries are
        accepted into the KinBot reaction network.
    """
    n_imag = np.count_nonzero(freqs < 0)
    n_large_imag = np.count_nonzero(freqs < -70)

    if order == 0:
        return not (n_imag >= 2 or n_large_imag >= 1)
    elif order == 1:
        return not (n_imag > 2 or n_large_imag >= 2 or n_imag == 0)
    return False


def setup_gaussian_calc(mol: Atoms, BASE_LABEL: str, CALC_DIR: str, mult: Optional[int] = None) -> None:
    """Configure Gaussian calculator for the molecule."""

    calc_params = {
        "mem": "8GB",
        "nprocshared": 4,
        "method": "wb97xd",
        "basis": "def2tzvp",
        "chk": f"{BASE_LABEL}_vib.chk",
        "extra": "freq=noraman scf=(tight, xqc) int=ultrafine guess=mix geom=nocrowd",
    }
    if mult is not None:
        calc_params["mult"] = mult

    mol.calc = Gaussian(**calc_params)
    mol.calc.label = os.path.join(CALC_DIR, f"{BASE_LABEL}_vib")


def calc_vibrations(
        mol: Atoms, BASE_LABEL: str, CALC_DIR: str
        ) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Calculate vibrational frequencies."""
    mol = mol.copy()

    # Try calculation with default multiplicity
    setup_gaussian_calc(mol=mol,
                        BASE_LABEL=BASE_LABEL,
                        CALC_DIR=CALC_DIR,
                        mult=get_valid_multiplicity(mol))
    dft_energy = None
    try:
        dft_energy = mol.get_potential_energy()
    except Exception as error:
        print(f"Initial calculation failed: {error}")
    if dft_energy is None:
        raise RuntimeError(
            f"Vibrational frequency calculation failed for {BASE_LABEL}")

    # Process frequency results
    chk_path = os.path.join(CALC_DIR, f"{BASE_LABEL}_vib.chk")
    fchk_path = os.path.join(CALC_DIR, f"{BASE_LABEL}_vib.fchk")
    log_path = os.path.join(CALC_DIR, f"{BASE_LABEL}_vib.log")
    subprocess.run(["formchk", chk_path, fchk_path], check=True)

    fchk_data = cclib.io.ccopen(fchk_path).parse()
    log_data = cclib.io.ccopen(log_path).parse()

    return fchk_data.vibfreqs, log_data.zpve, fchk_data.hessian, dft_energy


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

class SellaWrapper:
    """A wrapper for the Sella optimizer to handle lowest energy conformers."""
    def __init__(self, atoms, use_low_energy_conformer=False, **kwargs):
        """
        Initializes the SellaWrapper.
        :param atoms: The ASE Atoms object to be optimized.
        :param use_low_energy_conformer: If True, track and use the lowest
                                         energy conformer found during
                                         optimization.
        :param kwargs: Keyword arguments to be passed to the Sella optimizer.
        """
        self.atoms = atoms
        self.use_low_energy_conformer = use_low_energy_conformer
        self.optimizer = Sella(atoms, **kwargs)
        self.lowest_energy_info = None

        if self.use_low_energy_conformer:
            self.lowest_energy_info = {
                'min_energy': np.inf,
                'positions': self.atoms.get_positions().copy(),
                'step': 0
            }
            self.optimizer.attach(
                make_lowest_energy_callback(self.atoms, self.lowest_energy_info),
                interval=1
            )

    def run(self, *args, **kwargs):
        """
        Run the optimization.
        :param args: Positional arguments for Sella's run method.
        :param kwargs: Keyword arguments for Sella's run method.
        :return: The convergence status from the Sella optimizer.
        """
        converged = self.optimizer.run(*args, **kwargs)

        if self.use_low_energy_conformer:
            # After optimization, set the atoms' positions to the lowest
            # energy conformer found.
            self.atoms.set_positions(self.lowest_energy_info['positions'])
            logging.info(
                "Using lowest energy conformer from step "
                f"{self.lowest_energy_info['step']} with energy "
                f"{self.lowest_energy_info['min_energy']:.6f} eV"
            )

        return converged