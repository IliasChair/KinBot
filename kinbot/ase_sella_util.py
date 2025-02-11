from ase import Atoms
from ase.data import atomic_numbers
import numpy as np


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
          (-50 < freq < 0) to account for numerical noise.
        * Does not allow any significantly imaginary frequencies
          (freq < -50).
        * In other words, returns False if either:
            - Two or more frequencies are less than 0 (n_imag >= 2), or
            - Any frequency is less than -50 (n_large_imag >= 1).

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
        Small imaginary frequencies (between 0 and -50 cm^-1) often appear
        in quantum chemistry calculations due to numerical imprecision,
        particularly for large, floppy molecules or loose convergence
        criteria. These are typically not chemically meaningful and can be
        safely ignored. However, frequencies below -50 cm^-1 usually
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
    n_large_imag = np.count_nonzero(freqs < -50)

    if order == 0:
        return not (n_imag >= 2 or n_large_imag >= 1)
    elif order == 1:
        return not (n_imag > 2 or n_large_imag >= 2 or n_imag == 0)
    return False