import os
import sys
import shutil

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.db import connect
#from ase.vibrations import Vibrations
from ase.calculators.gaussian import Gaussian
from sella import Sella

from kinbot.constants import EVtoHARTREE
from kinbot.ase_modules.calculators.{code} import {Code}
#from kinbot.stationary_pt import StationaryPoint
#from kinbot.frequencies import get_frequencies
import cclib
import subprocess

def get_valid_multiplicity(atoms: Atoms, default_multiplicity: int = 1):
    """
    Ensures a valid spin multiplicity for an ASE Atoms object to prevent errors in Gaussian.

    Parameters:
    atoms (Atoms): ASE Atoms object
    default_multiplicity (int): Initial guess for multiplicity

    Returns:
    int: Corrected multiplicity
    """
    # Compute the total number of electrons
    total_electrons = sum(atomic_numbers[atom.symbol] for atom in atoms)

    # Ensure multiplicity is valid
    if total_electrons % 2 == 0:
        # Even electron count -> Multiplicity must be odd (1, 3, 5, ...)
        if default_multiplicity % 2 == 0:
            default_multiplicity += 1
    else:
        # Odd electron count -> Multiplicity must be even (2, 4, 6, ...)
        if default_multiplicity % 2 == 1:
            default_multiplicity += 1

    return default_multiplicity


def create_fchk():
    """
    Run Gaussian's formchk utility to create a formatted checkpoint file.

    :param label: Base label used for the checkpoint filename.
    :type label: str
    """
    chkfile = "{label}_vib.chk"
    fchkfile = "{label}_vib.fchk"
    subprocess.run(["formchk", chkfile, fchkfile], check=True)

def calc_vibrations(mol):
    mol = mol.copy()
    init_dir = os.getcwd()

    # First attempt
    try:
        mol.calc = Gaussian(
            mem='8GB',
            nprocshared=4,
            method='wb97xd',
            basis='6-31G(d)',
            chk="{label}_vib.chk",
            freq='',
            extra='SCF=(XQC, MaxCycle=200)'
        )
        mol.calc.label = '{label}_vib'

        # Create/change to temporary directory
        if not os.path.isdir('{label}_vib'):
            os.mkdir('{label}_vib')
        os.chdir('{label}_vib')

        e = mol.get_potential_energy()

    except RuntimeError as e:
        print(f"initial gaussian calculation failed in {label}")
        # Return to initial directory before second attempt
        os.chdir(init_dir)

        # Second attempt with corrected multiplicity
        mult = get_valid_multiplicity(mol)
        print(f"trying again witl multiplicity {{mult}}")
        mol.calc = Gaussian(
            mem='8GB',
            nprocshared=4,
            method='wb97xd',
            basis='6-31G(d)',
            chk="{label}_vib.chk",
            freq='',
            mult=mult,  # Use corrected multiplicity
            extra='SCF=(XQC, MaxCycle=200)'
        )
        mol.calc.label = '{label}_vib'

        # Change back to temporary directory
        os.chdir('{label}_vib')
        e = mol.get_potential_energy()

    # Rest of the function remains the same
    create_fchk()
    parsed_data = cclib.io.ccopen("{label}_vib.fchk").parse()
    freqs = parsed_data.vibfreqs
    hessian = parsed_data.hessian

    parsed_data = cclib.io.ccopen('{label}_vib.log').parse()
    zpe = parsed_data.zpve

    # Return to initial directory
    os.chdir(init_dir)
    return freqs, zpe, hessian

db = connect('{working_dir}/kinbot.db')
mol = Atoms(symbols={atom},
            positions={geom})

kwargs = {kwargs}
mol.calc = {Code}(**kwargs)
if '{Code}' == 'Gaussian':
    mol.get_potential_energy()
    kwargs['guess'] = 'Read'
    mol.calc = {Code}(**kwargs)

if os.path.isfile('{label}_sella.log'):
    os.remove('{label}_sella.log')

# For monoatomic wells, just calculate the energy and exit.
if len(mol) == 1:
    e = mol.get_potential_energy() * EVtoHARTREE
    db.write(mol, name='{label}',
             data={{'energy': e, 'frequencies': np.array([]), 'zpe': 0.0,
                    'hess': np.zeros([3, 3]), 'status': 'normal'}})
    with open('{label}.log', 'a') as f:
        f.write('done\n')
    sys.exit(0)

order = {order}
sella_kwargs = {sella_kwargs}
opt = Sella(mol,
            order=order,
            trajectory='{label}.traj',
            logfile='{label}_sella.log',
            **sella_kwargs)
freqs = []
try:
    converged = False
    fmax = 1e-4
    attempts = 1
    steps=500
    while not converged and attempts <= 3:
        mol.calc.label = '{label}'
        converged = opt.run(fmax=fmax, steps=steps)
        freqs, zpe, hessian = calc_vibrations(mol)
        if order == 0 and (np.count_nonzero(np.array(freqs) < 0) > 1
                           or np.count_nonzero(np.array(freqs) < -50) >= 1):
            print(f'Invalid frequencies for minimum (order={order}). '
                  f'Expected no imaginary frequencies, but found '
                  f'{{np.count_nonzero(np.array(freqs) < 0)}}. '
                  f'frequencies: {{freqs}}')
            converged = False
            mol.calc.label = '{label}'
            attempts += 1
            fmax *= 0.3
            if attempts <= 3:
                print(f'Retrying with a tighter criterion: fmax={{fmax}}.')
        elif order == 1 and (np.count_nonzero(np.array(freqs) < 0) > 2  # More than two imag frequencies
                             or np.count_nonzero(np.array(freqs) < -50) >= 2  # More than one imag frequency larger than 50i
                             or np.count_nonzero(np.array(freqs) < 0) == 0):  # No imaginary frequencies
            print(f'Incorrect number of imaginary frequencies for order={order}. '
                  f'Expected exactly 1 imaginary frequency, but found '
                  f'{{np.count_nonzero(np.array(freqs) < 0)}}. Frequencies: {{freqs}}')
            converged = False
            mol.calc.label = '{label}'
            attempts += 1
            fmax *= 0.3
            if attempts <= 3:
                print(f'Retrying with a tighter criterion: fmax={{fmax}}.')
        else:
            converged = True
            print(f"Converged, with frequencies: {{freqs}}")
            e = mol.get_potential_energy() * EVtoHARTREE
            db.write(mol, name='{label}',
                     data={{'energy': e, 'frequencies': freqs, 'zpe': zpe,
                            'hess': hessian, 'status': 'normal'}})
            print(f'Wrote {label} to database')
    if not converged:
        raise RuntimeError
except (RuntimeError, ValueError):
    data = {{'status': 'error'}}
    if freqs is not None:
        data['frequencies'] = freqs
    db.write(mol, name='{label}', data=data)

with open('{label}.log', 'a') as f:
    f.write('done\n')
    f.write("########################################")
    f.write(f"converged: {{converged}}\n")
    f.write(f"energy: {{e}}, zpe: {{zpe}}, hessian: {{hessian is not None}}\n")
    f.write(f"freqs: {{freqs}}\n")
    f.write("########################################")
    f.write('done\n')
