import os
import sys
import shutil

import numpy as np
from ase import Atoms
from ase.db import connect
#from ase.vibrations import Vibrations
from ase.calculators.gaussian import Gaussian
from sella import Sella

#from kinbot.constants import EVtoHARTREE
from kinbot.ase_modules.calculators.{code} import {Code}
#from kinbot.stationary_pt import StationaryPoint
#from kinbot.frequencies import get_frequencies
import cclib
import subprocess

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
    mol.calc = Gaussian(
        mem='8GB',
        nprocshared=4,
        method='HF',
        basis='STO-3G',
        # method='wb97xd',
        # basis='6-31G(d)',
        chk="{label}_vib.chk",
        freq='',
        mult=1,
        extra='SCF=(XQC, MaxCycle=200)'
    )
    mol.calc.label = '{label}_vib'
    # Compute frequencies in a separate temporary directory to avoid
    # conflicts accessing the cache in parallel calculations.
    if not os.path.isdir('{label}_vib'):
        os.mkdir('{label}_vib')
    init_dir = os.getcwd()
    os.chdir('{label}_vib')

    mol.get_potential_energy()

    create_fchk()

    parsed_data = cclib.io.ccopen("{label}_vib.fchk").parse()

    freqs = parsed_data.vibfreqs
    hessian = parsed_data.hessian

    parsed_data = cclib.io.ccopen('{label}_vib.log').parse()
    zpe = parsed_data.zpve
    os.chdir(init_dir)
    #shutil.rmtree('{label}_vib')
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

sella_kwargs = {sella_kwargs}
opt = Sella(mol, order=1,
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
        if (np.count_nonzero(np.array(freqs) < 0) > 2  # More than two imag frequencies
                or np.count_nonzero(np.array(freqs) < -50) >= 2  # More than one frequency smaller than 50i
                or np.count_nonzero(np.array(freqs) < 0) == 0):  # No imaginary frequencies
            print(f'Incorrect number of imaginary frequencies for order=1. '
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
            e = mol.get_potential_energy()
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
    f.write(f"{{converged}}\n")
    f.write(f"energy: {{e}}\n")
    f.write(f"freqs: {{freqs}}\n")
    f.write(f"zpe: {{zpe}}\n")
    f.write(f"hessian: {{hessian is not None}}\n")
