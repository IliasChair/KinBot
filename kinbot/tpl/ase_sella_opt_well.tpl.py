"ase_sella_opt_well.tpl.py"
import os
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import cclib
from ase import Atoms
from ase.db import connect
from ase.vibrations import Vibrations
from sella import Sella

from kinbot.constants import EVtoHARTREE
from kinbot.ase_sella_util import calc_vibrations, SellaWrapper
from kinbot.ase_modules.calculators.{code} import {Code}
from kinbot.ase_modules.calculators.nn_pes import Nn_surr
from ase.calculators.gaussian import Gaussian
from kinbot.stationary_pt import StationaryPoint
from kinbot.frequencies import get_frequencies

BASE_LABEL = os.path.basename("{label}")
CALC_DIR = os.path.join(os.path.dirname("{label}") or ".",
                        f"{{BASE_LABEL}}_dir")

USE_LOW_ENERGY_CONFORMER = os.environ.get('USE_LOW_ENERGY_CONFORMER', 'false').lower() == 'true'
print(f"USE_LOW_ENERGY_CONFORMER: {{USE_LOW_ENERGY_CONFORMER}}")

FMAX = float(os.environ.get('FMAX', 1e-4))
print(f"using fmax: {{FMAX}}")

db = connect('{working_dir}/kinbot.db')
mol = Atoms(symbols={atom},
            positions={geom})

kwargs = {kwargs}
if "label" not in kwargs:
    kwargs["label"] = "{label}"
mol.calc = Nn_surr()

if os.path.isfile('{label}_sella.log'):
    os.remove('{label}_sella.log')

# For monoatomic wells, just calculate the energy and exit.
if len(mol) == 1:
    #e = mol.calc.get_potential_energy_dft(mol, **kwargs)
    e = mol.get_potential_energy()
    print(f'Opt well for {label} converged with energy: {{e}}')
    db.write(mol, name='{label}',
             data={{'energy': e, 'frequencies': np.array([]), 'zpe': 0.0,
                    'hess': np.zeros([3, 3]), 'status': 'normal'}})
    with open('{label}.log', 'a') as f:
        f.write('done\n')
    sys.exit(0)

order = {order}
sella_kwargs = {sella_kwargs}
opt = SellaWrapper(mol,
                   use_low_energy_conformer=USE_LOW_ENERGY_CONFORMER,
                   order=order,
                   trajectory='{label}.traj',
                   logfile='{label}_sella.log',
                   **sella_kwargs)
freqs = []
try:
    converged = False
    fmax = FMAX
    attempts = 1
    steps=500
    while not converged and attempts <= 3:
        mol.calc.label = '{label}'
        converged = opt.run(fmax=fmax, steps=steps)
        freqs, zpe, hessian, dft_energy = calc_vibrations(mol, BASE_LABEL, CALC_DIR)

        if order == 0 and (np.count_nonzero(np.array(freqs) < 0) > 1
                           or np.count_nonzero(np.array(freqs) < -50) >= 1):
            print(f'Found one or more imaginary frequencies. {{freqs[:]}}')
            converged = False
            mol.calc.label = '{label}'
            attempts += 1
            fmax *= 0.3
            if attempts <= 3:
                print(f'Retrying with a tighter criterion: fmax={{fmax}}.')
        elif order == 1 and (np.count_nonzero(np.array(freqs) < 0) > 2  # More than two imag frequencies
                             or np.count_nonzero(np.array(freqs) < -50) >= 2  # More than one imag frequency larger than 50i
                             or np.count_nonzero(np.array(freqs) < 0) == 0):  # No imaginary frequencies
            print(f'Wrong number of imaginary frequencies: {{freqs[:]}}')
            converged = False
            mol.calc.label = '{label}'
            attempts += 1
            fmax *= 0.3
            if attempts <= 3:
                print(f'Retrying with a tighter criterion: fmax={{fmax}}.')
        else:
            converged = True
            #e = mol.calc.get_potential_energy_dft(mol, **kwargs)
            e = mol.get_potential_energy()
            print(f'Opt well for {label} converged with energy: {{e}}')
            db.write(mol, name='{label}',
                     data={{'energy': e, 'frequencies': freqs, 'zpe': zpe,
                            'hess': hessian, 'status': 'normal'}})
    if not converged:
        raise RuntimeError
except (RuntimeError, ValueError):
    data = {{'status': 'error'}}
    if freqs is not None and np.size(freqs) > 0:
        data['frequencies'] = freqs
    db.write(mol, name='{label}', data=data)

with open('{label}.log', 'a') as f:
    f.write('done\n')
