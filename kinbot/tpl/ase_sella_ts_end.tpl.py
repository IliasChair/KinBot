"ase_sella_ts_end.tpl.py"
import os
import shutil

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.vibrations import Vibrations
from sella import Sella

from kinbot.constants import EVtoHARTREE
from kinbot.ase_modules.calculators.{code} import {Code}
from kinbot.ase_modules.calculators.nn_pes import Nn_surr
from kinbot.stationary_pt import StationaryPoint
from kinbot.frequencies import get_frequencies
from kinbot.ase_sella_util import calc_vibrations, SellaWrapper

BASE_LABEL = os.path.basename("{label}")
CALC_DIR = os.path.join(os.path.dirname("{label}") or ".",
                        f"{{BASE_LABEL}}_dir")

USE_LOW_FORCE_CONFORMER = os.environ.get("USE_LOW_FORCE_CONFORMER", "false").lower() == "true"
print(f"USE_LOW_FORCE_CONFORMER: {{USE_LOW_FORCE_CONFORMER}}")

FMAX = float(os.environ.get('TS_FMAX', 5e-4))
print(f"using ts_fmax: {{TS_FMAX}}")

db = connect('{working_dir}/kinbot.db')
mol = Atoms(symbols={atom},
            positions={geom})

kwargs = {kwargs}
if "label" not in kwargs:
    kwargs["label"] = "{label}"
mol.calc = Nn_surr()

if os.path.isfile('{label}_sella.log'):
    os.remove('{label}_sella.log')

sella_kwargs = {{'delta0': 0.054701161,
                'eta': 0.000979549,
                'gamma': 0.392400424,
                'rho_dec': 4.819467426,
                'rho_inc': 1.082799457,
                'sigma_dec': 0.827264867,
                'sigma_inc': 1.165201893}}
opt = SellaWrapper(mol,
                   use_low_force_conformer=USE_LOW_FORCE_CONFORMER,
            order=1,
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
        if (np.count_nonzero(np.array(freqs) < 0) > 2  # More than two imag frequencies
                or np.count_nonzero(np.array(freqs) < -50) >= 2  # More than one frequency smaller than 50i
                or np.count_nonzero(np.array(freqs) < 0) == 0):  # No imaginary frequencies
            print(f'Wrong number of imaginary frequencies: {{freqs[6:]}}')
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
            print(f'TS end for {label} converged with energy: {{e}}')
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
