"ase_sella_ring_conf.tpl.py"
import os
import sys
import shutil

import numpy as np
from ase import Atoms
from ase.db import connect
from sella import Sella, Constraints
from kinbot.ase_modules.calculators.nn_pes import Nn_surr
from kinbot.ase_modules.calculators.{code} import {Code}
from kinbot.stationary_pt import StationaryPoint
from kinbot.ase_sella_util import SellaWrapper


USE_LOW_ENERGY_CONFORMER = os.environ.get("USE_LOW_ENERGY_CONFORMER", "False")

db = connect('{working_dir}/kinbot.db')
mol = Atoms(symbols={atom},
            positions={geom})

kwargs = {kwargs}
if "label" not in kwargs:
    kwargs["label"] = "{label}"
mol.calc = Nn_surr()
const = Constraints(mol)
fix_these = [[idx - 1 for idx in fix] for fix in {fix}]
for fix in fix_these:
    if len(fix) == 2:
        const.fix_bond(fix)
    elif len(fix) == 4:
        const.fix_dihedral(fix)
    else:
        raise ValueError(f'Unexpected length of fix: {{fix}}.')

for c in {change}:
    const.fix_dihedral((c[0]-1, c[1]-1, c[2]-1, c[3]-1), target=c[4])

if os.path.isfile('{label}_sella.log'):
    os.remove('{label}_sella.log')

sella_kwargs = {sella_kwargs}
opt = SellaWrapper(mol,
                   use_low_energy_conformer=USE_LOW_ENERGY_CONFORMER,
                   order=0,
                   constraints=const,
                   trajectory='{label}.traj',
                   logfile='{label}_sella.log',
                   **sella_kwargs,
                   )

try:
    mol.calc.label = '{label}'
    opt.run(fmax=1e-4, steps=100)
    #e = mol.calc.get_potential_energy_dft(mol, **kwargs)
    e = mol.get_potential_energy()
    print(f'Opt well for {label} converged with energy: {{e}}')
    db.write(mol, name='{label}',
             data={{'energy': e, 'status': 'normal'}})
except (RuntimeError, ValueError):
    data = {{'status': 'error'}}
    db.write(mol, name='{label}', data=data)
with open('{label}.log', 'a') as f:
    f.write('done\n')
