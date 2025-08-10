"ase_sella_ts_search.tpl.py"
import os

from ase import Atoms
from ase.db import connect
from sella import Sella, Constraints
from kinbot.ase_modules.calculators.nn_pes import Nn_surr
from kinbot.ase_modules.calculators.{code} import {Code}

db = connect('{working_dir}/kinbot.db')
mol = Atoms(symbols={atom},
            positions={geom})

const = Constraints(mol)
base_0_fix = [[idx - 1 for idx in fix] for fix in {fix}]
for fix in base_0_fix:
    if len(fix) == 2:
        const.fix_bond(fix)
    elif len(fix) == 3:
        const.fix_angle(fix)
    elif len(fix) == 4:
        const.fix_dihedral(fix)
    else:
        raise ValueError(f'Unexpected length of fix: {{fix}}')

kwargs = {kwargs}
if "label" not in kwargs:
    kwargs["label"] = "{label}"
mol.calc = Nn_surr()

if os.path.isfile('{label}_sella.log'):
    os.remove('{label}_sella.log')

sella_kwargs = {sella_kwargs}
opt = Sella(mol,
            order=0,
            constraints=const,
            trajectory='{label}.traj',
            logfile='{label}_sella.log',
            **sella_kwargs)
try:
    cvgd = opt.run(fmax=0.1, steps=300)
    if cvgd:
        #e = mol.calc.get_potential_energy_dft(mol, **kwargs)
        e = mol.get_potential_energy()
        print(f'TS search for {label} converged with energy: {{e}}')
    else:  # TODO Eventually we might want to correct something in case it fails.
        raise RuntimeError
except (RuntimeError, ValueError):
    e = 0.0

if not mol.positions.any():  # If all coordinates are 0
    mol.positions = {geom}   # Reset to the original geometry
db.write(mol, name='{label}', data={{'energy': e, 'status': 'normal'}})

with open('{label}.log', 'a') as f:
    f.write('done\n')