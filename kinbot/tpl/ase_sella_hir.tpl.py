"ase_sella_hir.tpl.py"
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
base_0_fix = [idx - 1 for idx in {fix}]
const.fix_dihedral(base_0_fix)

kwargs = {kwargs}
if "label" not in kwargs:
    kwargs["label"] = "{label}"
mol.calc = Nn_surr()

if os.path.isfile('{label}_sella.log'):
    os.remove('{label}_sella.log')

sella_kwargs = {sella_kwargs}
opt = Sella(mol,
            order={order},
            constraints=const,
            trajectory='{label}.traj',
            logfile='{label}_sella.log',
            **sella_kwargs)
try:
    converged = opt.run(fmax=0.001, steps=300)
    if converged:
        #e = mol.calc.get_potential_energy_dft(mol, **kwargs)
        e = mol.get_potential_energy()
        print(f'Hir for {label} converged with energy: {{e}}')
        db.write(mol, name='{label}', data={{'energy': e, 'status': 'normal'}})
    else:  # TODO Eventually we might want to correct something in case it fails.
        raise RuntimeError
except (RuntimeError, ValueError):
    db.write(mol, name='{label}', data={{'status': 'error'}})

with open('{label}.log', 'a') as f:
    f.write('done\n')
