import os
import sys
import shutil

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.vibrations import Vibrations
from sella import Sella, Constraints

from kinbot.constants import EVtoHARTREE
from kinbot.modify_geom import modify_coordinates
from kinbot.ase_modules.calculators.{code} import {Code}
from kinbot.stationary_pt import StationaryPoint
from kinbot.frequencies import get_frequencies

def calc_vibrations(mol):
        mol.calc.label = '{label}_vib'
        if 'chk' in mol.calc.parameters:
            del mol.calc.parameters['chk']
        # Compute frequencies in a separate temporary directory to avoid 
        # conflicts accessing the cache in parallel calculations.
        if not os.path.isdir('{label}_vib'):
            os.mkdir('{label}_vib')
        init_dir = os.getcwd()
        os.chdir('{label}_vib')
        if os.path.isdir('vib'):
            shutil.rmtree('vib')
        vib = Vibrations(mol)
        vib.run()
        # Use kinbot frequencies to avoid mixing low vib frequencies with 
        # the values associated with external rotations.
        _ = vib.get_frequencies()
        zpe = vib.get_zero_point_energy() * EVtoHARTREE
        hessian = vib.H / 97.17370087
        st_pt = StationaryPoint.from_ase_atoms(mol)
        st_pt.characterize()
        freqs, _ = get_frequencies(st_pt, hessian, st_pt.geom)
        os.chdir(init_dir)
        shutil.rmtree('{label}_vib')
        return freqs, zpe, hessian

db = connect('{working_dir}/kinbot.db')
mol = Atoms(symbols={atom}, 
            positions={geom})

kwargs = {kwargs}
mol.calc = {Code}(**kwargs)

const = Constraints(mol)
base_0_fixes = [[idx - 1 for idx in fix] for fix in {fix}]
for fix in base_0_fixes:
    if len(fix) == 2:
        const.fix_bond(fix)
    elif len(fix) == 3:
        const.fix_angle(fix)
    elif len(fix) == 4:
        const.fix_dihedral(fix)
    else:
        raise ValueError(f'Unexpected length of fix: {{fix}}.')

st_pt = StationaryPoint.from_ase_atoms(mol)
st_pt.characterize()
base_0_changes = []
for c in {change}:
    c_new = [ci - 1 for ci in c[:-1]]
    c_new.append(c[-1])
    base_0_changes.append(c_new)
if len(base_0_changes) > 0:
    _, mol.positions = modify_coordinates(st_pt, '{label}', mol.positions, 
                                           base_0_changes, st_pt.bond)
for change in base_0_changes:
    if len(change[:-1]) == 2:
        const.fix_bond(change[:-1])
    elif len(change[:-1]) == 3:
        const.fix_angle(change[:-1])
    elif len(change[:-1]) == 4:
        const.fix_dihedral(change[:-1])
    else:
        raise ValueError(f'Unexpected length of changes: {{change}}.')

if os.path.isfile('{label}_sella.log'):
    os.remove('{label}_sella.log')

order = {order}
opt  = Sella(mol, order=order, trajectory='{label}.traj', 
             logfile='{label}_sella.log')
try:
    converged = False
    fmax = 1e-4
    attempts = 1
    steps=300
    while not converged and attempts <= 3:
        converged = opt.run(fmax=fmax, steps=300)
        freqs, zpe, hessian = calc_vibrations(mol)
        if not converged:
            steps += 100
            attempts += 1
            print(f"Convergence not found in {{steps - 100}} steps. Retrying " \
                  f"with {{steps}} steps.")
        elif order == 0 and any([fr < -50 for fr in freqs]):
            converged == False
            mol.calc.label = '{label}'
            attempts += 1
            fmax *= 0.3
            print("Found one or more imaginary frequencies. Retrying with a " \
                  f"tighter criterion: fmax={{fmax}}.")
        elif order == 1 and np.count_nonzero([fr < -50 for fr in freqs]) > 1:
            converged == False
            mol.calc.label = '{label}'
            attempts += 1
            fmax *= 0.3
            print("Found more than one imaginary frequency. Retrying with a " \
                  f"tighter criterion: fmax={{fmax}}.")
        else:
            e = mol.get_potential_energy()
            db.write(mol, name='{label}', 
                     data={{'energy': e, 'frequencies': freqs, 'zpe': zpe, 
                        'hess': hessian, 'status': 'normal'}})
    if not converged:
        raise RuntimeError
except RuntimeError:
    db.write(mol, name='{label}', data={{'status': 'error'}})

with open('{label}.log', 'a') as f:
    f.write('done\n')
