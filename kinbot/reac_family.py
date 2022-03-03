import os
import numpy as np
import copy
import time
import pkg_resources
from kinbot import modify_geom
from kinbot import geometry
from reactions.reac_abstraction import abstraction_align


def carry_out_reaction(rxn, step, command, bimol=0):
    """
    Verify what has been done and what needs to be done
    skip: boolean which tells to skip the first 12 steps in case of an instance shorter than 4
    scan: boolean which tells if this is part of an energy scan along a bond length coordinate
    """
    if step > 0:
        status = rxn.qc.check_qc(rxn.instance_name)
        if status != 'normal' and status != 'error': return step
  
    kwargs = rxn.qc.get_qc_arguments(rxn.instance_name, rxn.species.mult, rxn.species.charge, ts=1,
                                     step=step, max_step=rxn.max_step, scan=rxn.scan)
    if step == 0:
        if rxn.qc.is_in_database(rxn.instance_name):
            if rxn.qc.check_qc(rxn.instance_name) == 'normal': 
                err, freq = rxn.qc.get_qc_freq(rxn.instance_name, rxn.species.natom)
                if err == 0 and len(freq) > 0.:
                    err, geom = rxn.qc.get_qc_geom(rxn.instance_name, rxn.species.natom)
                    step = rxn.max_step + 1
                    return step
        if rxn.skip and len(rxn.instance) < 4:
            step = 12
        geom = rxn.species.geom
        if bimol:
            if rxn.family_name == 'abstraction':
                # gives the reactant and product geometry guesses
                geom, _, _ = abstraction_align(rxn.species.geom, rxn.instance, rxn.species.atom, rxn.species.fragA.natom)

    elif step == rxn.max_step and rxn.scan:
        err, geom = rxn.qc.get_qc_geom(rxn.instance_name, rxn.species.natom, allow_error=1, previous=1)
    else:
        err, geom = rxn.qc.get_qc_geom(rxn.instance_name, rxn.species.natom, allow_error=1)
        if bimol:
            if rxn.family_name == 'abstraction':
                # gives the reactant and product geometry guesses
                _, geom_prod, geom_ts = abstraction_align(geom, rxn.instance, rxn.species.atom, rxn.species.fragA.natom)


    step, fix, change, release = rxn.get_constraints(step, geom)

    if step > rxn.max_step:
        return step
    
    #apply the geometry changes here and fix the coordinates that changed
    change_starting_zero = []
    for c in change:
        c_new = [ci - 1 for ci in c[:-1]]
        c_new.append(c[-1])
        change_starting_zero.append(c_new)
    if len(change_starting_zero) > 0:
        success, geom = modify_geom.modify_coordinates(rxn.species, rxn.instance_name, geom, change_starting_zero, rxn.species.bond)
        for c in change:
            fix.append(c[:-1])
        change = []

    #atom, geom, dummy = rxn.qc.add_dummy(rxn.species.atom, geom, rxn.species.bond)

    kwargs['addsec'] = ''
    if not bimol or step == 0:
        # here addsec contains the constraints
        for fixi in fix:
            kwargs['addsec'] += f"{' '.join(str(f) for f in fixi)} F\n"
        for chi in change:
            kwargs['addsec'] += f"{' '.join(str(ch) for ch in changei)} F\n"
        for reli in release:
            kwargs['addsec'] += f"{' '.join(str(rel) for rel in reli)} A\n"
    elif bimol and step == 1:
        kwargs['addsec'] = f'{rxn.instance[0] + 1} {rxn.instance[2] + 1}\n\n'
        # here addsec needs to contain the product and ts geometries and all the rest of the fluff
        kwargs['addsec'] += f'product geometry guess\n\n{rxn.species.charge} {rxn.species.mult}\n'
        for ii, at in enumerate(rxn.species.atom):
            kwargs['addsec'] += f'{at} {geom_prod[ii][0]} {geom_prod[ii][1]} {geom_prod[ii][2]}\n'
        kwargs['addsec'] += f'\n{rxn.instance[0] + 1} {rxn.instance[2] + 1}\n\n'
        kwargs['addsec'] += f'ts geometry guess\n\n{rxn.species.charge} {rxn.species.mult}\n'
        for ii, at in enumerate(rxn.species.atom):
            kwargs['addsec'] += f'{at} {geom_ts[ii][0]} {geom_ts[ii][1]} {geom_ts[ii][2]}\n'
        kwargs['addsec'] += f'\n{rxn.instance[0] + 1} {rxn.instance[2] + 1}\n\n'
    if not bimol:
        ntrial = 3
    else:
        ntrial = 1

    if step < rxn.max_step:
        template_file = pkg_resources.resource_filename('tpl', 'ase_{qc}_ts_search.tpl.py'.format(qc=rxn.qc.qc))
        template = open(template_file,'r').read()
        template = template.format(label=rxn.instance_name, 
                                   kwargs=kwargs, 
                                   #atom=list(atom),
                                   atom=list(rxn.species.atom),
                                   geom=list([list(gi) for gi in geom]), 
                                   #dummy=dummy,
                                   bimol=bimol,
                                   ppn=rxn.qc.ppn,
                                   qc_command=command,
                                   working_dir=os.getcwd(),
                                   scan=rxn.scan,
                                   ntrial=ntrial,
                                   )
    else:
        template_file = pkg_resources.resource_filename('tpl', 'ase_{qc}_ts_end.tpl.py'.format(qc=rxn.qc.qc))
        template = open(template_file,'r').read()
    
        template = template.format(label=rxn.instance_name, 
                                   kwargs=kwargs, 
                                   #atom=list(atom),
                                   atom=list(rxn.species.atom),
                                   geom=list([list(gi) for gi in geom]), 
                                   #dummy=dummy,
                                   ppn=rxn.qc.ppn,
                                   qc_command=command,
                                   working_dir=os.getcwd())
                                   
    with open('{}.py'.format(rxn.instance_name),'w') as f_out:
        f_out.write(template)
    
    step += rxn.qc.submit_qc(rxn.instance_name, singlejob=0)

    return step
