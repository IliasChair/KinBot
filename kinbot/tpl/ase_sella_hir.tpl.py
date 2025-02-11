import os
import logging
import traceback
from datetime import datetime

from ase import Atoms
from ase.db import connect
from sella import Sella, Constraints

from kinbot.ase_modules.calculators.{code} import {Code}

db = connect('{working_dir}/kinbot.db')
mol = Atoms(symbols={atom},
            positions={geom})

const = Constraints(mol)
base_0_fix = [idx - 1 for idx in {fix}]
const.fix_dihedral(base_0_fix)

kwargs = {kwargs}
mol.calc = {Code}(**kwargs)
if '{Code}' == 'Gaussian':
    mol.get_potential_energy()
    kwargs['guess'] = 'Read'
    mol.calc = {Code}(**kwargs)

# Configure logging
logging.basicConfig(
    filename='{label}_hir_detailed.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Log initial configuration
    logger.info(f"Starting optimization for {label}")
    logger.info(f"Configuration: order={order}, constraints={{const}}")

    if os.path.isfile('{label}_sella.log'):
        os.remove('{label}_sella.log')
        logger.debug("Removed existing sella log file")

    sella_kwargs = {sella_kwargs}
    opt = Sella(mol,
                order={order},
                constraints=const,
                trajectory='{label}.traj',
                logfile='{label}_sella.log',
                **sella_kwargs)

    converged = opt.run(fmax=0.001, steps=500)

    if converged:
        e = mol.get_potential_energy()
        db.write(mol, name='{label}', data={{'energy': e, 'status': 'normal'}})
        logger.info(f"Successfully converged. Final energy: {{e}}")
    else:  # TODO Eventually we might want to correct something in case it fails.
        error_msg = "Hindered Rotor Optimization failed to converge within 300 steps"
        logger.error(error_msg)
        db.write(mol, name='{label}',
                data={{'status': 'error', 'error_type': 'convergence_failure'}})
        raise RuntimeError(error_msg)

except (RuntimeError, ValueError) as e:
    error_details = {{
        'error_type': type(e).__name__,
        'error_message': str(e),
        'traceback': traceback.format_exc()
    }}
    logger.error(f"Error occurred: {{error_details['error_type']}}")
    logger.error(f"Error message: {{error_details['error_message']}}")
    logger.error(f"Traceback:\n{{error_details['traceback']}}")

    db.write(mol, name='{label}', data={{'status': 'error'}})

except Exception as e:
    # Catch any unexpected errors
    logger.critical(f"Unexpected error: {{type(e).__name__}}: {{str(e)}}")
    logger.critical(traceback.format_exc())
    db.write(mol, name='{label}', data={{'status': 'error'}})
    raise

finally:
    # Always write completion status to log file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('{label}.log', 'a') as f:
        f.write('done\n')
        f.write('########################\n')
        f.write(f"Timestamp: {{timestamp}}\n")
        f.write('done\n')
        f.write(f"converged: {{converged if 'converged' in locals() else 'N/A'}}\n")
        f.write('########################\n')
        f.write('done\n')