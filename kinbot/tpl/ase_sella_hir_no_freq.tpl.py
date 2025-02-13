"""
ASE-Sella Hindered Rotor Optimization template for KinBot.

This template performs hindered rotor optimization using ASE and Sella.
It sets up the molecular system, applies dihedral (hindered rotor)
constraints, configures the calculator, and performs optimization.
All events are logged and the results are stored in a database.

Template variables:
    {{label}}         - Calculation identifier (may include subdirectory)
    {{working_dir}}   - Working directory path
    {{atom}}          - Atomic symbols list
    {{geom}}          - Atomic coordinates list
    {{code}}          - Calculator module name
    {{Code}}          - Calculator class name
    {{kwargs}}        - Calculator keyword arguments
    {{order}}         - Optimization order (0 for minimum, 1 for TS)
    {{sella_kwargs}}  - Sella optimizer parameters
    {{fix}}           - Dihedral indices to fix (1-indexed)
"""

import os
import logging
import traceback

from ase import Atoms
from ase.db import connect
from sella import Sella, Constraints

from kinbot.constants import EVtoHARTREE
from kinbot.ase_modules.calculators.{code} import {Code}

# Determine the full label and extract the basename. If the provided
# label includes a directory, BASE_LABEL will be only the final component.
LABEL = "{label}"
BASE_LABEL = os.path.basename(LABEL)
# Build the calculation directory path. If LABEL includes a directory,
# we join that dirname with the base identifier directory.
CALC_DIR = os.path.join(os.path.dirname(LABEL) or ".",
                         f"{{BASE_LABEL}}_hir_dir")

FMAX = 0.001
DMAX = 0.1
STEPS = 500


def main():
    """
    Main routine for hindered rotor optimization.

    Constructs the molecular system with the provided atomic data,
    applies dihedral constraints, configures the calculator, and then
    runs the Sella optimizer. Results are logged and stored in the
    database.

    Raises:
        RuntimeError: If the optimization fails (e.g. non-convergence
                      or calculator error).
    """
    # Configure logging
    logging.basicConfig(
        filename=os.path.join(
            CALC_DIR, f"{{BASE_LABEL}}_hir_detailed.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    db = connect("{working_dir}/kinbot.db")
    mol = Atoms(symbols={atom}, positions={geom})
    converged = False
    # Apply dihedral constraints.
    const = Constraints(mol)
    fix_indices = [idx - 1 for idx in {fix}]
    const.fix_dihedral(fix_indices)

    error_free_exec = False
    try:
        # Setup initial calculator
        kwargs = {kwargs}
        mol.calc = {Code}(**kwargs)
        if "{Code}" == "Gaussian":
            mol.get_potential_energy()
            kwargs["guess"] = "Read"
            mol.calc = {Code}(**kwargs)

        mol.calc.label = "{label}"
        # Log initial configuration.
        logger.info("Starting hindered rotor optimization for {label}")
        logger.info(f"Configuration: order={order}, constraints={{const}}")

        sella_log = os.path.join(CALC_DIR, f"{{BASE_LABEL}}_sella.log")
        sella_traj = os.path.join(CALC_DIR, f"{{BASE_LABEL}}_sella.traj")
        if os.path.isfile(sella_log):
            os.remove(sella_log)

        opt = Sella(mol,
                    order={order},
                    constraints=const,
                    trajectory=sella_traj,
                    logfile=sella_log,
                    **{sella_kwargs})

        converged = opt.run(fmax=FMAX, steps=STEPS)

        if converged:
            e = mol.get_potential_energy() * EVtoHARTREE
            db.write(mol, name="{label}",
                     data={{"energy": e, "status": "normal"}})
            logger.info(("Hindered Rotor Optimization successfully converged. "
                        f"for {label}. Final energy: {{e}}"))
            error_free_exec = True
        else:
            raise RuntimeError(("Hindered Rotor Optimization failed to "
                            f"converge for {label}. within {{STEPS}} steps"))

    except Exception as err:
        logger.error(f"Hindered Rotor Optimization failed: {{err}}")
        data = {{"status": "error"}}
        db.write(mol, name="{label}", data=data)
        error_free_exec = False
        error_details = {{
            "error_type": type(err).__name__,
            "error_message": str(err),
            "traceback": traceback.format_exc()
        }}
        logger.error("Error occurred: %s", error_details["error_type"])
        logger.error("Error message: %s", error_details["error_message"])
        logger.error("Traceback:\n%s", error_details["traceback"])

    finally:
        # Build termination report.
        BOX_WIDTH = 79

        # Determine final status; non-convergence is considered failure.
        final_status = "SUCCESS ✔" if error_free_exec else "FAILURE ✘"

        # Build the report lines using formatted strings.
        report_lines = [
            "Hindered Rotor Optimization Termination Report",
            f"Status:         {{final_status}}",
            f"Converged:      {{converged}}",
            f"Error-free:     {{error_free_exec}}",
            f"order:          {order}"
        ]
        if error_free_exec and ("e" in locals()):
            report_lines.extend([
                f"Energy:         {{e}}",
            ])

        # Build the bounding box using fixed BOX_WIDTH.
        top_border = f"╔{{'═' * (BOX_WIDTH + 2)}}╗"
        bottom_border = f"╚{{'═' * (BOX_WIDTH + 2)}}╝"
        body = "\n".join(f"║ {{line.ljust(BOX_WIDTH)}} ║"
                         for line in report_lines)
        report_box = f"{{top_border}}\n{{body}}\n{{bottom_border}}"

        # Send the termination message to the logger and write it to file.
        logger.info("\n" + report_box)

        # Write the termination message to the log file.
        with open("{label}.log", "a") as f:
            f.write("\n" + report_box + "\n")
            f.write("done\n")


if __name__ == "__main__":
    if not os.path.exists(CALC_DIR):
        os.makedirs(CALC_DIR)
    main()