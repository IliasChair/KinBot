# ruff: noqa: E501
"""
ASE-Sella IRC Optimization template for KinBot.

This template performs intrinsic reaction coordinate (IRC)
optimization using ASE and Sella. It sets up the molecular system,
configures the calculator, and runs the IRC. If the IRC run is
successful, a subsequent product optimization is performed.
Results are logged and stored in the database.

Template variables:
    {{label}}         - Calculation identifier (may include subdirectory)
    {{working_dir}}   - Working directory path
    {{atom}}          - Atomic symbols list
    {{geom}}          - Atomic coordinates list
    {{code}}          - Calculator module name
    {{Code}}          - Calculator class name
    {{kwargs}}        - Calculator keyword arguments for IRC step
    {{prod_kwargs}}   - Calculator keyword arguments for product step
    {{sella_kwargs}}  - Sella optimizer parameters
"""

import os
import logging
import traceback

from ase import Atoms
from ase.db import connect
from sella import Sella, IRC

from kinbot.constants import EVtoHARTREE
from kinbot.ase_modules.calculators.{code} import {Code}
from kinbot.ase_modules.calculators.nn_pes import (
    make_rms_callback, RMSConvergence)
from kinbot.ase_sella_util import validate_frequencies, calc_vibrations

# Determine the full label and extract the basename. If the provided
# label includes a directory, BASE_LABEL will be only the final component.
LABEL = "{label}"
BASE_LABEL = os.path.basename(LABEL)
# Build the calculation directory path. If LABEL includes a directory,
# we join that dirname with the base identifier directory.
CALC_DIR = os.path.join(os.path.dirname(LABEL) or ".",
                         f"{{BASE_LABEL}}_irc_dir")

FMAXIRC = 0.01    # IRC convergence threshold (suggested: 0.01 or 0.0004)
STEPSIRC = 300

FMAXOPT = 0.0001
STEPSOPT = 300
RMS_THRESHOPT = 0.00002  # RMS displacement threshold (Å)


def get_direction() -> str:
    """
    Determine the direction of the IRC scan based on the label.
    """
    if "{label}".endswith("F"):
        return "forward"
    elif "{label}".endswith("R"):
        return "reverse"
    else:
        raise ValueError("Unexpected IRC name: {label}.")


def run_irc(mol, logger, db):
    """
    Run the IRC optimization.
    """

    # Set up the initial calculator for the IRC step.
    kwargs = {kwargs}
    mol.calc = {Code}(**kwargs)
    if "{Code}" == "Gaussian":
        mol.get_potential_energy()  # what was this used for?
        kwargs["guess"] = "Read"
        mol.calc = {Code}(**kwargs)

    # Define IRC log file path and remove any pre‐existing log.
    irc_logfile = os.path.join(CALC_DIR, f"{{BASE_LABEL}}_irc.log")
    irc_traj = os.path.join(CALC_DIR, f"{{BASE_LABEL}}_irc.traj")
    if os.path.isfile(irc_logfile):
        os.remove(irc_logfile)

    irc = IRC(mol,
              dx=0.1,
              eta=1e-4,
              gamma=0,
              ninner_iter=200,
              trajectory=irc_traj,
              logfile=irc_logfile)

    irc_converged = False
    irc_success = False  # e.g. error free execution
    try:
        irc_converged = irc.run(fmax=FMAXIRC,
                                steps=STEPSIRC,
                                direction=get_direction())
        if irc_converged:
            e = mol.get_potential_energy("gaussian") * EVtoHARTREE
            db.write(mol, name="{label}",
                     data={{'energy': e, 'status': 'normal'}})
            irc_success = True
            logger.info(
                f"IRC converged for {label}. Energy: {{e}} Hartree")
        elif mol.positions is not None and mol.positions.any():
            db.write(mol, name="{label}",
                     data={{"status": "normal"}})
            irc_success = True
            logger.warning(
                "IRC did not meet convergence criteria, but positions are "
                "valid for {label}")
        else:
            raise RuntimeError("IRC did not converge and no valid positions.")
    except Exception as err:
        logger.error(f"IRC scan failed: {{err}}")
        db.write(mol, name='{label}', data={{'status': 'error'}})
        error_details = {{
            "error_type": type(err).__name__,
            "error_message": str(err),
            "traceback": traceback.format_exc()
        }}
        logger.error("Error occurred: %s", error_details["error_type"])
        logger.error("Error message: %s", error_details["error_message"])
        logger.error("Traceback:\n%s", error_details["traceback"])

    # Write termination signal to the IRC log.
    with open(os.path.join(CALC_DIR, "{label}.log"), "a") as f:
        f.write(f"irc_converged: {{irc_converged}}\n")
        f.write(f"irc_success: {{irc_success}}\n")
        f.write("done\n")

    return irc_converged, irc_success


def irc_end_point_optimization(mol, logger, db):
    """
    Perform end point optimization of the IRC product.
    """
    dft_energy, freqs, zpe = (None,) * 3
    rms_threshold = [RMS_THRESHOPT]  # mutable
    converged_opt = False
    converged_freqs = False
    converged_rms = False
    error_free_exec = False
    try:
        prod_kwargs = {prod_kwargs}
        mol.calc = {Code}(**prod_kwargs)
        if "{Code}" == "Gaussian":
            mol.get_potential_energy()  # Initial energy call.
            prod_kwargs["guess"] = "Read"
            mol.calc = {Code}(**prod_kwargs)

        prod_traj = os.path.join(CALC_DIR, f"{{BASE_LABEL}}_prod.traj")
        prod_logfile = os.path.join(CALC_DIR, f"{{BASE_LABEL}}_prod_sella.log")
        if os.path.isfile(prod_logfile):
            os.remove(prod_logfile)

        sella_kwargs = {sella_kwargs}
        opt = Sella(mol,
                    order=0,
                    trajectory=prod_traj,
                    logfile=prod_logfile,
                    **sella_kwargs)
        opt.attach(make_rms_callback(mol, rms_threshold), interval=1)

        try:
            converged_opt = opt.run(fmax=FMAXOPT, steps=STEPSOPT)
        except RMSConvergence:
            converged_rms = True
            logger.info(
                "RMS convergence threshold reached. Stopping optimization "
                "for {label}.")
        freqs, zpe, _, dft_energy = calc_vibrations(mol,
                                                    logger,
                                                    BASE_LABEL,
                                                    CALC_DIR)
        if validate_frequencies(freqs, 0):
            converged_freqs = True
            error_free_exec = True

            if not converged_opt:
                logging.warning(("End point optimization did not "
                                "converge according fmax criteria at "
                                f"{{FMAXOPT}}, but freqs are valid"))

            db.write(mol, name="{label}_prod",
                        data={{"energy": dft_energy * EVtoHARTREE,
                               "status": "normal"}})
            logger.info("Product optimization converged for {label}.")
            error_free_exec = True
        else:
            logger.error((
                "End point optimization failed to converge for "
                f"{label}. within {{STEPSOPT}} steps"))
            error_free_exec = True  # E.g. non-convergence is not an error
    except (RuntimeError, ValueError):
        db.write(mol, name="{label}_prod",
                    data={{"status": "error"}})
        error_free_exec = False
        logger.exception("Product optimization failed for {label}.")

    finally:
        # Build termination report.
        BOX_WIDTH = 79

        # Determine final status; success as long as freqs are valid and no
        # errors occurred.
        is_success = True if (error_free_exec and
                                       converged_freqs) else False

        # build data dictionary
        data = {{"status": "normal" if is_success else "error"}}
        if dft_energy is not None:
            e = dft_energy * EVtoHARTREE
            data["energy"] = e
        if freqs is not None:
            data["frequencies"] = freqs
        if zpe is not None:
            data["zpe"] = zpe

        # Build the report lines using formatted strings.
        report_lines = [
            "Hindered Rotor Optimization Termination Report",
            f"Status:         {{'SUCCESS ✔' if is_success else 'FAILURE ✘'}}",
            f"Converged(fmax): {{converged_opt}}",
            f"Converged(freqs): {{converged_freqs}}",
            f"Converged(rms):  {{converged_rms}}",
            f"Error-free:     {{error_free_exec}}",
            f"order:          0"
        ]
        if "energy" in data:
            report_lines.append(f"Energy:         {{data['energy']}}")
        if "zpe" in data:
            report_lines.append(f"ZPE:            {{data['zpe']}}")
        if "frequencies" in data:
            report_lines.extend([
                "Frequencies: ",
                f"{{data['frequencies']}}"
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
        with open('{label}_prod.log', 'a') as f:
            f.write("\n" + report_box + "\n")
            f.write("done\n")



def main():
    """
    Main routine for IRC and subsequent product optimization.

    Sets up the molecular system, runs the IRC optimization, and if
    successful, proceeds with product optimization and vibrational analysis.
    Results are logged and stored in the database.

    :raises ValueError: If the IRC direction cannot be determined.
    """
    # Configure logging to write into the dedicated IRC log folder.
    logging.basicConfig(
        filename=os.path.join(
            CALC_DIR, f"{{BASE_LABEL}}_irc_detailed.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    db = connect("{working_dir}/kinbot.db")
    mol = Atoms(symbols={atom},
                positions={geom})

    irc_converged, irc_success = run_irc(mol, logger, db)

    # If IRC execution was successful, proceed with product optimization.
    if irc_success:
        irc_end_point_optimization(mol, logger, db)


if __name__ == "__main__":
    if not os.path.exists(CALC_DIR):
        os.makedirs(CALC_DIR)
    main()
