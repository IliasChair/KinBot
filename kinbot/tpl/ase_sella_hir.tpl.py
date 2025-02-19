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
from typing import Optional

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.calculators.gaussian import Gaussian
from sella import Sella, Constraints

from kinbot.constants import EVtoHARTREE
from kinbot.ase_modules.calculators.{code} import {Code}
from kinbot.ase_modules.calculators.nn_pes import make_rms_callback, RMSConvergence
from kinbot.ase_sella_util import get_valid_multiplicity, validate_frequencies
import cclib
import subprocess

# Determine the full label and extract the basename. If the provided
# label includes a directory, BASE_LABEL will be only the final component.
LABEL = "{label}"
BASE_LABEL = os.path.basename(LABEL)
# Build the calculation directory path. If LABEL includes a directory,
# we join that dirname with the base identifier directory.
CALC_DIR = os.path.join(os.path.dirname(LABEL) or ".",
                         f"{{BASE_LABEL}}_hir_dir")

FMAX = 0.001
STEPS = 500
RMS_THRESH = 0.00005  # RMS displacement threshold (Å)


def setup_gaussian_calc(mol: Atoms, mult: Optional[int] = None) -> None:
    """Configure Gaussian calculator for the molecule."""

    calc_params = {{
        "mem": "8GB",
        "nprocshared": 4,
        "method": "wb97xd",
        "basis": "6-31G(d)",
        "chk": f"{{BASE_LABEL}}_vib.chk",
        "freq": "",
        "extra": "SCF=(XQC, MaxCycle=200)"
    }}
    if mult is not None:
        calc_params["mult"] = mult

    mol.calc = Gaussian(**calc_params)
    mol.calc.label = os.path.join(CALC_DIR, f"{{BASE_LABEL}}_vib")


# can later be adaped as a drop in for any calculator
def calc_vibrations(
        mol: Atoms, logger: logging.Logger
        ) -> tuple[np.ndarray, float, np.ndarray]:
    """Calculate vibrational frequencies."""
    mol = mol.copy()

    # Try calculation with default multiplicity
    setup_gaussian_calc(mol)
    dft_energy = None
    try:
        dft_energy = mol.get_potential_energy()
    except RuntimeError:
        logger.warning(
            "Initial calculation failed, retrying with corrected multiplicity")
        mult = get_valid_multiplicity(mol)
        setup_gaussian_calc(mol, mult)
        dft_energy = mol.get_potential_energy()
    if dft_energy is None:
        raise RuntimeError(
            "Vibrational frequency calculation failed for {label}")

    # Process frequency results
    chk_path = os.path.join(CALC_DIR, f"{{BASE_LABEL}}_vib.chk")
    fchk_path = os.path.join(CALC_DIR, f"{{BASE_LABEL}}_vib.fchk")
    log_path = os.path.join(CALC_DIR, f"{{BASE_LABEL}}_vib.log")
    subprocess.run(["formchk", chk_path, fchk_path], check=True)

    fchk_data = cclib.io.ccopen(fchk_path).parse()
    log_data = cclib.io.ccopen(log_path).parse()

    return fchk_data.vibfreqs, log_data.zpve, fchk_data.hessian, dft_energy


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

    dft_energy, freqs, zpe, hessian = (None,) * 4

    # Apply dihedral constraints.
    const = Constraints(mol)
    fix_indices = [idx - 1 for idx in {fix}]
    const.fix_dihedral(fix_indices)

    error_free_exec = False
    rms_threshold = [RMS_THRESH] # mutable
    converged_fmax = False
    converged_freqs = False
    converged_rms = True
    try:
        # Setup initial calculator
        kwargs = {kwargs}
        mol.calc = {Code}(**kwargs)
        if "{Code}" == "Gaussian":
            mol.get_potential_energy()  # what was this used for?
            kwargs["guess"] = "Read"
            mol.calc = {Code}(**kwargs)

        mol.calc.label = "{label}"
        # Log initial configuration.
        logger.info("Starting hindered rotor optimization for {label}")
        logger.info(f"Configuration: order={order}, constraints={{const}}")

        # Handle monoatomic case
        if len(mol) == 1:
            raise ValueError("Hindered rotor optimization is not "
                             "applicable to monoatomic species.")

        # Optimization
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

        # Attach RMS callback to check every step
        opt.attach(make_rms_callback(mol, rms_threshold), interval=1)

        try:
            converged_fmax = opt.run(fmax=FMAX, steps=STEPS)
        except RMSConvergence:
            converged_rms = True
            logger.info("RMS convergence threshold reached. "
                        "Stopping optimization.")

        freqs, zpe, hessian, dft_energy = calc_vibrations(mol, logger)
        if validate_frequencies(freqs, {order}):
            converged_freqs = True
            error_free_exec = True

            if not converged_fmax:
                logging.warning(("Hindered Rotor Optimization did not "
                                "converge according fmax criteria at "
                                f"{{FMAX}}, but freqs are valid"))

            logger.info(
                ("Hindered Rotor Optimization successfully converged. "
                f"for {label}. Final energy: {{dft_energy*EVtoHARTREE}} "
                "Hartree"))

        else:
            logger.error((
                "Hindered Rotor Optimization failed to converge for "
                f"{label}. within {{STEPS}} steps"))
            error_free_exec = True  # E.g. non-convergence is not an error

    except Exception as err:
        logger.error(f"Hindered Rotor Optimization failed: {{err}}")
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
        if hessian is not None:
            data["hess"] = hessian

        db.write(mol, name="{label}",
                data=data)

        # Build the report lines using formatted strings.
        report_lines = [
            "Hindered Rotor Optimization Termination Report",
            f"Status:         {{'SUCCESS ✔' if is_success else 'FAILURE ✘'}}",
            f"Converged(fmax): {{converged_fmax}}",
            f"Converged(freqs): {{converged_freqs}}",
            f"Error-free:     {{error_free_exec}}",
            f"order:          {order}"
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
        with open("{label}.log", "a") as f:
            f.write("\n" + report_box + "\n")
            f.write("done\n")


if __name__ == "__main__":
    if not os.path.exists(CALC_DIR):
        os.makedirs(CALC_DIR)
    main()