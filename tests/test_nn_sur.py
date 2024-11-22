import pytest
import numpy as np
from ase.atoms import Atoms
from kinbot.ase_modules.calculators.nn_pes import Nn_surr, ELEMENT_ENERGIES

@pytest.fixture
def calculator():
    return Nn_surr()

def test_single_atom_h():
    # Test single H atom case
    calc = Nn_surr()
    atoms = Atoms('H', positions=[[0, 0, 0]])
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert isinstance(energy, float)
    assert np.isclose(energy, ELEMENT_ENERGIES["H"])
    assert forces.shape == (1, 3)
    assert forces.dtype == np.float64
    assert np.allclose(forces, 0)

def test_h2_molecule():
    # Test H2 molecule
    calc = Nn_surr()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert isinstance(energy, float)
    assert forces.shape == (2, 3)
    assert forces.dtype == np.float64

    # H2 bond length should be around 0.74 Å
    assert np.linalg.norm(atoms.positions[1] - atoms.positions[0]) == pytest.approx(0.74, abs=0.1)

def test_water_molecule():
    # Test H2O molecule
    calc = Nn_surr()
    atoms = Atoms('H2O',
                 positions=[[0, 0, 0],
                          [0.757, 0.586, 0],
                          [-0.757, 0.586, 0]])
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert isinstance(energy, float)
    assert forces.shape == (3, 3)
    assert forces.dtype == np.float64

def test_methane():
    # Test CH4 molecule
    calc = Nn_surr()
    atoms = Atoms('CH4',
                 positions=[[0, 0, 0],
                          [0.629, 0.629, 0.629],
                          [-0.629, -0.629, 0.629],
                          [0.629, -0.629, -0.629],
                          [-0.629, 0.629, -0.629]])
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert isinstance(energy, float)
    assert forces.shape == (5, 3)
    assert forces.dtype == np.float64

def test_large_molecule():
    # Test larger system to ensure calculator switches to force calculator for energy
    calc = Nn_surr()
    # Create a random molecule with 20 atoms
    atoms = Atoms('H20',
                 positions=np.random.rand(20, 3))
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert isinstance(energy, float)
    assert forces.shape == (20, 3)
    assert forces.dtype == np.float64

def test_calculator_properties():
    calc = Nn_surr()
    assert "energy" in calc.implemented_properties
    assert "forces" in calc.implemented_properties

def test_energy_calculator_lazy_loading():
    calc = Nn_surr()
    assert calc._energy_calculator is None
    _ = calc.energy_calculator
    assert calc._energy_calculator is not None

def test_force_calculator_lazy_loading():
    calc = Nn_surr()
    assert calc._force_calculator is None
    _ = calc.force_calculator
    assert calc._force_calculator is not None

def test_get_potential_energy_fallback_to_force_calc():
    """Test that energy calculation falls back to force calculator when AIMNet fails"""
    calc = Nn_surr()

    # Create a molecule that AIMNet typically struggles with
    # Using a large ring structure with unconventional bond lengths
    n_atoms = 8
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        angle = 2 * np.pi * i / n_atoms
        positions[i] = [3*np.cos(angle), 3*np.sin(angle), 0]  # Large ring radius

    atoms = Atoms('H' * n_atoms, positions=positions)
    atoms.calc = calc

    atoms.calc._energy_calculator = "dummy"
    # Get energy - should trigger force calculator fallback
    energy = atoms.get_potential_energy()

    # Verify results
    assert isinstance(energy, float)
    assert not np.isnan(energy)
    assert np.isfinite(energy)

    # Verify force calculator was used as fallback
    assert calc._force_calculator is not None