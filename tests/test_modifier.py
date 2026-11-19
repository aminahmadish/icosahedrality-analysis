"""
Tests for the OVITO icosahedrality modifier (ovito/icosahedrality_modifier.py).

Run with pytest:           pytest tests/test_modifier.py
or stand-alone (no pytest): python tests/test_modifier.py

Requires the ``ovito`` and ``numpy`` Python packages (``pip install ovito``).
MDTraj / scikit-learn are NOT needed: the reference labels in tests/reference/ were
produced once by calc_icosahedrality_v1.1.py (default parameters).
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ovito"))

from ovito.io import import_file  # noqa: E402
from ovito.modifiers import ExpressionSelectionModifier  # noqa: E402

from icosahedrality_modifier import IcosahedralityModifier  # noqa: E402

# example file -> (number of atoms, expected `Icosahedrality.count`, expected `.involved_count`)
EXPECTED = {
    "B80_c1coreshell_unitcell_final": (80, 16, 36),      # cluster, no cell in file
    "boron_alpha-rhombohedral": (2592, 2592, 2592),      # periodic crystal
    "boron_beta-rhombohedral": (1260, 1248, 1248),       # periodic crystal
    "last_npt3600_small_boron_final": (1260, 146, 217),  # disordered, no cell in file
}


def _pdb(name):
    return os.path.join(ROOT, "examples", name + ".pdb")


def _reference_labels(name):
    with open(os.path.join(ROOT, "tests", "reference", name + ".labels")) as f:
        return np.array([int(x) for x in f if not x.startswith("#") and x.strip()])


def _run(name, *extra_modifiers, **params):
    pipeline = import_file(_pdb(name))
    for m in extra_modifiers:
        pipeline.modifiers.append(m)
    pipeline.modifiers.append(IcosahedralityModifier(**params))
    return pipeline.compute()


def test_expected_counts():
    """Global attributes match the values documented in ovito/README.md."""
    for name, (n_atoms, n_icosa, n_involved) in EXPECTED.items():
        data = _run(name)
        assert data.particles.count == n_atoms, name
        assert data.attributes["Icosahedrality.count"] == n_icosa, name
        assert data.attributes["Icosahedrality.involved_count"] == n_involved, name
        assert np.isclose(data.attributes["Icosahedrality.fraction"], n_icosa / n_atoms), name


def test_per_atom_labels_match_original_script():
    """Atom-by-atom identical to calc_icosahedrality_v1.1.py."""
    for name in EXPECTED:
        got = np.asarray(_run(name).particles["Icosahedrality"])
        ref = _reference_labels(name)
        assert got.shape == ref.shape, name
        assert np.array_equal(got, ref), f"{name}: {np.sum(got != ref)} atoms differ"


def test_periodic_translation_invariance():
    """Shifting a periodic crystal and re-wrapping it into the cell must not change any label."""
    name = "boron_beta-rhombohedral"
    reference = np.asarray(_run(name).particles["Icosahedrality"])

    def shift_and_wrap(frame, data):
        cell = np.asarray(data.cell[...])
        lengths, origin = np.diag(cell[:, :3]), cell[:, 3]
        positions = data.particles_.positions_
        positions[...] = (np.asarray(positions) - origin + (3.3, -2.1, 5.7)) % lengths + origin

    shifted = np.asarray(_run(name, shift_and_wrap).particles["Icosahedrality"])
    assert np.array_equal(shifted, reference)


def test_only_selected_particles():
    """With 'only_selected', only selected atoms can be centers; results equal the full run there."""
    name = "boron_beta-rhombohedral"
    full = np.asarray(_run(name).particles["Icosahedrality"])
    data = _run(name, ExpressionSelectionModifier(expression="ParticleIndex < 100"), only_selected=True)
    labels = np.asarray(data.particles["Icosahedrality"])
    assert np.all(np.flatnonzero(labels == 2) < 100)
    assert np.array_equal(labels[:100] == 2, full[:100] == 2)
    assert np.isclose(data.attributes["Icosahedrality.fraction"], np.sum(full[:100] == 2) / 100)


def test_only_selected_without_selection_fails():
    try:
        _run("B80_c1coreshell_unitcell_final", only_selected=True)
    except Exception as exc:
        assert "Selection" in str(exc)
    else:
        raise AssertionError("expected an error when there is no Selection property")


if __name__ == "__main__":
    failed = 0
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {fn_name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {fn_name}: {exc!r}")
    sys.exit(1 if failed else 0)
