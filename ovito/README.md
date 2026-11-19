# OVITO modifier: icosahedrality

[`icosahedrality_modifier.py`](icosahedrality_modifier.py) is a port of
[`calc_icosahedrality_v1.1.py`](../calc_icosahedrality_v1.1.py) to an OVITO Python modifier. It runs the same
algorithm and gives the same per-atom result as the script (verified, see *Tests*), but works interactively in OVITO
and needs only `numpy` (no MDTraj / scikit-learn).

An atom is *icosahedral* if its neighbours contain a planar pentagon (built from near-60° neighbour pairs) with the
atom sticking out of the pentagon plane, i.e. it is the apex of a pentagonal pyramid.

## Use in the OVITO GUI (OVITO 3.8+, Python scripting required)

1. Load a structure file (e.g. `examples/boron_alpha-rhombohedral.pdb`).
2. In the pipeline editor (right panel) choose **Add modification… → Python script**.
   Use the *modifier* — not the "Python script" entry in the top *Pipelines* menu, which creates a data source and
   fails with *"does not define a function named create()"*.
3. Click **Edit script**, replace the template with the contents of `icosahedrality_modifier.py`, and commit
   (Ctrl+Enter). The parameters below appear in the panel; the **Script output** box prints a one-line summary.
4. Results: open the data inspector (bottom strip) → **Particles** (column `Icosahedrality`) and
   **Global Attributes**. To color atoms, add a **Color coding** modifier *after* the script on `Icosahedrality` (0–2).

## Use from Python

```python
import sys; sys.path.insert(0, "ovito")                       # run from the repository root
from ovito.io import import_file
from icosahedrality_modifier import IcosahedralityModifier

pipeline = import_file("examples/boron_beta-rhombohedral.pdb")
pipeline.modifiers.append(IcosahedralityModifier())
data = pipeline.compute()
print(data.attributes["Icosahedrality.count"])               # -> 1248
```

## Parameters

OVITO works in the file's length unit (Å for PDB). The script used nm (MDTraj), so its constants are converted:

| Parameter | Default | Original script |
|---|---|---|
| Neighbor cutoff | 2.1 Å | `CUTOFF = 0.21` nm |
| Angle tolerance around 60° | 20° | `ANGLE_TOLERANCE = 20.0` |
| Planarity threshold (lower = stricter) | 0.099 Å | `PLANARITY_THRESHOLD = 0.0099` nm |
| Min. apex height above pentagon (higher = stricter) | 0.4 Å | `MIDDLE_BUMP_THRESHOLD = 0.04` nm |
| Only selected particles | off | the script's `selection` argument |

With *Only selected particles*, only selected atoms are tested as icosahedral centres; neighbours are still searched
among all atoms (as in the script). Select first, e.g. with an *Expression selection* modifier.

## Output

* Particle property `Icosahedrality`: **2** = icosahedral atom, **1** = atom in an accepted pentagon of an
  icosahedral atom, **0** = other (same coding as the B-factor column the script writes to its PDB).
* Global attributes: `Icosahedrality.count` (# of atoms with value 2), `Icosahedrality.involved_count`
  (# with value ≥ 1), `Icosahedrality.fraction` (count / # tested atoms). For a trajectory they are evaluated per frame
  and can be plotted over time.

## Expected results (default parameters)

| Example (`examples/`) | Atoms | Cell in file | `count` | `involved_count` |
|---|---|---|---|---|
| `B80_c1coreshell_unitcell_final.pdb` | 80 | none (non-periodic) | 16 | 36 |
| `boron_alpha-rhombohedral.pdb` | 2592 | periodic | 2592 | 2592 |
| `boron_beta-rhombohedral.pdb` | 1260 | periodic | 1248 | 1248 |
| `last_npt3600_small_boron_final.pdb` | 1260 | none (non-periodic) | 146 | 217 |

Notes:

* α-boron: every atom is an icosahedron vertex, so all 2592 are found. β-boron: 12 atoms are not.
* Files without a `CRYST1` record are analysed as non-periodic, so atoms near the edge lose neighbours. Use files
  with a cell for periodic crystals.

## Tests

```bash
pip install ovito pytest
pytest tests/test_modifier.py          # or: python tests/test_modifier.py
```

The tests check the counts above, compare every atom against `tests/reference/*.labels` (output of
`calc_icosahedrality_v1.1.py` with default parameters; for the two files without a cell it was run on a copy with a
large dummy `CRYST1` line, because the script needs a cell), and check invariance under periodic shift + re-wrapping
and the *Only selected particles* option.
