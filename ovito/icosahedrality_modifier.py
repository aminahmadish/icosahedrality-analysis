"""
Icosahedrality modifier for OVITO
=================================

Port of ``calc_icosahedrality_v1.1.py`` (icosahedrality-analysis, used in
"ReaxFF Parameter Set for Boron Clusters and Icosahedral Boron Crystals", 2025)
to an OVITO Python modifier.

Algorithm (per central particle *i*, identical to the original script):

 1. Collect all neighbours of *i* within ``cutoff``.  Skip *i* if it has < 5.
 2. Keep the neighbour pairs (j, k) for which the angle j-i-k is within
    ``angle_tolerance`` degrees of 60 degrees ("equilateral" triangles).
    Skip *i* if there are fewer than 5 such pairs.
 3. In the graph formed by those pairs, find every cycle of 5 distinct
    neighbours (a candidate pentagon around *i*).
 4. A pentagon is accepted if
      * it is planar: the mean distance of its 5 atoms from the best-fit plane
        (PCA normal = direction of smallest variance) is <= ``planarity_threshold``, and
      * the central atom *i* sticks out of that plane by more than ``bump_threshold``
        (i.e. *i* is the apex of a pentagonal pyramid, not a flat centre).
 5. *i* is an icosahedral atom if at least one pentagon is accepted.  Every atom
    that is part of an accepted pentagon is "involved" in an icosahedron.

Outputs
-------
Particle property ``Icosahedrality`` (integer; same convention as the B-factor
column written by the original script):

    2 = icosahedral atom (apex of at least one accepted pentagonal pyramid)
    1 = atom belonging to an accepted pentagon of some icosahedral atom
    0 = everything else

Global attributes: ``Icosahedrality.count`` (# of atoms with value 2),
``Icosahedrality.involved_count`` (# of atoms with value >= 1) and
``Icosahedrality.fraction`` (count / number of tested atoms).

Units: OVITO works in the length units of the simulation file (Angstrom for
PDB/LAMMPS-real/etc.).  The original script worked in nm (MDTraj), so its defaults
0.21 / 0.0099 / 0.04 nm are 2.1 / 0.099 / 0.4 here.

Usage
-----
 * OVITO GUI:  load a structure file, then use "Add modification..." -> "Python script"
   (the *modifier*, NOT the Python script data source, which expects a create() function).
   Click "Edit script", paste this file's contents and commit; the parameters
   below appear automatically in the panel.
 * ``ovito`` Python module::

       from ovito.io import import_file
       from icosahedrality_modifier import IcosahedralityModifier
       pipeline = import_file("traj.dump")
       pipeline.modifiers.append(IcosahedralityModifier(cutoff=2.1))
       data = pipeline.compute()
       print(data.attributes["Icosahedrality.count"])

Notes
-----
 * Only the ``numpy`` and ``traits`` packages (both shipped with OVITO) are needed.
 * Periodic boundaries are handled by OVITO's neighbour finder (triclinic cells
   included).  A system without a simulation cell is treated as non-periodic.
 * To restrict the analysis to certain particles (the original script's
   ``selection`` argument), select them first (e.g. with an Expression selection
   modifier) and enable "Only selected particles".  As in the original, *neighbours*
   are always searched among all particles; only the central atoms are restricted.
"""

import numpy as np
from traits.api import Bool, Range

from ovito.data import CutoffNeighborFinder, DataCollection
from ovito.pipeline import ModifierInterface

__all__ = ["IcosahedralityModifier"]


def _five_cycles(adjacency):
    """Return the set of node sets (as sorted tuples) that form a 5-cycle.

    ``adjacency`` maps node -> iterable of adjacent nodes.  Cycles are
    de-duplicated by node set, like the original ``find_pentacycles``.
    """
    adj = {u: set(vs) for u, vs in adjacency.items()}
    found = set()
    for v0 in adj:
        # v0 is the smallest node of the cycle -> every cycle is found from its minimum.
        for v1 in adj[v0]:
            if v1 <= v0:
                continue
            for v2 in adj[v1]:
                if v2 <= v0 or v2 == v1:
                    continue
                for v3 in adj[v2]:
                    if v3 <= v0 or v3 == v1 or v3 == v2:
                        continue
                    for v4 in adj[v3]:
                        if v4 <= v0 or v4 in (v1, v2, v3):
                            continue
                        if v0 in adj[v4]:
                            found.add(tuple(sorted((v0, v1, v2, v3, v4))))
    return found


def _accepted_pentagons(vectors, cycles, planarity_threshold, bump_threshold):
    """Planarity and bump test for many candidate pentagons at once.

    ``vectors``: (n, 3) neighbour vectors relative to the central atom (which is
    therefore the origin).  ``cycles``: (c, 5) integer array of rows of ``vectors``.
    Returns a boolean array of length c.
    """
    pts = vectors[cycles]                                  # (c, 5, 3)
    centroid = pts.mean(axis=1)                            # (c, 3)
    centered = pts - centroid[:, None, :]                  # (c, 5, 3)

    # Plane normal = eigenvector of the smallest eigenvalue of the covariance matrix
    # (the "third principal component" in the original PCA-based implementation).
    cov = np.einsum("cki,ckj->cij", centered, centered)    # (c, 3, 3)
    _, eigvec = np.linalg.eigh(cov)                        # ascending eigenvalues
    normal = eigvec[:, :, 0]                               # (c, 3), unit length

    # Mean absolute distance of the 5 atoms from the plane through the centroid.
    total_variation = np.abs(np.einsum("ckj,cj->ck", centered, normal)).mean(axis=1)
    # Height of the central atom (origin) above that plane.
    mid_distance = np.abs(np.einsum("cj,cj->c", -centroid, normal))

    return (total_variation <= planarity_threshold) & (mid_distance > bump_threshold)


class IcosahedralityModifier(ModifierInterface):
    """Identifies icosahedral atoms (apex atoms of pentagonal pyramids)."""

    cutoff = Range(low=0.1, high=100.0, value=2.1, label="Neighbor cutoff (length units)")
    angle_tolerance = Range(low=0.0, high=60.0, value=20.0, label="Angle tolerance around 60° (deg)")
    planarity_threshold = Range(low=0.0, high=10.0, value=0.099, label="Planarity threshold (lower = stricter)")
    bump_threshold = Range(low=0.0, high=10.0, value=0.4, label="Min. apex height above pentagon (higher = stricter)")
    only_selected = Bool(False, label="Only selected particles")

    def modify(self, data: DataCollection, *, frame: int, **kwargs):
        particles = data.particles
        if particles is None:
            raise RuntimeError("The input contains no particles.")
        n_particles = particles.count

        # --- Which atoms are tested as central atoms? --------------------------------
        if self.only_selected:
            if "Selection" not in particles:
                raise RuntimeError(
                    "'Only selected particles' is enabled but the input has no 'Selection' property."
                )
            centers = np.flatnonzero(np.asarray(particles["Selection"]))
        else:
            centers = np.arange(n_particles)

        labels = np.zeros(n_particles, dtype=np.int32)

        if len(centers) > 0:
            yield "Finding neighbors"
            finder = CutoffNeighborFinder(self.cutoff, data)
            # Neighbour lists grouped per central particle, in the order of 'centers'.
            neigh_idx, neigh_vec = finder.find_all(indices=centers, sort_by="index")
            owner = neigh_idx[:, 0]
            neighbor_atom = neigh_idx[:, 1]
            # Block boundaries of each central particle inside the flat arrays
            # ('centers' is sorted ascending, so searchsorted maps owner -> block number).
            counts = np.bincount(np.searchsorted(centers, owner), minlength=len(centers))
            offsets = np.concatenate(([0], np.cumsum(counts)))

            tol = self.angle_tolerance
            icosa = []        # indices of icosahedral (apex) atoms
            involved = set()  # indices of all atoms in accepted pentagons

            for c, i in enumerate(centers):
                if c % 256 == 0:
                    yield c / len(centers)
                lo, hi = offsets[c], offsets[c + 1]
                n = hi - lo
                if n < 5:
                    continue
                vec = neigh_vec[lo:hi]
                ids = neighbor_atom[lo:hi]

                # Pairwise angles j-i-k between neighbours.
                dist = np.linalg.norm(vec, axis=1)
                unit = vec / dist[:, None]
                angles = np.degrees(np.arccos(np.clip(unit @ unit.T, -1.0, 1.0)))
                near60 = np.triu(np.abs(angles - 60.0) <= tol, k=1)
                if np.count_nonzero(near60) < 5:
                    continue

                adjacency = {}
                for j, k in zip(*np.nonzero(near60)):
                    adjacency.setdefault(int(j), []).append(int(k))
                    adjacency.setdefault(int(k), []).append(int(j))

                cycles = _five_cycles(adjacency)
                if not cycles:
                    continue
                cycles = np.array(sorted(cycles), dtype=np.intp)

                ok = _accepted_pentagons(vec, cycles, self.planarity_threshold, self.bump_threshold)
                if ok.any():
                    icosa.append(i)
                    involved.add(int(i))
                    involved.update(int(a) for a in ids[cycles[ok].ravel()])

            if involved:
                labels[np.fromiter(involved, dtype=np.intp)] = 1
            if icosa:
                labels[np.asarray(icosa, dtype=np.intp)] = 2

        # --- Output ----------------------------------------------------------------------
        particles_ = data.particles_
        particles_.create_property("Icosahedrality", data=labels)

        n_icosa = int(np.count_nonzero(labels == 2))
        n_involved = int(np.count_nonzero(labels >= 1))
        data.attributes["Icosahedrality.count"] = n_icosa
        data.attributes["Icosahedrality.involved_count"] = n_involved
        data.attributes["Icosahedrality.fraction"] = n_icosa / len(centers) if len(centers) else 0.0
        print(f"Icosahedrality (frame {frame}): {n_icosa} icosahedral atoms, "
              f"{n_involved} involved, out of {len(centers)} tested")
        yield 1.0
