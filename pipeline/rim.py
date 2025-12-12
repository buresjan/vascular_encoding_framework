"""
Rim extraction utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def extract_rim(vcs_map: Path, out: Path, tol: float = 1e-3) -> Path:
    """Extract tau≈0 rim from a VCS map and write it to `out`."""
    mesh = pv.read(str(vcs_map))
    if "tau" not in mesh.point_data:
        raise ValueError(f"'tau' array not found in {vcs_map}")

    tau = mesh["tau"]
    t0 = float(tau.min())
    rim_region = mesh.threshold(value=t0 + tol, scalars="tau", invert=False).extract_surface()
    if rim_region.n_points == 0:
        raise RuntimeError("No points found near tau=0. Increase tol.")

    edges = rim_region.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    ).connectivity("largest")

    if isinstance(edges, pv.MultiBlock):
        if edges.n_blocks == 0:
            raise RuntimeError("Could not extract rim edges.")
        lengths = [edges[i].n_points for i in range(edges.n_blocks)]
        edges = edges[np.argmax(lengths)]

    edges.save(str(out), binary=True)
    return out
