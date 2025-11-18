#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix broken tube endings in STL files by detecting open boundaries
and extending or capping them with planar surfaces.

Input:  LPA.stl, RPA.stl, SVC.stl
Output: LPA_fixed.stl, RPA_fixed.stl, SVC_fixed.stl
"""

import pyvista as pv
import numpy as np
import os

def fix_open_ends(stl_path, extension_length=3.0, visualize=True):
    mesh = pv.read(stl_path)
    mesh = mesh.triangulate()

    # Detect boundary edges and extract open edges as PolyData
    edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False)
    open_edges = edges.connectivity()

    # Each open edge region corresponds to an open end
    n_regions = open_edges['RegionId'].max() + 1
    print(f"{os.path.basename(stl_path)}: Detected {n_regions} open regions")

    capped = mesh.copy()
    for rid in range(n_regions):
        boundary = open_edges.threshold(value=(rid, rid))
        points = boundary.points

        # Fit a plane to boundary points
        centroid = np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(points - centroid)
        normal = vh[2, :]

        # Create an offset (cap or extension)
        offset = centroid + normal * extension_length
        plane = pv.Plane(center=offset, direction=normal, i_size=100, j_size=100, i_resolution=2, j_resolution=2)

        # Clip mesh slightly past the open end and fill hole
        clipped = capped.clip(normal=normal, origin=centroid - normal * 0.1, invert=False)
        filled = clipped.fill_holes(100.0)
        capped = filled

    if visualize:
        p = pv.Plotter()
        p.add_mesh(capped, color="lightgreen")
        p.add_mesh(mesh, color="red", style="wireframe", opacity=0.5)
        p.show()

    # Export result
    out_path = stl_path.replace(".stl", "_fixed.stl")
    capped.save(out_path)
    print(f"Saved fixed mesh: {out_path}")


if __name__ == "__main__":
    for name in ["LPA.stl", "RPA.stl", "SVC.stl"]:
        if os.path.exists(name):
            fix_open_ends(name)
