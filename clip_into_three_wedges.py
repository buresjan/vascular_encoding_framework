#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clip an STL into three wedge regions defined by three planes that all
contain the global Y-axis. Each plane's orientation is given by a vector
lying in the X–Z plane (or equivalently by an azimuth angle).

Output: three STL files, one per wedge sector centrally symmetric to the
CCW sectors [P0→P1], [P1→P2], and [P2→P0]. In other words, for each
consecutive plane pair, we keep the opposite wedge (rotated 180° around
the shared Y-axis line). A preview window shows the mesh and the planes.

Optional: rotate the input mesh within the Y–Z plane (about the X-axis)
before clipping to better align the geometry with the planes.

Requirements:
    pip install pyvista

Usage (angles in degrees, CCW seen from +Y looking toward origin):
    python clip_into_three_wedges.py glenn_extended.stl \
        --angles 0 140 280 \
        --axis-point -31 57 84 \
        --plane-size 100

-25 rotation

Or with explicit XZ vectors (normalized automatically):
    python clip_into_three_wedges.py glenn_extended.stl \
        --vectors 1 0   -0.5 0.866   -0.707 -0.707

Notes:
- All three planes pass through the infinite line defined by axis-point
  and direction (0,1,0). Their normals have ny=0.
- Planes are sorted by azimuth to form consecutive sectors: [P0→P1],
  [P1→P2], [P2→P0] in CCW order (seen from +Y).
- If your geometry isn't centered at x=z=0, set --axis-point to a point
  on the desired Y-axis line (e.g., a centerline point).
"""

import argparse
import math
from typing import List, Tuple

import numpy as np
import pyvista as pv


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector provided.")
    return v / n


def _angles_to_normals_deg(angles_deg: List[float]) -> np.ndarray:
    normals = []
    for a in angles_deg:
        rad = math.radians(a)
        nx = math.cos(rad)
        nz = math.sin(rad)
        normals.append([nx, 0.0, nz])
    return np.array(normals, dtype=float)


def _vectors_to_normals_xz(vectors_xz: List[Tuple[float, float]]) -> np.ndarray:
    normals = []
    for (vx, vz) in vectors_xz:
        n = _normalize(np.array([vx, 0.0, vz], dtype=float))
        normals.append(n)
    return np.array(normals, dtype=float)


def _azimuth_deg(nx: float, nz: float) -> float:
    # angle in degrees in X–Z plane, measured CCW from +X toward +Z
    return (math.degrees(math.atan2(nz, nx)) + 360.0) % 360.0


def _sort_by_azimuth(normals: np.ndarray) -> np.ndarray:
    az = np.array([_azimuth_deg(nx, nz) for nx, _, nz in normals])
    idx = np.argsort(az)
    return normals[idx], az[idx]


def _clip_halfspace(mesh: pv.DataSet, origin: np.ndarray, normal: np.ndarray, keep_ge: bool) -> pv.DataSet:
    """
    Keep the half-space where (p - origin)·normal >= 0  if keep_ge True,
    else keep (p - origin)·normal <= 0.

    PyVista's clip keeps the "negative" side for invert=False.
    For plane function f(p) = n·(p - o):
        keep f<=0  -> invert=False
        keep f>=0  -> invert=True
    """
    return mesh.clip(normal=normal, origin=origin, invert=keep_ge)


def clip_sector_between(mesh: pv.PolyData,
                        origin: np.ndarray,
                        nA: np.ndarray,
                        nB: np.ndarray) -> pv.PolyData:
    """
    Return the wedge sector between planes A and B in CCW order around +Y.
    We define sector as intersection of:
        half-space A: f_A(p) >= 0
        half-space B: f_B(p) <  0
    where f_i(p) = n_i · (p - origin).
    """
    m = _clip_halfspace(mesh, origin, nA, keep_ge=True)
    m = _clip_halfspace(m,     origin, nB, keep_ge=False)
    # Clean small artifacts
    if hasattr(m, "clean"):
        m = m.clean(tolerance=1e-8, absolute=False)
    return m


def clip_opposite_sector_between(mesh: pv.PolyData,
                                 origin: np.ndarray,
                                 nA: np.ndarray,
                                 nB: np.ndarray) -> pv.PolyData:
    """
    Return the wedge sector centrally symmetric to clip_sector_between(A,B),
    i.e., the intersection of:
        half-space A: f_A(p) <= 0
        half-space B: f_B(p) >= 0
    where f_i(p) = n_i · (p - origin).
    This corresponds to a 180° rotation around the shared Y-axis line.
    """
    m = _clip_halfspace(mesh, origin, nA, keep_ge=False)
    m = _clip_halfspace(m,     origin, nB, keep_ge=True)
    if hasattr(m, "clean"):
        m = m.clean(tolerance=1e-8, absolute=False)
    return m


def visualize(mesh: pv.PolyData,
              origin: np.ndarray,
              normals_sorted: np.ndarray,
              plane_size: float = 400.0) -> None:
    pl = pv.Plotter()
    pl.add_mesh(mesh, color="white", opacity=1.0, smooth_shading=True)

    # Draw three translucent planes
    for i, n in enumerate(normals_sorted):
        # build a finite rectangle to show the infinite plane
        plane_geom = pv.Plane(center=origin,
                              direction=n,
                              i_size=plane_size,
                              j_size=plane_size)
        pl.add_mesh(plane_geom, color=None, opacity=0.3, show_edges=False)
        # draw normal arrow at a small offset from origin for clarity
        tail = origin + np.array([0.0, 0.0, 0.0])
        head = tail + n * (0.25 * plane_size)
        pl.add_arrows(tail.reshape(1, 3), (head - tail).reshape(1, 3),
                      mag=1.0, color=None)

        # Label the plane as P0/P1/P2 directly on the plane surface.
        # Place label away from the shared Y-axis to avoid overlap.
        y_axis = np.array([0.0, 1.0, 0.0])
        tangential = np.cross(y_axis, n)
        tn = np.linalg.norm(tangential)
        if tn > 0:
            tangential /= tn
        # Offset along the plane and slightly along the normal to reduce z-fighting
        in_plane_offset = 0.35 * plane_size
        out_of_plane_offset = 0.02 * plane_size
        label_pos = origin + tangential * in_plane_offset + n * out_of_plane_offset

        label_text = f"P{i}"
        # Prefer 3D text geometry when available (PyVista >= 0.43)
        try:
            text_mesh = pv.Text3D(label_text, depth=0.0,
                                   height=0.10 * plane_size,
                                   center=label_pos,
                                   normal=n)
            pl.add_mesh(text_mesh, color="yellow")
        except Exception:
            # Fallback: screen-facing point label at label_pos
            pl.add_point_labels(np.array([label_pos]), [label_text],
                                point_size=18, font_size=14)

    pl.add_axes()
    pl.show_grid()
    pl.show()


def main():
    ap = argparse.ArgumentParser(description="Clip STL into three wedge sectors defined by three Y-axis planes.")
    ap.add_argument("stl", help="Path to input STL.")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--angles", nargs=3, type=float,
                     help="Three azimuth angles in degrees (normals in X–Z plane).")
    grp.add_argument("--vectors", nargs=6, type=float,
                     metavar=("n1x", "n1z", "n2x", "n2z", "n3x", "n3z"),
                     help="Three XZ vectors guiding plane normals.")
    ap.add_argument("--axis-point", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                    help="Point on the Y-axis line shared by all planes. Default 0 0 0.")
    ap.add_argument("--plane-size", type=float, default=400.0,
                    help="Size of plane patches in the preview.")
    ap.add_argument("--rotate-yz", type=float, default=0.0,
                    help=("Rotate the mesh by this angle in degrees in the Y–Z plane "
                          "(about the X-axis, right-hand rule: +Y toward +Z). "
                          "Rotation is applied before clipping. Default 0."))
    ap.add_argument("--no-preview", action="store_true",
                    help="Skip popup visualization.")
    ap.add_argument("--out-prefix", default="sector",
                    help="Prefix for output STL files. Default 'sector'.")
    args = ap.parse_args()

    origin = np.array(args.axis_point, dtype=float)

    # Build three normals with ny=0
    if args.angles is not None:
        normals = _angles_to_normals_deg(args.angles)
    else:
        vxz = list(zip(args.vectors[0::2], args.vectors[1::2]))
        normals = _vectors_to_normals_xz(vxz)

    if normals.shape != (3, 3):
        raise RuntimeError("Expected exactly three plane normals.")

    # Sort by azimuth so sectors are consecutive CCW: [P0→P1], [P1→P2], [P2→P0]
    normals_sorted, az_sorted = _sort_by_azimuth(normals)

    # Load mesh
    mesh = pv.read(args.stl)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface().triangulate()

    # Optional rotation in the Y–Z plane (about X-axis) around the given axis-point
    rot = float(args.rotate_yz)
    if abs(rot) > 1e-12:
        # Rotate in-place about X-axis passing through 'origin'
        mesh.rotate_x(rot, point=origin, inplace=True)

    # Optional preview
    if not args.no_preview:
        visualize(mesh, origin, normals_sorted, plane_size=args.plane_size)

    # Build the three wedges centrally symmetric to the consecutive sectors
    n0, n1, n2 = normals_sorted
    # Sector A': centrally symmetric to P0→P1
    secA = clip_opposite_sector_between(mesh, origin, n0, n1)
    # Sector B': centrally symmetric to P1→P2
    secB = clip_opposite_sector_between(mesh, origin, n1, n2)
    # Sector C': centrally symmetric to P2→P0
    secC = clip_opposite_sector_between(mesh, origin, n2, n0)

    # Save STLs (each file corresponds to the opposite wedge of the named pair)
    outA = f"LPA.stl"
    outB = f"SVC.stl"
    outC = f"RPA.stl"
    for part, path in zip((secA, secB, secC), (outA, outB, outC)):
        if part.n_cells == 0:
            print(f"[warn] '{path}' would be empty. Check plane orientations and axis-point.")
        else:
            part.save(path, binary=True)
            print(f"[ok] wrote {path}  ({part.n_cells} cells)")

    print("Done.")


if __name__ == "__main__":
    main()
