#!/usr/bin/env python3
"""
repair_tube_end.py  (v2.1)

Modes
-----
- plane        : extend boundary to plane and clip there.
- circularize  : as v2.0, fit circle and union a short cylinder before final clip.
- taper        : generate a tapered "collar" that shrinks toward a user-specified center
                 from the start of the artificial extension, with shrinking increasing
                 with distance to the plane. Finally clip at the plane.

Taper parameters
----------------
--center  CX CY CZ       : 3D point toward which the rim shrinks (required in taper mode).
--end-scale S            : multiplicative radius factor at the plane (0<S<=1). Default 0.7.
--layers K               : number of interpolation layers of the taper surface. Default 12.

Usage example
-------------
python repair_tube_end.py -i LPA.stl -o LPA_fixed_taper.stl \
  --plane -31 58 84 --normal -1.6 0 0.53 \
  --mode taper --center -28 60 82 --end-scale 0.8 --layers 16
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

try:
    import pyvista as pv
except Exception as e:
    raise SystemExit("Requires pyvista. pip install pyvista. " + str(e))


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n


@dataclass
class Plane:
    origin: np.ndarray
    normal: np.ndarray
    def n(self): return _unit(self.normal)
    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.origin) @ self.n()


def read_mesh(path: Path) -> pv.PolyData:
    m = pv.read(str(path))
    if not isinstance(m, pv.PolyData):
        m = m.extract_surface()
    return m.triangulate().clean()


# ---------- boundary utilities ----------

def find_open_edge_loops(surf: pv.PolyData):
    edges = surf.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False
    ).clean()
    blocks = edges.split_bodies()
    loops = [blocks[i] for i in range(blocks.n_blocks) if blocks[i] is not None and blocks[i].n_points > 0]
    loops = [L for L in loops if L.n_cells >= 4]
    return loops


def choose_loop_near_plane(loops, plane: Plane):
    if not loops:
        raise RuntimeError("No open loops found.")
    d = [abs(np.mean(plane.signed_distance(np.asarray(L.points)))) for L in loops]
    return loops[int(np.argmin(d))]


def to_lines_polydata(loop_ds: pv.DataSet) -> pv.PolyData:
    pts = np.asarray(loop_ds.points)
    if hasattr(loop_ds, "lines") and loop_ds.lines is not None and len(loop_ds.lines) > 0:
        lines = np.array(loop_ds.lines)
    else:
        conn = loop_ds.cell_connectivity
        offsets = loop_ds.offset
        if conn is not None and offsets is not None and len(conn) > 0:
            cells = []
            start = 0
            for j in range(loop_ds.n_cells):
                end = offsets[j+1] if j+1 < len(offsets) else len(conn)
                ids = conn[start:end]
                start = end
                cells.extend([len(ids)] + list(ids))
            lines = np.array(cells, dtype=np.int64)
        else:
            ids = np.arange(len(pts), dtype=np.int64)
            lines = np.hstack(([len(ids)], ids))
    return pv.PolyData(pts, lines=lines)


def order_loop_points(loop_pd: pv.PolyData) -> np.ndarray:
    lines = loop_pd.lines.reshape(-1, 3)[:, 1:3]
    npts = loop_pd.n_points
    nbrs = [[] for _ in range(npts)]
    for a, b in lines:
        nbrs[a].append(b)
        nbrs[b].append(a)
    start = int(np.argmin(np.arange(npts)))
    order = [start]
    prev = None
    curr = start
    while True:
        candidates = nbrs[curr]
        nxt = candidates[0] if candidates[0] != prev else (candidates[1] if len(candidates) > 1 else None)
        if nxt is None or nxt == start:
            break
        order.append(nxt)
        prev, curr = curr, nxt
        if len(order) > npts + 5:
            break
    return np.array(order, dtype=int)


# ---------- fitting ----------

def orthonormal_basis_from_normal(n: np.ndarray):
    n = _unit(n)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(n, a))
    v = np.cross(n, u)
    return u, v, n


def project_to_plane_coords(pts: np.ndarray, plane: Plane):
    u, v, n = orthonormal_basis_from_normal(plane.n())
    rel = pts - plane.origin
    x = rel @ u
    y = rel @ v
    z = rel @ n
    return x, y, z, u, v, n


def fit_circle_2d(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    a = -D / 2.0
    b0 = -E / 2.0
    r = np.sqrt(max(a*a + b0*b0 - F, 1e-16))
    return a, b0, r


# ---------- operations ----------

def extrude_to_plane(mesh: pv.PolyData, plane: Plane, overshoot: float):
    loops = find_open_edge_loops(mesh)
    loop = choose_loop_near_plane(loops, plane)
    pts = np.asarray(loop.points)
    s = plane.signed_distance(pts)
    s_mean = float(np.mean(s))

    direction = -plane.n() if s_mean > 0 else plane.n()
    L = float(np.max(np.abs(s))) + float(overshoot)

    loop_pd = to_lines_polydata(loop)

    try:
        wall = loop_pd.extrude(direction * L, capping=False).triangulate().clean()
    except Exception:
        jitter = 1e-4 * L * direction
        loop_pd = pv.PolyData(loop_pd.points + jitter, lines=loop_pd.lines)
        wall = loop_pd.extrude(direction * L, capping=False).triangulate().clean()

    if wall.n_cells == 0:
        raise RuntimeError("Extrusion produced no geometry. Try increasing --overshoot or check plane.")

    merged = mesh.merge(wall, merge_points=True).clean(tolerance=1e-9)
    invert = True if s_mean < 0 else False
    clipped = merged.clip(origin=plane.origin, normal=plane.n(), invert=invert).triangulate().clean()
    return clipped, loop_pd, s_mean


def build_taper_wall(loop_pd: pv.PolyData, plane: Plane, center: np.ndarray,
                     L: float, end_scale: float, layers: int) -> pv.PolyData:
    order = order_loop_points(loop_pd)
    P0 = loop_pd.points[order]

    x0, y0, z0, u, v, n = project_to_plane_coords(P0, plane)
    cx, cy, _, _, _, _ = project_to_plane_coords(center[None, :], plane)
    cx = cx[0]; cy = cy[0]

    r0 = np.stack([x0 - cx, y0 - cy], axis=1)  # (m,2)

    K = int(max(layers, 2))
    pts_layers = []
    for k in range(K + 1):
        t = k / K  # 0 at rim, 1 at plane
        scale = (1.0 - t) + end_scale * t
        xk = cx + scale * r0[:, 0]
        yk = cy + scale * r0[:, 1]
        zk = z0 + t * L
        pts_k = plane.origin + np.outer(xk, u) + np.outer(yk, v) + np.outer(zk, n)
        pts_layers.append(pts_k)

    pts = np.vstack(pts_layers)
    m = P0.shape[0]
    faces = []
    for k in range(K):
        base0 = k * m
        base1 = (k + 1) * m
        for i in range(m):
            a = base0 + i
            b = base0 + ((i + 1) % m)
            c = base1 + ((i + 1) % m)
            d = base1 + i
            faces.extend([3, a, b, c, 3, a, c, d])
    faces = np.array(faces, dtype=np.int64)
    wall = pv.PolyData(pts, faces=faces).clean()
    return wall


def extrude_taper_to_plane(mesh: pv.PolyData, plane: Plane, center: np.ndarray,
                           overshoot: float, end_scale: float, layers: int):
    loops = find_open_edge_loops(mesh)
    loop = choose_loop_near_plane(loops, plane)
    pts = np.asarray(loop.points)
    s = plane.signed_distance(pts)
    s_mean = float(np.mean(s))
    direction = -plane.n() if s_mean > 0 else plane.n()
    L = float(np.max(np.abs(s))) + float(overshoot)

    loop_pd = to_lines_polydata(loop)
    wall = build_taper_wall(loop_pd, plane, np.asarray(center, float), L, end_scale, layers)

    merged = mesh.merge(wall, merge_points=True).clean(tolerance=1e-9)
    invert = True if s_mean < 0 else False
    clipped = merged.clip(origin=plane.origin, normal=plane.n(), invert=invert).triangulate().clean()
    return clipped, wall, s_mean


def circularize_at_plane(mesh_after_plane: pv.PolyData, loop_on_plane: pv.PolyData, plane: Plane,
                         collar_len: float, radius_scale: float):
    boundary = mesh_after_plane.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False
    ).clean()
    loops = boundary.split_bodies()
    candidates = [loops[i] for i in range(loops.n_blocks)]
    candidates = [c for c in candidates if c is not None and c.n_points > 0]
    if candidates:
        d = [np.mean(np.abs(plane.signed_distance(np.asarray(c.points)))) for c in candidates]
        plane_loop = candidates[int(np.argmin(d))]
    else:
        plane_loop = loop_on_plane

    pts = np.asarray(plane_loop.points)
    x, y, _, u, v, n = project_to_plane_coords(pts, plane)
    a2d, b2d, r = fit_circle_2d(x, y)
    r *= float(radius_scale)
    center3d = plane.origin + a2d * u + b2d * v
    h = float(collar_len)
    cyl = pv.Cylinder(center=center3d + 0.5 * h * (-n), direction=-n, radius=r, height=h, resolution=128, capping=True)
    united = mesh_after_plane.boolean_union(cyl).triangulate().clean()
    sign_centroid = np.sign(np.mean(plane.signed_distance(np.asarray(mesh_after_plane.points))))
    invert = True if sign_centroid < 0 else False
    result = united.clip(origin=plane.origin, normal=plane.n(), invert=invert).triangulate().clean()
    return result, float(r), center3d


def compute_default_overshoot(mesh: pv.PolyData) -> float:
    b = np.array(mesh.bounds)
    return 0.05 * np.linalg.norm(b[1::2] - b[::2])


def repair_one(input_path: Path, output_path: Path, plane: Plane,
               mode: str = "plane", overshoot: float | None = None,
               collar_len: float = 5.0, radius_scale: float = 1.0,
               center: np.ndarray | None = None, end_scale: float = 0.7, layers: int = 12,
               verbose=True):
    mesh = read_mesh(input_path)
    if overshoot is None:
        overshoot = compute_default_overshoot(mesh)

    if mode == "taper":
        if center is None:
            raise SystemExit("--center is required in taper mode")
        result, wall, s_mean = extrude_taper_to_plane(mesh, plane, center, overshoot, end_scale, layers)
        fitted = None
    else:
        clipped, loop_pd, s_mean = extrude_to_plane(mesh, plane, overshoot)
        if mode == "circularize":
            result, fitted, _ = circularize_at_plane(clipped, loop_pd, plane, collar_len=collar_len, radius_scale=radius_scale)
        else:
            result, fitted = clipped, None

    result.save(str(output_path))
    if verbose:
        print(f"[OK] {input_path.name} -> {output_path.name}")
        print(f" mode: {mode}")
        print(f" plane: origin={plane.origin.tolist()}, normal={plane.n().tolist()}")
        print(f" loop mean signed distance: {s_mean:.6g}")
        if mode == "taper":
            print(f" end-scale: {end_scale}  layers: {layers}  center: {center.tolist()}")
        if mode == "circularize":
            print(f" radius-scale: {radius_scale}")
        print(f" output cells: {result.n_cells}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Repair a broken tube end to a plane; modes: plane, circularize, taper.")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--plane", nargs=3, type=float, required=True, metavar=("PX","PY","PZ"))
    ap.add_argument("--normal", nargs=3, type=float, required=True, metavar=("NX","NY","NZ"))
    ap.add_argument("--mode", choices=["plane","circularize","taper"], default="plane")
    ap.add_argument("--overshoot", type=float, default=None, help="Extra distance beyond plane for the boundary extension.")
    ap.add_argument("--collar", type=float, default=6.0, help="Length of the cylinder collar when circularizing.")
    ap.add_argument("--radius-scale", type=float, default=1.0, help="Multiply fitted radius to adjust tightness.")
    ap.add_argument("--center", nargs=3, type=float, help="3D point toward which to taper the outlet (taper mode).")
    ap.add_argument("--end-scale", type=float, default=0.7, help="Final radius factor at the plane (0<S<=1) in taper mode.")
    ap.add_argument("--layers", type=int, default=12, help="Number of interpolation layers for the tapered wall.")
    args = ap.parse_args()

    plane = Plane(origin=np.array(args.plane, float), normal=np.array(args.normal, float))
    center = np.array(args.center, float) if args.center is not None else None

    ok = repair_one(Path(args.input), Path(args.output), plane,
                    mode=args.mode, overshoot=args.overshoot,
                    collar_len=args.collar, radius_scale=args.radius_scale,
                    center=center, end_scale=args.end_scale, layers=args.layers, verbose=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()