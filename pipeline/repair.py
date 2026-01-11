#!/usr/bin/env python3
"""
Repair a fragmented STL surface (with exactly four axis-aligned open ends) via voxel remeshing.

Strategy (robust):
1) Clean + drop tiny disconnected fragments.
2) Detect the 4 intended end boundary loops (axis-aligned + planar).
3) Temporarily cap those 4 ends (only for volumetric remeshing).
4) Voxelize + morphological closing + fill to get a single solid, watertight volume.
5) Marching cubes surface extraction (uniform triangles, manifold, no cracks).
6) Re-open the 4 ends by slicing exactly on the detected end planes.
7) Export the repaired *open* surface STL (NOT capped).

Dependencies:
    pip install trimesh numpy scipy scikit-image
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import trimesh
from skimage.morphology import ball, binary_closing
from trimesh.intersections import slice_mesh_plane
from trimesh.voxel import VoxelGrid, creation as vcreate, encoding as venc, ops as vops


def remove_duplicate_faces_compat(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Remove duplicate faces in a way that works across trimesh versions.
    """
    f = m.faces
    if f is None or len(f) == 0:
        return m

    # treat faces with same vertex set as duplicates (ignore winding)
    fs = np.sort(f, axis=1)
    _, unique_idx = np.unique(fs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    if len(unique_idx) != len(f):
        m.update_faces(unique_idx)
        m.remove_unreferenced_vertices()

    return m


def remove_degenerate_faces_compat(m: trimesh.Trimesh, area_eps: float = 1e-15) -> trimesh.Trimesh:
    """
    Remove degenerate/zero-area faces across trimesh versions.
    """
    a = m.area_faces
    if a is None or len(a) == 0:
        return m

    keep = a > area_eps
    if not np.all(keep):
        m.update_faces(keep)
        m.remove_unreferenced_vertices()

    return m


def edge_stats(mesh: trimesh.Trimesh) -> Dict[str, int]:
    edges = mesh.edges_sorted
    unique_edges, inv = np.unique(edges, axis=0, return_inverse=True)
    counts = np.bincount(inv)
    return {
        "unique_edges": int(len(unique_edges)),
        "boundary_edges": int(np.sum(counts == 1)),
        "manifold_edges": int(np.sum(counts == 2)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "max_faces_per_edge": int(counts.max()) if len(counts) else 0,
    }


def area_stats(mesh: trimesh.Trimesh) -> Dict[str, float]:
    a = mesh.area_faces
    if len(a) == 0:
        return dict(min=0.0, p1=0.0, p5=0.0, median=0.0, p95=0.0, max=0.0)
    return {
        "min": float(a.min()),
        "p1": float(np.percentile(a, 1)),
        "p5": float(np.percentile(a, 5)),
        "median": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def print_analysis(title: str, mesh: trimesh.Trimesh) -> None:
    es = edge_stats(mesh)
    ast = area_stats(mesh)
    print(f"\n[{title}]")
    print(f"  faces:    {len(mesh.faces):,}")
    print(f"  vertices: {len(mesh.vertices):,}")
    print(f"  watertight:          {mesh.is_watertight}")
    print(f"  winding consistent:  {mesh.is_winding_consistent}")
    print(f"  edges: boundary={es['boundary_edges']:,} nonmanifold={es['nonmanifold_edges']:,} max_faces/edge={es['max_faces_per_edge']}")
    print(f"  face area: min={ast['min']:.3e}  p1={ast['p1']:.3e}  median={ast['median']:.3e}  p95={ast['p95']:.3e}  max={ast['max']:.3e}")


def boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    edges = mesh.edges_sorted
    unique_edges, inv = np.unique(edges, axis=0, return_inverse=True)
    counts = np.bincount(inv)
    return unique_edges[counts == 1]


def connected_components_from_edges(edges: np.ndarray) -> List[np.ndarray]:
    if len(edges) == 0:
        return []
    adj: Dict[int, List[int]] = {}
    for a, b in edges:
        a = int(a); b = int(b)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    seen = set()
    comps: List[np.ndarray] = []
    for v0 in adj.keys():
        if v0 in seen:
            continue
        stack = [v0]
        seen.add(v0)
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj.get(v, []):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(np.array(comp, dtype=int))
    return comps


def order_loop(boundary_edges_comp: np.ndarray) -> np.ndarray:
    adj: Dict[int, List[int]] = {}
    for a, b in boundary_edges_comp:
        a = int(a); b = int(b)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    degrees = [len(v) for v in adj.values()]
    if min(degrees) != 2 or max(degrees) != 2:
        raise ValueError("Boundary component is not a single simple loop (vertex degree != 2).")

    start = int(boundary_edges_comp[0, 0])
    loop = [start]
    prev = None
    cur = start
    for _ in range(len(adj) + 5):
        nbs = adj[cur]
        nxt = nbs[0] if prev is None or nbs[0] != prev else nbs[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt

    if len(loop) != len(adj):
        raise ValueError("Failed to order loop (did not visit all vertices).")

    return np.array(loop, dtype=int)


def polygon_normal(points: np.ndarray) -> np.ndarray:
    n = np.zeros(3, dtype=float)
    for i in range(len(points)):
        p = points[i]
        q = points[(i + 1) % len(points)]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else n


@dataclass
class EndLoop:
    axis: int
    plane_value: float
    loop: np.ndarray


def find_four_axis_end_loops(
    mesh: trimesh.Trimesh,
    axis_dot_tol: float = 0.999,
    plane_range_tol: float | None = None,
) -> List[EndLoop]:
    if plane_range_tol is None:
        scale = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        plane_range_tol = max(1e-6, 1e-6 * scale)

    be = boundary_edges(mesh)
    comps = connected_components_from_edges(be)
    if len(comps) == 0:
        raise RuntimeError("No boundary edges found; mesh might already be watertight (no open ends).")

    def edges_for(comp_vertices: np.ndarray) -> np.ndarray:
        s = set(map(int, comp_vertices))
        mask = np.array([(int(a) in s and int(b) in s) for a, b in be], dtype=bool)
        return be[mask]

    candidates: List[Tuple[int, float, float, np.ndarray]] = []
    for comp in comps:
        pts = mesh.vertices[comp]
        center = pts.mean(axis=0)
        cov = np.cov((pts - center).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, int(np.argmin(eigvals))]
        normal = normal / np.linalg.norm(normal)

        dots = np.abs(np.eye(3) @ normal)
        axis = int(np.argmax(dots))
        axis_dot = float(dots[axis])

        coord = pts[:, axis]
        coord_range = float(coord.max() - coord.min())
        plane_value = float(np.median(coord))

        if axis_dot >= axis_dot_tol and coord_range <= plane_range_tol:
            score = float(len(comp))
            loop = order_loop(edges_for(comp))
            candidates.append((axis, plane_value, score, loop))

    picked: List[EndLoop] = []
    for axis, needed in [(2, 2), (1, 1), (0, 1)]:
        group = [c for c in candidates if c[0] == axis]
        group.sort(key=lambda t: t[2], reverse=True)
        if len(group) < needed:
            raise RuntimeError(f"Not enough axis-aligned end loops found for axis={axis}.")
        for j in range(needed):
            _, pv, _, loop = group[j]
            picked.append(EndLoop(axis=axis, plane_value=pv, loop=loop))

    z_planes = sorted([e.plane_value for e in picked if e.axis == 2])
    if abs(z_planes[1] - z_planes[0]) < plane_range_tol:
        raise RuntimeError("The two Z end loops appear co-planar; expected two distinct z planes.")

    return picked


def add_fan_cap(mesh: trimesh.Trimesh, loop: np.ndarray, axis: int, plane_value: float) -> None:
    mesh.vertices[loop, axis] = plane_value

    pts = mesh.vertices[loop]
    center = pts.mean(axis=0)
    center[axis] = plane_value

    b = mesh.bounds
    sign = -1.0 if abs(plane_value - b[0, axis]) < abs(plane_value - b[1, axis]) else 1.0
    outward = np.zeros(3); outward[axis] = sign

    n = polygon_normal(pts)
    if np.dot(n, outward) < 0:
        loop = loop[::-1]

    center_idx = len(mesh.vertices)
    mesh.vertices = np.vstack([mesh.vertices, center])

    faces = np.column_stack([loop, np.roll(loop, -1), np.full(len(loop), center_idx, dtype=int)])
    mesh.faces = np.vstack([mesh.faces, faces.astype(int)])


def voxel_remesh(mesh_closed: trimesh.Trimesh, pitch: float, closing_radius: int) -> trimesh.Trimesh:
    vg = vcreate.voxelize(mesh_closed, pitch=pitch, method="subdivide")
    dense = vg.matrix.astype(bool)

    if closing_radius > 0:
        dense = binary_closing(dense, footprint=ball(closing_radius))

    filled = vops.fill_orthographic(dense)
    vg_filled = VoxelGrid(venc.DenseEncoding(filled), transform=vg.transform)

    mc = vg_filled.marching_cubes
    mc.apply_transform(vg.transform)

    mc.merge_vertices()
    mc = remove_duplicate_faces_compat(mc)
    mc.remove_unreferenced_vertices()
    mc.fix_normals()

    if not mc.is_watertight:
        raise RuntimeError("Voxel remesh did not produce a watertight mesh; increase closing_radius or pitch.")

    return mc


def slice_open(mesh: trimesh.Trimesh, axis: int, plane_value: float, keep: str) -> trimesh.Trimesh:
    normal = np.zeros(3)
    normal[axis] = 1.0 if keep == "positive" else -1.0
    origin = np.zeros(3)
    origin[axis] = plane_value
    return slice_mesh_plane(mesh, plane_normal=normal, plane_origin=origin, cap=False)


def reopen_four_ends(mesh_closed: trimesh.Trimesh, ends: List[EndLoop]) -> trimesh.Trimesh:
    z_planes = sorted([e.plane_value for e in ends if e.axis == 2])
    zmin, zmax = z_planes[0], z_planes[1]
    x_plane = [e.plane_value for e in ends if e.axis == 0][0]
    y_plane = [e.plane_value for e in ends if e.axis == 1][0]

    out = mesh_closed
    out = slice_open(out, axis=2, plane_value=zmin, keep="positive")
    out = slice_open(out, axis=2, plane_value=zmax, keep="negative")
    out = slice_open(out, axis=1, plane_value=y_plane, keep="positive")
    out = slice_open(out, axis=0, plane_value=x_plane, keep="positive")

    out.merge_vertices()
    out = remove_duplicate_faces_compat(out)
    out.remove_unreferenced_vertices()
    out.fix_normals()
    return out


def extend_open_ends_to_planes(
    input_stl: str | Path,
    output_stl: str | Path,
    *,
    plane_targets: Mapping[str, float] | None,
    plane_range_tol: float | None = None,
    merge_digits: int = 6,
    extension_sections: int = 2,
    verbose: bool = True,
) -> None:
    """
    Extend axis-aligned open ends to target planes, using outlet labels as keys.

    plane_targets expects keys like "YZ_minX", "XZ_minY", "XY_minZ", "XY_maxZ".
    """
    if not plane_targets:
        return

    if not isinstance(plane_targets, Mapping):
        raise TypeError("plane_targets must be a mapping of outlet labels to plane values.")

    input_stl = Path(input_stl)
    output_stl = Path(output_stl)

    mesh = trimesh.load_mesh(str(input_stl), process=False)
    mesh.merge_vertices(digits_vertex=merge_digits)
    mesh = remove_duplicate_faces_compat(mesh)
    mesh = remove_degenerate_faces_compat(mesh)
    mesh.remove_unreferenced_vertices()

    ends = find_four_axis_end_loops(mesh, plane_range_tol=plane_range_tol)
    bounds = mesh.bounds
    axis_to_plane = {0: "YZ", 1: "XZ", 2: "XY"}
    axis_to_label = {0: "X", 1: "Y", 2: "Z"}

    labeled: Dict[str, Tuple[EndLoop, float]] = {}
    for end in ends:
        dist_min = abs(end.plane_value - bounds[0, end.axis])
        dist_max = abs(end.plane_value - bounds[1, end.axis])
        side = "min" if dist_min < dist_max else "max"
        label = f"{axis_to_plane[end.axis]}_{side}{axis_to_label[end.axis]}"
        axis_sign = -1.0 if side == "min" else 1.0
        labeled[label] = (end, axis_sign)

    unknown = sorted(set(plane_targets) - set(labeled))
    if unknown:
        available = ", ".join(sorted(labeled))
        unknown_str = ", ".join(unknown)
        raise ValueError(f"Unknown outlet labels: {unknown_str}. Available: {available}.")

    if extension_sections < 2:
        raise ValueError("extension_sections must be >= 2.")

    scale = float(np.linalg.norm(bounds[1] - bounds[0]))
    snap_tol = max(1e-6, 1e-6 * scale)

    # Import here to avoid pulling in taper helpers unless needed.
    from .taper_stl_end import add_tapered_extension

    mesh_out = mesh
    for label, target_value in plane_targets.items():
        end, axis_sign = labeled[label]
        axis = end.axis
        target = float(target_value)
        delta = target - float(end.plane_value)
        if abs(delta) <= snap_tol:
            mesh_out.vertices[end.loop, axis] = target
            if verbose:
                print(f"[extend] {label}: snapped to {target:.6f}")
            continue

        if delta * axis_sign > 0:
            direction = np.zeros(3, dtype=float)
            direction[axis] = axis_sign
            length = abs(delta)
            if verbose:
                print(f"[extend] {label}: extending by {length:.6f} to {target:.6f}")
            mesh_out = add_tapered_extension(
                mesh_out,
                end.loop,
                direction,
                length_mm=length,
                target_scale=1.0,
                n_sections=extension_sections,
                profile="linear",
            )
        else:
            mesh_out.vertices[end.loop, axis] = target
            if verbose:
                print(f"[extend] {label}: target is inward; snapped to {target:.6f}")

    mesh_out.export(str(output_stl))
    if verbose:
        print(f"[extend] Wrote outlet-extended STL: {output_stl}")


def repair_surface_four_open_ends(
    input_stl: str | Path,
    output_open_stl: str | Path,
    *,
    pitch: float | None = None,
    target_voxels_min_dim: int = 80,
    closing_radius: int = 1,
    merge_digits: int = 6,
    keep_area_ratio: float = 0.01,
    plane_range_tol: float | None = None,
    verbose: bool = True,
) -> float:
    """
    Repair a stitched STL with exactly four axis-aligned open ends and write the repaired open STL.
    Returns the pitch used for voxel remeshing.
    """
    input_stl = Path(input_stl)
    output_open_stl = Path(output_open_stl)

    raw = trimesh.load_mesh(str(input_stl), process=False)
    if verbose:
        print_analysis("RAW STL (as loaded)", raw)

    m = raw.copy()
    m.merge_vertices(digits_vertex=merge_digits)
    m = remove_duplicate_faces_compat(m)
    m = remove_degenerate_faces_compat(m)
    m.remove_unreferenced_vertices()
    if verbose:
        print_analysis(f"CLEANED (merge_vertices digits={merge_digits})", m)

    comps = m.split(only_watertight=False)
    areas = np.array([c.area for c in comps], dtype=float)
    max_area = float(areas.max()) if len(areas) else 0.0
    keep = [comps[i] for i, a in enumerate(areas) if a >= max_area * keep_area_ratio]
    if len(keep) == 0:
        raise RuntimeError("All components filtered out; reduce keep_area_ratio.")
    m = trimesh.util.concatenate(keep)
    m.merge_vertices(digits_vertex=merge_digits)
    m = remove_duplicate_faces_compat(m)
    m = remove_degenerate_faces_compat(m)
    m.remove_unreferenced_vertices()
    if verbose:
        print_analysis("AFTER COMPONENT FILTER", m)

    ends = find_four_axis_end_loops(m, plane_range_tol=plane_range_tol)
    if verbose:
        print("\nDetected intended end planes:")
        for e in sorted(ends, key=lambda e: (e.axis, e.plane_value)):
            ax = "xyz"[e.axis]
            print(f"  {ax} = {e.plane_value:.6f} (loop vertices={len(e.loop)})")

    if pitch is None:
        ext = m.bounds[1] - m.bounds[0]
        min_extent = float(np.min(ext))
        pitch_used = max(min_extent / float(target_voxels_min_dim), 1e-3)
        if verbose:
            print(f"\nAuto pitch: {pitch_used:.6f}")
    else:
        pitch_used = float(pitch)

    m_closed = m.copy()
    for e in ends:
        add_fan_cap(m_closed, loop=e.loop, axis=e.axis, plane_value=e.plane_value)

    m_closed.merge_vertices(digits_vertex=merge_digits)
    m_closed = remove_duplicate_faces_compat(m_closed)
    m_closed = remove_degenerate_faces_compat(m_closed)
    m_closed.remove_unreferenced_vertices()

    if verbose:
        print(f"\nVoxel remeshing: pitch={pitch_used:.6f}, closing_radius={closing_radius}")
    closed_vox = voxel_remesh(m_closed, pitch=pitch_used, closing_radius=closing_radius)
    repaired_open = reopen_four_ends(closed_vox, ends)
    if verbose:
        print_analysis("REPAIRED OPEN", repaired_open)

    be = boundary_edges(repaired_open)
    loops = connected_components_from_edges(be)
    if len(loops) != 4:
        raise RuntimeError(f"Expected 4 open boundary loops; found {len(loops)}.")
    if edge_stats(repaired_open)["nonmanifold_edges"] != 0:
        raise RuntimeError("Non-manifold edges remain after repair.")

    repaired_open.export(str(output_open_stl))
    if verbose:
        print(f"\nWrote repaired open STL: {output_open_stl}")

    return pitch_used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_stl", type=str)
    ap.add_argument("output_open_stl", type=str)
    ap.add_argument("--pitch", type=float, default=None)
    ap.add_argument("--target_voxels_min_dim", type=int, default=80)
    ap.add_argument("--closing-radius", type=int, default=1)
    ap.add_argument("--merge-digits", type=int, default=6)
    ap.add_argument("--keep-area-ratio", type=float, default=0.01)
    ap.add_argument("--plane-range-tol", type=float, default=None)
    args = ap.parse_args()

    repair_surface_four_open_ends(
        input_stl=args.input_stl,
        output_open_stl=args.output_open_stl,
        pitch=args.pitch,
        target_voxels_min_dim=args.target_voxels_min_dim,
        closing_radius=args.closing_radius,
        merge_digits=args.merge_digits,
        keep_area_ratio=args.keep_area_ratio,
        plane_range_tol=args.plane_range_tol,
        verbose=True,
    )


if __name__ == "__main__":
    main()
