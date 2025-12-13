"""
Extrude an open end of an STL and smoothly taper its cross-section.

This is a lightweight, robust implementation built around simple boundary loop
extraction (no heavy dependencies beyond numpy/trimesh). It can be used as a
standalone CLI or invoked from the pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh

# End identifiers: plane the rim lies in + min/max along the remaining axis.
END_MAP: Dict[str, Tuple[int, str]] = {
    "yz_min_x": (0, "min"),
    "yz_max_x": (0, "max"),
    "xz_min_y": (1, "min"),
    "xz_max_y": (1, "max"),
    "xy_min_z": (2, "min"),
    "xy_max_z": (2, "max"),
}


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 3.0 * t * t - 2.0 * t * t * t


def find_boundary_loops(mesh: trimesh.Trimesh) -> List[List[int]]:
    """
    Extract boundary loops (ordered) by walking edges that belong to exactly one face.
    """
    faces = mesh.faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    edge_tuples = [tuple(e) for e in edges_sorted]

    counts = {}
    for e in edge_tuples:
        counts[e] = counts.get(e, 0) + 1
    boundary_edges = [e for e, c in counts.items() if c == 1]

    adj: Dict[int, List[int]] = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    loops: List[List[int]] = []
    visited: set[Tuple[int, int]] = set()
    for a, b in boundary_edges:
        if (a, b) in visited or (b, a) in visited:
            continue

        loop = [a, b]
        visited.add((a, b))
        visited.add((b, a))
        cur = b
        prev = a
        while True:
            nbs = adj.get(cur, [])
            nxt = None
            for n in nbs:
                if n != prev and ((cur, n) not in visited and (n, cur) not in visited):
                    nxt = n
                    break
            if nxt is None:
                break
            if nxt == loop[0]:
                break
            loop.append(nxt)
            visited.add((cur, nxt))
            visited.add((nxt, cur))
            prev, cur = cur, nxt

        loops.append(loop)
    return loops


def characterize_loops(mesh: trimesh.Trimesh, loops: Sequence[Sequence[int]]):
    infos = []
    verts = mesh.vertices
    for idx, loop in enumerate(loops):
        pts = verts[np.asarray(loop, dtype=int)]
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        size = maxs - mins
        axis = int(np.argmin(size))
        center = pts.mean(axis=0)
        infos.append(
            {
                "index": idx,
                "axis": axis,
                "center": center,
                "size": size,
                "mins": mins,
                "maxs": maxs,
                "n": len(loop),
            }
        )
    return infos


def select_loop(mesh: trimesh.Trimesh, loops, loop_infos, end_name: str, tol_ratio: float = 0.02):
    if end_name not in END_MAP:
        raise ValueError(f"Unknown end_name {end_name}")
    axis, extreme = END_MAP[end_name]

    bounds = mesh.bounds
    target_val = bounds[1, axis] if extreme == "max" else bounds[0, axis]
    tol = (bounds[1, axis] - bounds[0, axis]) * tol_ratio

    candidates = []
    for info in loop_infos:
        if info["axis"] != axis:
            continue
        dist = abs(info["center"][axis] - target_val)
        if dist <= tol:
            candidates.append(info)

    if not candidates:
        # fall back to closest if none within tolerance
        for info in loop_infos:
            if info["axis"] != axis:
                continue
            dist = abs(info["center"][axis] - target_val)
            candidates.append(info)

    if not candidates:
        raise RuntimeError(f"No boundary loop found for {end_name}")

    best = max(candidates, key=lambda inf: (inf["n"], -abs(inf["center"][axis] - target_val)))
    return best


def extrude_with_taper(
    mesh: trimesh.Trimesh,
    loop_indices: Sequence[int],
    axis: int,
    extreme: str,
    *,
    length: float,
    scale_target: float,
    segments: int = 10,
) -> trimesh.Trimesh:
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()

    rim = list(map(int, loop_indices))
    rim_pts = verts[rim]
    plane_value = float(np.median(rim_pts[:, axis]))
    rim_pts[:, axis] = plane_value  # flatten to plane

    center = rim_pts.mean(axis=0)
    plane_axes = [0, 1, 2]
    plane_axes.remove(axis)
    offsets = rim_pts[:, plane_axes] - center[plane_axes]

    direction = 1.0 if extreme == "max" else -1.0

    rings = [rim]
    for k in range(1, segments + 1):
        t = k / float(segments)
        smooth = _smoothstep(t)
        s = 1.0 + (scale_target - 1.0) * smooth
        axis_offset = direction * length * t

        new_pts = rim_pts.copy()
        new_pts[:, axis] = plane_value + axis_offset
        new_pts[:, plane_axes] = center[plane_axes] + s * offsets

        start = len(verts)
        verts = np.vstack([verts, new_pts])
        rings.append(list(range(start, start + len(rim))))

    faces_list = [faces]
    rim_len = len(rim)
    for k in range(len(rings) - 1):
        ra = rings[k]
        rb = rings[k + 1]
        for i in range(rim_len):
            a0 = ra[i]
            a1 = ra[(i + 1) % rim_len]
            b0 = rb[i]
            b1 = rb[(i + 1) % rim_len]
            faces_list.append([[a0, a1, b1], [a0, b1, b0]])

    faces_all = np.vstack(faces_list)
    out = trimesh.Trimesh(vertices=verts, faces=faces_all, process=True)
    out.remove_unreferenced_vertices()
    out.merge_vertices()
    return out


@dataclass
class TaperMeta:
    end_name: str
    axis: int
    extreme: str
    length: float
    scale_target: float
    segments: int
    loop_vertices: int


def taper_stl_end(
    input_stl: str | Path,
    output_stl: str | Path,
    *,
    end_name: str = "yz_min_x",
    length: float = 14.0,
    scale_target: float = 0.75,
    segments: int = 12,
    tol_ratio: float = 0.02,
    verbose: bool = True,
) -> TaperMeta:
    input_stl = Path(input_stl)
    output_stl = Path(output_stl)

    mesh = trimesh.load_mesh(str(input_stl), process=False)
    loops = find_boundary_loops(mesh)
    infos = characterize_loops(mesh, loops)
    chosen = select_loop(mesh, loops, infos, end_name, tol_ratio=tol_ratio)
    axis, extreme = END_MAP[end_name]
    loop = loops[chosen["index"]]

    extruded = extrude_with_taper(
        mesh,
        loop,
        axis=axis,
        extreme=extreme,
        length=length,
        scale_target=scale_target,
        segments=segments,
    )
    extruded.export(output_stl)

    if verbose:
        print(
            f"[taper] end={end_name} axis={axis} extreme={extreme} "
            f"len={length} scale={scale_target} segs={segments} loop_n={len(loop)}"
        )
        print(f"[taper] wrote: {output_stl}")

    return TaperMeta(
        end_name=end_name,
        axis=axis,
        extreme=extreme,
        length=length,
        scale_target=scale_target,
        segments=segments,
        loop_vertices=len(loop),
    )


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Extrude and taper one outlet of an STL.")
    ap.add_argument("input_stl", type=Path, help="Input STL path")
    ap.add_argument("output_stl", type=Path, help="Output STL path")
    ap.add_argument(
        "--end",
        choices=sorted(END_MAP.keys()),
        default="yz_min_x",
        help="Which outlet to extrude (plane + min/max axis).",
    )
    ap.add_argument("--length", type=float, default=14.0, help="Extrusion length")
    ap.add_argument("--scale", type=float, default=0.75, help="Final cross-section scale factor")
    ap.add_argument("--segments", type=int, default=12, help="Interpolation segments for smooth taper")
    ap.add_argument(
        "--tol-ratio",
        type=float,
        default=0.02,
        help="Tolerance (fraction of axis span) for selecting the outlet near min/max.",
    )
    args = ap.parse_args()
    taper_stl_end(
        args.input_stl,
        args.output_stl,
        end_name=args.end,
        length=args.length,
        scale_target=args.scale,
        segments=args.segments,
        tol_ratio=args.tol_ratio,
        verbose=True,
    )


if __name__ == "__main__":
    _cli()
