#!/usr/bin/env python3
"""
Cap the four axis-aligned open ends of an STL and report watertightness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import trimesh

# Allow this module to run both as part of the package and as a script.
if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    __package__ = "pipeline"

from .repair import (
    add_fan_cap,
    boundary_edges,
    connected_components_from_edges,
    edge_stats,
    find_four_axis_end_loops,
    remove_degenerate_faces_compat,
    remove_duplicate_faces_compat,
)


@dataclass(frozen=True)
class CapReport:
    watertight: bool
    boundary_loops: int
    boundary_edges: int
    nonmanifold_edges: int


def cap_four_open_axis_aligned_ends(
    input_stl: str | Path,
    output_capped_stl: str | Path,
    *,
    merge_digits: int = 6,
    plane_range_tol: float | None = None,
    verbose: bool = True,
) -> CapReport:
    """
    Detect the four axis-aligned open boundary loops, cap them with fans, and save the result.
    Returns a report containing watertightness and boundary/nonmanifold edge counts.
    """
    input_stl = Path(input_stl)
    output_capped_stl = Path(output_capped_stl)

    mesh = trimesh.load_mesh(str(input_stl), process=False)
    mesh.merge_vertices(digits_vertex=merge_digits)
    mesh = remove_duplicate_faces_compat(mesh)
    mesh = remove_degenerate_faces_compat(mesh)
    mesh.remove_unreferenced_vertices()

    ends = find_four_axis_end_loops(mesh, plane_range_tol=plane_range_tol)
    if verbose:
        print("\n[cap] Detected end planes for capping:")
        for e in sorted(ends, key=lambda e: (e.axis, e.plane_value)):
            ax = "xyz"[e.axis]
            print(f"  {ax} = {e.plane_value:.6f} (loop vertices={len(e.loop)})")

    capped = mesh.copy()
    for end in ends:
        add_fan_cap(capped, loop=end.loop, axis=end.axis, plane_value=end.plane_value)

    capped.merge_vertices(digits_vertex=merge_digits)
    capped = remove_duplicate_faces_compat(capped)
    capped = remove_degenerate_faces_compat(capped)
    capped.remove_unreferenced_vertices()
    capped.fix_normals()

    be = boundary_edges(capped)
    loops = connected_components_from_edges(be)
    es = edge_stats(capped)
    report = CapReport(
        watertight=bool(capped.is_watertight),
        boundary_loops=len(loops),
        boundary_edges=es["boundary_edges"],
        nonmanifold_edges=es["nonmanifold_edges"],
    )

    capped.export(str(output_capped_stl))
    if verbose:
        print(
            f"[cap] watertight={report.watertight} "
            f"boundary_loops={report.boundary_loops} "
            f"boundary_edges={report.boundary_edges} "
            f"nonmanifold_edges={report.nonmanifold_edges}"
        )
        print(f"[cap] wrote capped STL: {output_capped_stl}")

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_stl", type=str)
    ap.add_argument("output_capped_stl", type=str)
    ap.add_argument("--merge-digits", type=int, default=6)
    ap.add_argument("--plane-range-tol", type=float, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cap_four_open_axis_aligned_ends(
        input_stl=args.input_stl,
        output_capped_stl=args.output_capped_stl,
        merge_digits=args.merge_digits,
        plane_range_tol=args.plane_range_tol,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
