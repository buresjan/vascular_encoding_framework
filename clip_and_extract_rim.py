#!/usr/bin/env python3
"""Clip rpa.stl with a fixed plane and export the rim of the cut.

Defaults match the requested plane:
origin [126.772, 137.760, 84.974], normal [0.475, -0.501, -0.724].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

# DEFAULT_ORIGIN = [189.915, 122.552, 80.136] LPA BASIC
# DEFAULT_NORMAL = [-1.0, 0., 0.]

# DEFAULT_ORIGIN = [126.772, 137.760, 84.974]  RPA
# DEFAULT_NORMAL = [0.475, -0.501, -0.724]  RPA


DEFAULT_ORIGIN = [189.021, 156.72245900771176, 78.19331323725298]
DEFAULT_NORMAL = [-1.0, 0, -0.25]

# Coords: (189.915, 122.552, 80.136)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clip an STL with a plane, save the clipped mesh, and export the "
            "intersection rim for downstream extrusion."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("rpa.stl"),
        help="Input STL path (default: rpa.stl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rpa_clipped.stl"),
        help="Where to write the clipped STL (default: rpa_clipped.stl).",
    )
    parser.add_argument(
        "--rim-output",
        type=Path,
        default=Path("rpa_rim.vtp"),
        help="Path for the rim/intersection polyline (default: rpa_rim.vtp).",
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=DEFAULT_ORIGIN,
        metavar=("X", "Y", "Z"),
        help="Plane origin (default matches the requested plane).",
    )
    parser.add_argument(
        "--normal",
        type=float,
        nargs=3,
        default=DEFAULT_NORMAL,
        metavar=("NX", "NY", "NZ"),
        help="Plane normal (default matches the requested plane).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Keep the opposite side of the plane.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the interactive viewer.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Optional screenshot path; forces rendering even with --no-show.",
    )
    return parser.parse_args()


def _normalize(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Vector must have 3 components.")
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Vector must be non-zero.")
    return arr / norm


def _format_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in vec) + "]"


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input STL not found: {args.input}")

    origin = np.asarray(args.origin, dtype=float)
    normal = _normalize(args.normal)

    mesh = pv.read(args.input)

    clipped = mesh.clip(origin=origin, normal=normal, invert=args.invert)
    if clipped.n_points == 0:
        raise SystemExit("Plane does not intersect the mesh; nothing to save.")

    # Intersection curve of the original mesh with the cutting plane.
    rim = mesh.slice(origin=origin, normal=normal, generate_triangles=False)
    rim.clean(inplace=True)
    if rim.n_points == 0:
        raise SystemExit("Failed to compute rim; slice produced no intersection.")
    if rim.lines.size == 0:
        # Fallback for datasets that output polygons instead of polylines.
        rim_fallback = rim.extract_all_edges().clean()
        if rim_fallback.n_points == 0:
            raise SystemExit("Failed to compute rim; no intersection lines were found.")
        rim = rim_fallback

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rim_output.parent.mkdir(parents=True, exist_ok=True)
    clipped.save(args.output)
    rim.save(args.rim_output)

    plane_size = float(mesh.length) * 0.35
    plane = pv.Plane(center=origin, direction=normal, i_size=plane_size, j_size=plane_size)
    rim_color = "tomato"

    print(f"Clipped mesh written to {args.output}")
    print(f"Rim/intersection written to {args.rim_output}")
    print(f"Plane origin={_format_vec(origin)}, normal={_format_vec(normal)}, invert={args.invert}")

    if args.no_show and not args.screenshot:
        return

    plotter = pv.Plotter(title="Clipped STL with rim")
    plotter.add_mesh(clipped, name="clipped", color="lightgray", opacity=0.5, show_edges=False)
    plotter.add_mesh(rim, name="rim", color=rim_color, line_width=6, lighting=False)
    plotter.add_mesh(plane, name="plane", color="deepskyblue", opacity=0.15, style="wireframe")
    plotter.add_axes()

    if args.screenshot:
        plotter.show(screenshot=str(args.screenshot))
    else:
        plotter.show()


if __name__ == "__main__":
    main()
