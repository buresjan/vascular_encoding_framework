#!/usr/bin/env python3
"""Extrude the open rim of a clipped STL and bend it until the rim faces +Z.

Defaults target the rim stored in rpa_rim.vtp produced by clip_and_extract_rim.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pyvista as pv

# RPA
# TARGET_NORMAL = np.array([0.0, 0.0, 1.0])
# DEFAULT_RIM_NORMAL = np.array([-0.475, 0.501, 0.724])

TARGET_NORMAL = np.array([1.0, 0.0, 0.0])
# DEFAULT_RIM_NORMAL = np.array([1., 0, 0.])
DEFAULT_RIM_NORMAL = np.array([1., 0, 0.25])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bend-extrude an open rim so its far end is aligned with the XY plane "
            "(normal = +Z) and save the result."
        )
    )
    parser.add_argument(
        "--input-mesh",
        type=Path,
        default=Path("rpa_clipped.stl"),
        help="Clipped STL that has the open rim (default: rpa_clipped.stl).",
    )
    parser.add_argument(
        "--rim",
        type=Path,
        default=Path("rpa_rim.vtp"),
        help="Polyline describing the rim to extrude (default: rpa_rim.vtp).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rpa_extruded_bent.stl"),
        help="Final STL with bent extrusion merged in (default: rpa_extruded_bent.stl).",
    )
    parser.add_argument(
        "--extension-output",
        type=Path,
        default=Path("rpa_bent_extension.vtp"),
        help="Optional save path for the bent extension only (default: rpa_bent_extension.vtp).",
    )
    parser.add_argument(
        "--length",
        type=float,
        help="Total extrusion length. If omitted, 1.5x the rim diagonal is used.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=24,
        help="How many intermediate loops to create along the bend (default: 24).",
    )
    parser.add_argument(
        "--rim-normal-hint",
        type=float,
        nargs=3,
        default=DEFAULT_RIM_NORMAL,
        metavar=("NX", "NY", "NZ"),
        help="Optional hint for the rim normal to lock its direction.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Save a screenshot instead of opening a window when provided.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip the viewer (overridden when --screenshot is used).",
    )
    return parser.parse_args()


def normalize(vec: Iterable[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("Vector must be non-zero.")
    return arr / norm


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9 or angle == 0.0:
        return np.eye(3)
    axis = axis / norm
    kx, ky, kz = axis
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def best_fit_normal(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    _, vecs = np.linalg.eigh(cov)
    return normalize(vecs[:, 0])


def order_rim_points(
    rim_points: np.ndarray, normal_hint: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project rim points to their best-fit plane and sort them by angle."""
    pts = np.asarray(rim_points, dtype=float)
    center = pts.mean(axis=0)
    normal = best_fit_normal(pts)
    if normal_hint is not None and np.dot(normal, normal_hint) < 0:
        normal *= -1.0

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, normal))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])

    u = normalize(np.cross(ref, normal))
    v = normalize(np.cross(normal, u))
    coords_2d = np.stack(((pts - center) @ u, (pts - center) @ v), axis=1)
    angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])
    order = np.argsort(angles)
    return pts[order], normal, center


def build_bent_loops(
    ordered_points: np.ndarray,
    rim_normal: np.ndarray,
    center: np.ndarray,
    length: float,
    steps: int,
) -> tuple[list[np.ndarray], np.ndarray, float, list[np.ndarray]]:
    rim_normal = normalize(rim_normal)
    target = TARGET_NORMAL
    axis = np.cross(rim_normal, target)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        axis = rim_normal
    else:
        axis = axis / axis_norm
    angle = float(np.arccos(np.clip(np.dot(rim_normal, target), -1.0, 1.0)))

    offsets = ordered_points - center
    loops: list[np.ndarray] = []
    centers: list[np.ndarray] = [center.copy()]
    step_len = float(length) / max(steps - 1, 1)

    for i in range(steps):
        t = 0.0 if steps == 1 else i / (steps - 1)
        ang = angle * t
        R = rotation_matrix(axis, ang)
        direction = normalize(R @ rim_normal)
        if i > 0:
            centers.append(centers[-1] + direction * step_len)
        loop = centers[-1] + (R @ offsets.T).T
        loops.append(loop)

    return loops, axis, angle, centers


def loft_quads(loops: list[np.ndarray]) -> pv.PolyData:
    n_loops = len(loops)
    if n_loops < 2:
        raise ValueError("Need at least two loops to loft an extrusion.")
    n_pts = loops[0].shape[0]
    vertices = np.vstack(loops)
    faces = []
    for k in range(n_loops - 1):
        base = k * n_pts
        nxt = (k + 1) * n_pts
        for i in range(n_pts):
            j = (i + 1) % n_pts
            faces.extend([3, base + i, base + j, nxt + j])
            faces.extend([3, base + i, nxt + j, nxt + i])
    faces_arr = np.asarray(faces, dtype=np.int64)
    return pv.PolyData(vertices, faces=faces_arr)


def polyline_from_loop(loop: np.ndarray) -> pv.PolyData:
    n = loop.shape[0]
    line = np.concatenate([[n], np.arange(n, dtype=np.int64)])
    return pv.PolyData(loop.copy(), lines=line)


def main() -> None:
    args = parse_args()

    if not args.input_mesh.exists():
        raise SystemExit(f"Input mesh not found: {args.input_mesh}")
    if not args.rim.exists():
        raise SystemExit(f"Rim file not found: {args.rim}")

    rim = pv.read(args.rim).clean()
    rim_points, rim_normal, rim_center = order_rim_points(
        rim.points, normal_hint=np.asarray(args.rim_normal_hint, dtype=float)
    )

    rim_diag = np.linalg.norm(np.array([rim.bounds[1] - rim.bounds[0], rim.bounds[3] - rim.bounds[2], rim.bounds[5] - rim.bounds[4]]))
    length = float(args.length) if args.length is not None else rim_diag * 1.5

    loops, axis, angle, centers = build_bent_loops(
        ordered_points=rim_points,
        rim_normal=rim_normal,
        center=rim_center,
        length=length,
        steps=args.steps,
    )
    extension = loft_quads(loops).clean()

    base_mesh = pv.read(args.input_mesh)
    combined = base_mesh.merge(extension, merge_points=True, tolerance=1e-5).clean()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.extension_output.parent.mkdir(parents=True, exist_ok=True)
    combined.save(args.output, binary=True)
    extension.save(args.extension_output, binary=True)

    print(f"Combined STL written to {args.output}")
    print(f"Extension only written to {args.extension_output}")
    print(f"Rim normal (fit) {_fmt_vec(rim_normal)}, rotation angle to +Z {np.degrees(angle):.2f} deg")
    print(f"Extrusion length {length:.3f}, steps {args.steps}")

    if args.no_show and not args.screenshot:
        return

    centerline = pv.Spline(np.asarray(centers))
    start_loop = polyline_from_loop(loops[0])
    end_loop = polyline_from_loop(loops[-1])

    plotter = pv.Plotter(title="Bent rim extrusion")
    plotter.add_mesh(base_mesh, color="lightgray", opacity=0.35, show_edges=False, name="base")
    plotter.add_mesh(extension, color="salmon", opacity=0.75, show_edges=True, name="extension")
    plotter.add_mesh(start_loop, color="tomato", line_width=4, name="start_loop")
    plotter.add_mesh(end_loop, color="mediumblue", line_width=4, name="end_loop")
    plotter.add_mesh(centerline, color="gold", line_width=3, name="centerline")
    plotter.add_axes()
    plotter.add_text("Red: source rim, Blue: final rim, Yellow: centerline", font_size=10)

    if args.screenshot:
        plotter.show(screenshot=str(args.screenshot))
    else:
        plotter.show()


def _fmt_vec(vec: np.ndarray) -> str:
    vec = np.asarray(vec).ravel()
    return "[" + ", ".join(f"{v:.3f}" for v in vec) + "]"


if __name__ == "__main__":
    main()
