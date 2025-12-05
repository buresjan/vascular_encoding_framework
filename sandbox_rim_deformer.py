#!/usr/bin/env python
"""Interactive sandbox to deform the master STL by reshaping its rim.

Run this file to get a PyVista window with sliders that move/rotate/scale the
rim (stored in ``basic_loop.vtp``) within its plane. The displacement between
the original and transformed rim is diffused into the STL
(``not_conduit_extruded.stl``) with a thin-plate spline warp so the geometry
adjusts smoothly without sharp transitions.

Example:
    python sandbox_rim_deformer.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyvista as pv


def build_plane_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (origin, x_axis, y_axis, normal) spanning the rim plane."""
    origin = points.mean(axis=0)
    pts0 = points - origin
    _, _, vh = np.linalg.svd(pts0, full_matrices=False)
    x_axis = vh[0]
    x_axis /= np.linalg.norm(x_axis)
    # Ensure orthonormal frame inside the plane
    normal = np.cross(vh[0], vh[1])
    normal /= np.linalg.norm(normal)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return origin, x_axis, y_axis, normal


def rim_transform(
    points: np.ndarray,
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tx: float,
    ty: float,
    rot_deg: float,
    scale: float,
) -> np.ndarray:
    """Move rim points within the plane using translation/rotation/scale."""
    origin, x_axis, y_axis, normal = frame
    rel = points - origin
    # Project to plane coordinates
    x = rel @ x_axis
    y = rel @ y_axis
    z = rel @ normal  # Should be near zero; keep to preserve thickness if any.

    rot = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    x_rot = cos_r * x - sin_r * y
    y_rot = sin_r * x + cos_r * y

    x_new = scale * x_rot + tx
    y_new = scale * y_rot + ty

    return origin + np.outer(x_new, x_axis) + np.outer(y_new, y_axis) + np.outer(z, normal)


def thin_plate_phi(r: np.ndarray) -> np.ndarray:
    """Thin-plate spline radial basis."""
    eps = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r * r * np.log(r + eps)
    out[r == 0.0] = 0.0
    return out


def build_tps_inverse(ctrl: np.ndarray, reg: float = 1e-3, reg_affine: float = 1e-2) -> np.ndarray:
    """Precompute inverse of the TPS system matrix for faster solves."""
    n = ctrl.shape[0]
    diff = ctrl[:, None, :] - ctrl[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    K = thin_plate_phi(r)
    K.flat[:: n + 1] += reg
    P = np.column_stack([np.ones(n), ctrl])
    upper = np.hstack([K, P])
    lower = np.hstack([P.T, reg_affine * np.eye(4)])
    M = np.vstack([upper, lower])
    return np.linalg.inv(M)


@dataclass
class RimSandbox:
    rim_path: Path
    stl_path: Path
    params: Dict[str, float] = field(
        default_factory=lambda: {
            "tx": 0.0,
            "ty": 0.0,
            "rotation": 0.0,
            "scale": 1.0,
            "strength": 1.0,
            "influence_center": 0.0,
            "influence_width": 10.0,
        }
    )

    def __post_init__(self) -> None:
        self.rim = pv.read(self.rim_path)
        self.stl = pv.read(self.stl_path)
        self.rim_points = np.asarray(self.rim.points)
        self.stl_base_points = np.asarray(self.stl.points)

        self.frame = build_plane_frame(self.rim_points)

        # Precompute TPS helpers (depends only on rim and STL points).
        self.tps_inv = build_tps_inverse(self.rim_points, reg=1e-3, reg_affine=1e-2)
        diff = self.stl_base_points[:, None, :] - self.rim_points[None, :, :]
        r_eval = np.linalg.norm(diff, axis=2)
        self.phi_eval = thin_plate_phi(r_eval)
        self.P_eval = np.column_stack(
            [np.ones(self.stl_base_points.shape[0]), self.stl_base_points]
        )

        # In-plane coordinate for localized influence (e.g., left/right).
        origin, x_axis, _, _ = self.frame
        self.rim_plane_x = (self.rim_points - origin) @ x_axis
        self.rim_x_min = self.rim_plane_x.min()
        self.rim_x_max = self.rim_plane_x.max()
        self.params["influence_center"] = float(self.rim_plane_x.mean())
        self.params["influence_width"] = max(
            1.0, float((self.rim_x_max - self.rim_x_min) * 0.4)
        )

        self.rim_deformed = self.rim.copy()
        self.stl_deformed = self.stl.copy()
        self.plotter = pv.Plotter()

    def update_meshes(self) -> None:
        """Recompute rim transform and diffuse displacement into the STL."""
        p = self.params
        rim_new = rim_transform(
            self.rim_points,
            self.frame,
            tx=p["tx"],
            ty=p["ty"],
            rot_deg=p["rotation"],
            scale=p["scale"],
        )
        width = max(p["influence_width"], 1e-3)
        influence = np.exp(-0.5 * ((self.rim_plane_x - p["influence_center"]) / width) ** 2)

        rim_disp = (rim_new - self.rim_points) * influence[:, None] * p["strength"]
        if np.linalg.norm(rim_disp, ord=np.inf) < 1e-9:
            rim_disp[:] = 0.0
        rim_target = self.rim_points + rim_disp
        self.rim_deformed.points = rim_target

        n = self.rim_points.shape[0]
        rhs = np.zeros((n + 4, 3))
        rhs[:n, :] = rim_disp
        sol = self.tps_inv @ rhs
        w = sol[:n, :]
        a = sol[n:, :]

        disp = self.phi_eval @ w + self.P_eval @ a
        self.stl_deformed.points = self.stl_base_points + disp

        self.plotter.render()

    def _add_sliders(self) -> None:
        """Wire up sliders for the interactive controls."""
        slider_specs = [
            ("tx", (-20.0, 20.0), "Translate U (plane)", self.params["tx"]),
            ("ty", (-20.0, 20.0), "Translate V (plane)", self.params["ty"]),
            ("rotation", (-60.0, 60.0), "Rotate (deg)", self.params["rotation"]),
            ("scale", (0.5, 1.5), "Scale (in-plane)", self.params["scale"]),
            ("strength", (0.2, 2.0), "Displacement strength", self.params["strength"]),
            (
                "influence_center",
                (self.rim_x_min - 5.0, self.rim_x_max + 5.0),
                "Influence center (rim U)",
                self.params["influence_center"],
            ),
            (
                "influence_width",
                (0.5, max(1.0, (self.rim_x_max - self.rim_x_min) * 1.2)),
                "Influence width",
                self.params["influence_width"],
            ),
        ]

        # Place sliders vertically down the left margin.
        y = 0.92
        for name, rng, title, value in slider_specs:
            def _make_cb(param_name: str):
                def _cb(val: float) -> None:
                    self.params[param_name] = val
                    self.update_meshes()

                return _cb

            self.plotter.add_slider_widget(
                _make_cb(name),
                rng=rng,
                value=value,
                title=title,
                pointa=(0.02, y),
                pointb=(0.32, y),
                style="modern",
                slider_width=0.012,
                tube_width=0.004,
                title_height=0.014,
            )
            y -= 0.06

    def _edges_or_poly(self, poly: pv.PolyData) -> pv.PolyData:
        """Return edge representation if available, else the input polyline."""
        edges = poly.extract_all_edges()
        return edges if edges.n_points > 0 else poly

    def _add_meshes(self) -> None:
        """Add STL and rim actors to the plotter."""
        rim_edges = self._edges_or_poly(self.rim)
        rim_def_edges = self._edges_or_poly(self.rim_deformed)

        self.plotter.add_mesh(
            self.stl_deformed,
            color="lightgray",
            opacity=0.65,
            specular=0.3,
            name="stl",
        )
        self.plotter.add_mesh(
            rim_edges,
            color="orangered",
            line_width=4,
            render_lines_as_tubes=True,
            name="rim_original",
        )
        self.plotter.add_mesh(
            rim_def_edges,
            color="cyan",
            line_width=4,
            render_lines_as_tubes=True,
            name="rim_deformed",
        )
        self.plotter.add_axes(line_width=2)
        self.plotter.add_text(
            "Rim sandbox\norange = original rim\ncyan = transformed rim\nuse sliders on the left\ninfluence sliders limit which rim segment moves",
            position="upper_right",
            font_size=10,
        )

    def show(self) -> None:
        """Launch the interactive window."""
        pv.global_theme.background = "white"
        pv.global_theme.cmap = "viridis"
        pv.global_theme.window_size = [1200, 800]

        self._add_meshes()
        self._add_sliders()
        self.update_meshes()
        self.plotter.show()


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    rim_path = repo_root / "basic_loop.vtp"
    stl_path = repo_root / "not_conduit_extruded.stl"
    if not rim_path.exists():
        raise FileNotFoundError(rim_path)
    if not stl_path.exists():
        raise FileNotFoundError(stl_path)

    sandbox = RimSandbox(rim_path=rim_path, stl_path=stl_path)
    sandbox.show()


if __name__ == "__main__":
    main()
