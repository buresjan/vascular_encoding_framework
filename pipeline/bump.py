"""
Bump the sim_conduit radius spline and export a bumped VCS map + preview PNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

from vascular_encoding_framework.centerline.centerline import Centerline  # noqa: E402
from vascular_encoding_framework.encoding.radius import Radius  # noqa: E402


def load_encoding(encoding_path: Path):
    root = pv.read(str(encoding_path))[0]
    cl_meta = json.loads("".join(root[0].field_data["_PYVISTA_USER_DICT"].tolist()))
    rad_meta = json.loads("".join(root[1].field_data["_PYVISTA_USER_DICT"].tolist()))["radius"]
    return cl_meta, rad_meta


def build_centerline(meta: dict) -> Centerline:
    cl = Centerline()
    cl.set_parameters(
        build=True,
        t0=meta["t0"],
        t1=meta["t1"],
        k=meta["k"],
        n_knots=meta["n_knots"],
        coeffs=np.array(meta["coeffs"]),
        extrapolation=meta["extrapolation"],
    )
    return cl


def build_radius(meta: dict) -> Radius:
    rd = Radius()
    coeffs = np.array(meta["feature vector"])
    rd.set_parameters(
        build=True,
        x0=meta["x0"],
        x1=meta["x1"],
        y0=meta["y0"],
        y1=meta["y1"],
        kx=meta["kx"],
        ky=meta["ky"],
        n_knots_x=meta["n_knots_x"],
        n_knots_y=meta["n_knots_y"],
        coeffs=coeffs.reshape(
            meta["n_knots_x"] + meta["kx"] + 1, meta["n_knots_y"] + meta["ky"] + 1
        ),
    )
    return rd


def gaussian_bump(tau, theta, tau0, theta0, amp, sigma_t, sigma_th):
    """Smooth bump with periodic theta wrapping."""
    dtheta = np.abs(theta - theta0)
    dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
    return amp * np.exp(-0.5 * ((tau - tau0) / sigma_t) ** 2 - 0.5 * (dtheta / sigma_th) ** 2)


def write_bumped_vtp(
    cl: Centerline,
    vcs_map_path: Path,
    out_path: Path,
    tau0: float,
    theta0: float,
    amp: float,
    sigma_t: float,
    sigma_th: float,
):
    """Apply bump on rho, map back to Cartesian, and save VTP."""
    mesh = pv.read(str(vcs_map_path))
    tau = mesh["tau"]
    theta = mesh["theta"]
    rho = mesh["rho"]

    bump = gaussian_bump(tau, theta, tau0, theta0, amp, sigma_t, sigma_th)
    rho_bump = rho + bump

    pts = cl.vcs_to_cartesian(tau=tau, theta=theta, rho=rho_bump, gridded=False)

    bumped = mesh.copy()
    bumped.points = pts
    bumped["rho"] = rho_bump
    bumped = bumped.compute_normals(point_normals=True, cell_normals=False, inplace=False)

    bumped.save(str(out_path))
    return bumped


def write_bumped_png(
    rd: Radius,
    png_path: Path,
    tau0: float,
    theta0: float,
    amp: float,
    sigma_t: float,
    sigma_th: float,
    grid_size: int = 400,
):
    """Evaluate bumped spline on a grid and save PNG."""
    tau = np.linspace(rd.x0, rd.x1, grid_size)
    theta = np.linspace(rd.y0, rd.y1, grid_size)
    T, Th = np.meshgrid(tau, theta, indexing="xy")

    base = rd.evaluate(tau, theta, grid=True)
    bump = gaussian_bump(T, Th, tau0, theta0, amp, sigma_t, sigma_th)
    bumped = base + bump

    fig, ax = plt.subplots(figsize=(6, 6))
    mesh = ax.pcolormesh(tau, theta, bumped, shading="auto", cmap="viridis")
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$\\theta$ (rad)")
    ax.set_title(f"Bump at (tau={tau0:.3f}, theta={theta0:.3f}), amp={amp}")
    fig.colorbar(mesh, ax=ax, label="$\\rho$")
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=300)
    return png_path
