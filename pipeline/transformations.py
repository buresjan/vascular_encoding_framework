"""
Transformations applied to the canonical VCS encoding before the downstream pipeline runs.

Supported transforms:
- Multiple Gaussian bumps on the rho field.
- Global scaling of the rho field.
- Centerline straightening toward the end-to-end chord while keeping the tau=0 rim normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

from vascular_encoding_framework.centerline.centerline import Centerline  # noqa: E402
from vascular_encoding_framework.encoding.radius import Radius  # noqa: E402
from vascular_encoding_framework.encoding.vessel_encoding import VesselAnatomyEncoding  # noqa: E402

from .bump import build_centerline, build_radius, gaussian_bump


@dataclass(frozen=True)
class BumpSpec:
    tau0: float
    theta0: float
    amp: float
    sigma_t: float
    sigma_theta: float

    @classmethod
    def from_any(cls, data, defaults: dict) -> "BumpSpec":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise TypeError(f"Unsupported bump specification type: {type(data)}")

        return cls(
            tau0=float(data.get("tau0", defaults["tau0"])),
            theta0=float(data.get("theta0", defaults["theta0"])),
            amp=float(data.get("amp", defaults["amp"])),
            sigma_t=float(data.get("sigma_t", defaults["sigma_t"])),
            sigma_theta=float(data.get("sigma_theta", defaults["sigma_theta"])),
        )


def normalize_bump_specs(
    bumps: Sequence[BumpSpec | dict] | None,
    *,
    tau0: float,
    theta0: float,
    bump_amp: float,
    sigma_t: float,
    sigma_theta: float,
) -> list[BumpSpec]:
    """
    Normalize user-provided bumps into a list of BumpSpec, falling back to a single bump.
    """
    defaults = dict(tau0=tau0, theta0=theta0, amp=bump_amp, sigma_t=sigma_t, sigma_theta=sigma_theta)
    specs: list[BumpSpec] = []

    if bumps:
        for b in bumps:
            spec = BumpSpec.from_any(b, defaults)
            if abs(spec.amp) > 0:
                specs.append(spec)
    elif abs(bump_amp) > 0:
        specs.append(BumpSpec(**defaults))

    return specs


def build_bump_field(tau: np.ndarray, theta: np.ndarray, specs: Iterable[BumpSpec]) -> np.ndarray:
    """Sum multiple Gaussian bumps."""
    total = np.zeros_like(tau, dtype=float)
    for spec in specs:
        total += gaussian_bump(
            tau,
            theta,
            tau0=spec.tau0,
            theta0=spec.theta0,
            amp=spec.amp,
            sigma_t=spec.sigma_t,
            sigma_th=spec.sigma_theta,
        )
    return total


def straighten_centerline(
    cl_meta: dict,
    *,
    strength: float,
    exponent: float = 2.0,
    preserve_ends: int = 4,
) -> Centerline:
    """
    Move centerline control points toward the end-to-end chord.

    - strength in [0, 1] controls the fraction of the offset removed (1 = fully projected).
    - Points further from the chord are moved more aggressively (dist**exponent weighting).
    - The first/last `preserve_ends` control points are kept to preserve the rim normal at tau=0.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    coeffs = np.array(cl_meta["coeffs"])
    n = len(coeffs)
    preserve_ends = int(np.clip(preserve_ends, 0, n // 2))

    if strength <= 0 or n == 0:
        return build_centerline(cl_meta)

    start = coeffs[0]
    end = coeffs[-1]
    chord = end - start
    norm = np.linalg.norm(chord)
    if norm < 1e-9:
        return build_centerline(cl_meta)

    direction = chord / norm
    projections = start + np.dot(coeffs - start, direction)[:, None] * direction
    offsets = coeffs - projections
    dist = np.linalg.norm(offsets, axis=1)

    inner_slice = dist[preserve_ends : n - preserve_ends]
    max_dist = float(inner_slice.max() if inner_slice.size else dist.max())
    if max_dist < 1e-12:
        return build_centerline(cl_meta)

    weights = (dist / max_dist) ** exponent
    weights *= strength
    weights = np.clip(weights, 0.0, 1.0)

    new_coeffs = coeffs.copy()
    for i in range(n):
        if i < preserve_ends or i >= n - preserve_ends:
            continue
        if weights[i] <= 0:
            continue
        new_coeffs[i] = coeffs[i] - weights[i] * offsets[i]

    new_meta = dict(cl_meta)
    new_meta["coeffs"] = new_coeffs
    return build_centerline(new_meta)


def scale_radius(rd: Radius, size_scale: float) -> Radius:
    """Uniformly scale a Radius spline."""
    if size_scale <= 0:
        raise ValueError("size_scale must be positive.")
    new = Radius()
    new.set_parameters(
        build=True,
        x0=rd.x0,
        x1=rd.x1,
        kx=rd.kx,
        ky=rd.ky,
        n_knots_x=rd.n_knots_x,
        n_knots_y=rd.n_knots_y,
        extra_x=rd.extra_x,
        extra_y=rd.extra_y,
        coeffs=np.array(rd.coeffs) * float(size_scale),
    )
    return new


def fit_radius(
    tau: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    *,
    rad_meta: dict,
    centerline: Centerline,
    laplacian_penalty: float,
) -> Radius:
    """Fit a Radius spline to the provided (tau, theta, rho) samples."""
    pts = np.column_stack((tau, theta, rho))
    return Radius.from_points(
        pts,
        tau_knots=rad_meta["n_knots_x"],
        theta_knots=rad_meta["n_knots_y"],
        laplacian_penalty=laplacian_penalty,
        cl=centerline,
        debug=False,
    )


def write_radius_png(rd: Radius, png_path, title: str, grid_size: int = 400):
    """Render the rho field for visualization."""
    tau = np.linspace(rd.x0, rd.x1, grid_size)
    theta = np.linspace(rd.y0, rd.y1, grid_size)
    rho = rd.evaluate(tau, theta, grid=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    mesh = ax.pcolormesh(tau, theta, rho, shading="auto", cmap="viridis")
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$\\theta$ (rad)")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="$\\rho$")
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=300)
    return png_path


def save_encoding(centerline: Centerline, radius: Radius, out_path) -> None:
    """Write a VCS encoding (centerline + radius) to disk."""
    enc = VesselAnatomyEncoding()
    enc.set_centerline(centerline)
    enc.set_data(radius=radius)
    mb = enc.to_multiblock(add_attributes=True, tau_res=150, theta_res=80)
    mb.save(str(out_path))


def transform_vcs_map(
    cl_meta: dict,
    rad_meta: dict,
    vcs_map_path,
    out_vtp_path,
    png_path,
    *,
    bumps: Sequence[BumpSpec],
    size_scale: float,
    straighten_strength: float,
    straighten_exponent: float,
    straighten_preserve: int,
    rho_min: float,
    radius_fit_laplacian: float,
) -> tuple[Centerline, Radius, pv.PolyData]:
    """
    Apply VCS transformations and write an updated map + preview PNG.

    Returns the transformed centerline, radius, and the updated VTP mesh.
    """
    if size_scale <= 0:
        raise ValueError("size_scale must be positive.")
    if rho_min < 0:
        raise ValueError("rho_min must be non-negative.")

    base_radius = build_radius(rad_meta)
    cl = straighten_centerline(
        cl_meta,
        strength=straighten_strength,
        exponent=straighten_exponent,
        preserve_ends=straighten_preserve,
    )

    mesh = pv.read(str(vcs_map_path))
    tau = mesh["tau"]
    theta = mesh["theta"]
    rho_base = mesh["rho"]

    bump_field = build_bump_field(tau, theta, bumps)
    has_bumps = bumps and len(bumps) > 0
    scale_applied = not np.isclose(size_scale, 1.0)

    if has_bumps:
        rho_target = rho_base * size_scale + bump_field
        radius = fit_radius(
            tau,
            theta,
            rho_target,
            rad_meta=rad_meta,
            centerline=cl,
            laplacian_penalty=radius_fit_laplacian,
        )
        rho_used = radius(tau, theta, grid=False)
    elif scale_applied:
        radius = scale_radius(base_radius, size_scale)
        rho_used = radius(tau, theta, grid=False)
    else:
        radius = base_radius
        rho_used = rho_base

    rho_used = np.asarray(rho_used).ravel()
    rho_used = np.maximum(rho_min, rho_used)

    pts = cl.vcs_to_cartesian(tau=tau, theta=theta, rho=rho_used, gridded=False)

    bumped = mesh.copy()
    bumped.points = pts
    bumped["rho"] = rho_used
    bumped = bumped.compute_normals(point_normals=True, cell_normals=False, inplace=False)
    bumped.save(str(out_vtp_path))

    title = f"rho map (scale={size_scale:.3f}, bumps={len(bumps)}, straighten={straighten_strength:.3f})"
    write_radius_png(radius, png_path, title=title)

    return cl, radius, bumped
