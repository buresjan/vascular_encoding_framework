"""
High-level pipeline to bump the sim_conduit VCS radius, extract rims, deform a partner STL,
combine meshes, optionally taper an outlet, clip, and repair the final geometry.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import pyvista as pv

from .bump import build_centerline, build_radius, load_encoding, write_bumped_png, write_bumped_vtp
from .deform import deform_rim_to_target
from .mesh_ops import append_meshes, clip_bottom, hash_params, stl_from_vtp
from .repair import repair_surface_four_open_ends
from .rim import extract_rim
from .taper_stl_end import taper_stl_end


def run_pipeline(
    tau0: float,
    theta0: float,
    bump_amp: float,
    sigma_t: float = 0.05,
    sigma_theta: float = 0.25,
    rim_tol: float = 1e-3,
    deform_r1: float = 5.0,
    deform_r2: float = 20.0,
    clip_offset: float = 0.5,
    taper_enabled: bool = False,
    taper_end: str = "yz_min_x",
    taper_length: float = 14.0,
    taper_scale: float = 0.75,
    taper_segments: int = 12,
    taper_tol_ratio: float = 0.02,
    taper_verbose: bool = True,
    repair_pitch: float | None = None,
    repair_closing_radius: int = 1,
    repair_target_voxels_min_dim: int = 80,
    repair_merge_digits: int = 6,
    repair_keep_area_ratio: float = 0.01,
    repair_plane_range_tol: float | None = None,
    repair_verbose: bool = True,
    output_dir: Path = Path("."),
    temp_dir: Path | None = None,
    *,
    encoding_path: Path = Path("pipeline/sim_conduit/Encoding/encoding.vtm"),
    vcs_map_path: Path = Path("pipeline/sim_conduit/Encoding/vcs_map.vtp"),
    partner_stl: Path = Path("pipeline/not_conduit_extruded_canon.stl"),
    partner_orig_rim: Path = Path("pipeline/basic_loop_canon.vtp"),
) -> Tuple[Path, str]:
    """
    Execute full pipeline and return path to the repaired combined STL and the parameter hash.
    Intermediates are kept in a temp directory (default: per-run folder under output_dir).

    Steps:
    1) Bump VCS radius by (tau0, theta0, bump_amp) -> bumped VTP + PNG.
    2) Extract rim at tau≈0 from bumped VTP.
    3) Convert bumped VTP to STL.
    4) Deform partner STL to match new rim.
    5) Append partner and bumped STLs.
    6) (Optional) Taper one chosen open end.
    7) Clip bottom, rebase to Z=0.
    8) Repair merged surface while keeping the 4 outlets open via voxel remeshing.
    """
    params = dict(
        tau0=tau0,
        theta0=theta0,
        bump_amp=bump_amp,
        sigma_t=sigma_t,
        sigma_theta=sigma_theta,
        rim_tol=rim_tol,
        deform_r1=deform_r1,
        deform_r2=deform_r2,
        clip_offset=clip_offset,
        taper_enabled=taper_enabled,
        taper_end=taper_end,
        taper_length=taper_length,
        taper_scale=taper_scale,
        taper_segments=taper_segments,
        taper_tol_ratio=taper_tol_ratio,
        repair_pitch=repair_pitch,
        repair_closing_radius=repair_closing_radius,
        repair_target_voxels_min_dim=repair_target_voxels_min_dim,
        repair_merge_digits=repair_merge_digits,
        repair_keep_area_ratio=repair_keep_area_ratio,
        repair_plane_range_tol=repair_plane_range_tol,
    )
    uid = hash_params(params)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"sim_{uid}.stl"

    if temp_dir is None:
        # Keep intermediates in a per-run temp folder alongside outputs.
        tmp = Path(
            tempfile.mkdtemp(prefix=f"sim_{uid}_tmp_", dir=str(output_dir))
        )
    else:
        tmp = Path(temp_dir).expanduser().resolve()
        tmp.mkdir(parents=True, exist_ok=True)

    # Build encoding primitives
    cl_meta, rad_meta = load_encoding(encoding_path)
    cl = build_centerline(cl_meta)
    rd = build_radius(rad_meta)

    # 1) Bump encoding
    bumped_vtp = tmp / "vcs_map_bump.vtp"
    bump_png = tmp / "vcs_map_bump.png"
    write_bumped_png(
        rd,
        png_path=bump_png,
        tau0=tau0,
        theta0=theta0,
        amp=bump_amp,
        sigma_t=sigma_t,
        sigma_th=sigma_theta,
    )
    write_bumped_vtp(
        cl,
        vcs_map_path=vcs_map_path,
        out_path=bumped_vtp,
        tau0=tau0,
        theta0=theta0,
        amp=bump_amp,
        sigma_t=sigma_t,
        sigma_th=sigma_theta,
    )

    # 2) Rim extraction and STL conversion
    rim_path = tmp / "rim_transformed.vtp"
    extract_rim(bumped_vtp, rim_path, tol=rim_tol)

    sim_stl = tmp / "sim_conduit_transformed.stl"
    stl_from_vtp(bumped_vtp, sim_stl)

    # 3) Deform partner STL to match rim
    partner_deformed = tmp / "not_conduit_extruded_canon_transformed.stl"
    deform_rim_to_target(
        stl_path=str(partner_stl),
        orig_rim_vtp=str(partner_orig_rim),
        target_rim_vtp=str(rim_path),
        out_stl_path=str(partner_deformed),
        r1=deform_r1,
        r2=deform_r2,
    )

    # 5) Combine meshes
    combined_tmp = tmp / f"sim_{uid}_tmp.stl"
    append_meshes(sim_stl, partner_deformed, combined_tmp)

    # 6) Optional outlet taper
    clip_source = combined_tmp
    if taper_enabled:
        tapered_tmp = tmp / f"sim_{uid}_tmp_tapered.stl"
        taper_stl_end(
            combined_tmp,
            tapered_tmp,
            end_name=taper_end,
            length=taper_length,
            scale_target=taper_scale,
            segments=taper_segments,
            tol_ratio=taper_tol_ratio,
            verbose=taper_verbose,
        )
        clip_source = tapered_tmp

    # 7) Clip and rebase to Z=0
    clipped = tmp / f"sim_{uid}_clipped.stl"
    clip_bottom(clip_source, clipped, clip_offset=clip_offset)

    # 8) Repair via voxel remeshing
    repaired = tmp / f"sim_{uid}_repaired.stl"
    pitch_used = repair_surface_four_open_ends(
        clipped,
        repaired,
        pitch=repair_pitch,
        target_voxels_min_dim=repair_target_voxels_min_dim,
        closing_radius=repair_closing_radius,
        merge_digits=repair_merge_digits,
        keep_area_ratio=repair_keep_area_ratio,
        plane_range_tol=repair_plane_range_tol,
        verbose=repair_verbose,
    )
    if repair_verbose:
        print(f"[repair] pitch used: {pitch_used}")

    # Reload with pyvista for consistent STL writing
    pv.read(str(repaired)).save(str(final_path))

    return final_path, uid
