"""
High-level pipeline to bump the sim_conduit VCS radius, extract rims, deform a partner STL,
combine meshes, clip, and cap the final watertight geometry.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import pyvista as pv
import trimesh

from .bump import build_centerline, build_radius, load_encoding, write_bumped_png, write_bumped_vtp
from .deform import deform_rim_to_target
from .mesh_fix import cap_outlets_pipeline, repair_pipeline, voxel_rebuild_pipeline
from .mesh_ops import append_meshes, clip_bottom, hash_params, stl_from_vtp
from .rim import extract_rim


def _load_trimesh_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    return mesh


def _watertight_with_mesh_fix(
    mesh_path: Path,
    *,
    voxel_pitch: float | None,
    verbose: bool = False,
):
    """
    Run the mesh_fix pipeline (repair -> cap -> voxel rebuild) on the mesh at mesh_path.
    Returns the watertight trimesh and the pitch that was actually used.
    """
    mesh = _load_trimesh_mesh(mesh_path)
    mesh = repair_pipeline(mesh, verbose=verbose)
    mesh = cap_outlets_pipeline(mesh, verbose=verbose)
    return voxel_rebuild_pipeline(mesh, pitch=voxel_pitch, verbose=verbose)


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
    hole_size: float = 1_000.0,
    voxel_pitch: float | None = 1.0,
    mesh_fix_verbose: bool = False,
    output_dir: Path = Path("."),
    *,
    encoding_path: Path = Path("pipeline/sim_conduit/Encoding/encoding.vtm"),
    vcs_map_path: Path = Path("pipeline/sim_conduit/Encoding/vcs_map.vtp"),
    partner_stl: Path = Path("pipeline/not_conduit_extruded_canon.stl"),
    partner_orig_rim: Path = Path("pipeline/basic_loop_canon.vtp"),
) -> Tuple[Path, str]:
    """
    Execute full pipeline and return path to capped combined STL and the parameter hash.

    Steps:
    1) Bump VCS radius by (tau0, theta0, bump_amp) -> bumped VTP + PNG.
    2) Extract rim at tau≈0 from bumped VTP.
    3) Convert bumped VTP to STL.
    4) Deform partner STL to match new rim.
    5) Append partner and bumped STLs.
    6) Clip bottom, rebase to Z=0.
    7) Repair + cap 4 outlets + voxel-rebuild a watertight outer surface.
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
        hole_size=hole_size,
        voxel_pitch=voxel_pitch,
    )
    uid = hash_params(params)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"sim_{uid}.stl"

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

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

        # 4) Combine and clip/cap
        combined_tmp = tmp / f"sim_{uid}_tmp.stl"
        append_meshes(sim_stl, partner_deformed, combined_tmp)

        clipped_capped = tmp / f"sim_{uid}_clipped.stl"
        clip_bottom(combined_tmp, clipped_capped, clip_offset=clip_offset)

        watertight_mesh, _pitch_used = _watertight_with_mesh_fix(
            clipped_capped,
            voxel_pitch=voxel_pitch,
            verbose=mesh_fix_verbose,
        )
        if mesh_fix_verbose:
            print(f"[mesh_fix] voxel pitch used: {_pitch_used}")
        watertight_tmp = tmp / f"sim_{uid}_watertight.stl"
        watertight_mesh.export(watertight_tmp)

        # Reload with pyvista for consistent STL writing
        pv.read(str(watertight_tmp)).save(str(final_path))

    return final_path, uid
