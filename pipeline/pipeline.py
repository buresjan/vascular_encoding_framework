"""
High-level pipeline to bump the sim_conduit VCS radius, extract rims, deform a partner STL,
combine meshes, clip, taper an open end, repair, and cap the final geometry.
"""

from __future__ import annotations

import sys
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import pyvista as pv

# Allow this module to be executed both as part of the package and as a script.
if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    __package__ = "pipeline"

from .cap import CapReport, cap_four_open_axis_aligned_ends
from .bump import load_encoding
from .deform import deform_rim_to_target
from .mesh_ops import append_meshes, clip_bottom, hash_params, stl_from_vtp
from .repair import repair_surface_four_open_ends
from .rim import extract_rim
from .taper_stl_end import taper_stl_end
from .transformations import (
    BumpSpec,
    normalize_bump_specs,
    save_encoding,
    transform_vcs_map,
)


@dataclass(frozen=True)
class PipelinePaths:
    """
    Collect all paths for one pipeline run so orchestration stays linear and explicit.
    """

    output_dir: Path
    temp_dir: Path
    final: Path
    bumped_vtp: Path
    bump_png: Path
    rim: Path
    sim_stl: Path
    partner_deformed: Path
    combined_tmp: Path
    clipped: Path
    tapered: Path
    repaired: Path
    capped: Path
    encoding: Path

    @classmethod
    def build(cls, uid: str, output_dir: Path, temp_dir: Path | None) -> "PipelinePaths":
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if temp_dir is None:
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"sim_{uid}_tmp_", dir=str(output_dir)))
        else:
            tmp_dir = Path(temp_dir).expanduser().resolve()
            tmp_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            output_dir=output_dir,
            temp_dir=tmp_dir,
            final=output_dir / f"sim_{uid}.stl",
            bumped_vtp=tmp_dir / "vcs_map_bump.vtp",
            bump_png=tmp_dir / "vcs_map_bump.png",
            rim=tmp_dir / "rim_transformed.vtp",
            sim_stl=tmp_dir / "sim_conduit_transformed.stl",
            partner_deformed=tmp_dir / "not_conduit_extruded_canon_transformed.stl",
            combined_tmp=tmp_dir / f"sim_{uid}_tmp.stl",
            clipped=tmp_dir / f"sim_{uid}_clipped.stl",
            tapered=tmp_dir / f"sim_{uid}_tapered.stl",
            repaired=tmp_dir / f"sim_{uid}_repaired.stl",
            capped=tmp_dir / f"sim_{uid}_capped.stl",
            encoding=output_dir / f"sim_{uid}_encoding.vtm",
        )


def _cleanup_temp_dir(temp_dir: Path, final: Path, verbose: bool):
    """
    Remove the temporary folder unless it contains the final output.
    """
    temp_dir = temp_dir.resolve()
    final = final.resolve()

    try:
        final.relative_to(temp_dir)
        final_inside_temp = True
    except ValueError:
        final_inside_temp = False

    if final_inside_temp:
        if verbose:
            print(f"[cleanup] Skipping removal of temp dir {temp_dir} because it holds the final STL.")
        return

    if not temp_dir.exists():
        return

    try:
        if verbose:
            print(f"[cleanup] Removing temp dir {temp_dir}")
        shutil.rmtree(temp_dir)
    except OSError as exc:
        if verbose:
            print(f"[cleanup] Failed to remove temp dir {temp_dir}: {exc}")


def _build_encoding_primitives(encoding_path: Path):
    cl_meta, rad_meta = load_encoding(encoding_path)
    return cl_meta, rad_meta


def _apply_transformations(
    cl_meta: dict,
    rad_meta: dict,
    vcs_map_path: Path,
    paths: PipelinePaths,
    *,
    bump_specs: Sequence[BumpSpec],
    size_scale: float,
    straighten_strength: float,
    straighten_exponent: float,
    straighten_preserve: int,
    rho_min: float,
    radius_fit_laplacian: float,
):
    cl, radius, _ = transform_vcs_map(
        cl_meta,
        rad_meta,
        vcs_map_path,
        paths.bumped_vtp,
        paths.bump_png,
        bumps=bump_specs,
        size_scale=size_scale,
        straighten_strength=straighten_strength,
        straighten_exponent=straighten_exponent,
        straighten_preserve=straighten_preserve,
        rho_min=rho_min,
        radius_fit_laplacian=radius_fit_laplacian,
    )
    save_encoding(cl, radius, paths.encoding)


def _extract_rim_and_export_stl(
    bumped_vtp: Path,
    rim_out: Path,
    sim_stl_out: Path,
    *,
    rim_tol: float,
):
    extract_rim(bumped_vtp, rim_out, tol=rim_tol)
    stl_from_vtp(bumped_vtp, sim_stl_out)


def _deform_partner_to_rim(
    partner_stl: Path,
    partner_orig_rim: Path,
    target_rim: Path,
    out_path: Path,
    *,
    deform_r1: float,
    deform_r2: float,
):
    deform_rim_to_target(
        stl_path=str(partner_stl),
        orig_rim_vtp=str(partner_orig_rim),
        target_rim_vtp=str(target_rim),
        out_stl_path=str(out_path),
        r1=deform_r1,
        r2=deform_r2,
    )


def _append_and_clip(
    sim_stl: Path,
    partner_deformed: Path,
    combined_out: Path,
    clipped_out: Path,
    *,
    clip_offset: float,
):
    append_meshes(sim_stl, partner_deformed, combined_out)
    clip_bottom(combined_out, clipped_out, clip_offset=clip_offset)


def _taper_selected_end(
    clipped: Path,
    tapered_out: Path,
    *,
    taper_end: str,
    taper_length_mm: float,
    taper_target_scale: float,
    taper_sections: int,
):
    taper_stl_end(
        input_stl_path=str(clipped),
        output_stl_path=str(tapered_out),
        target_end=taper_end,
        extrusion_length_mm=taper_length_mm,
        target_scale=taper_target_scale,
        n_sections=taper_sections,
    )


def _repair_and_save(
    tapered: Path,
    repaired_out: Path,
    *,
    repair_pitch: float | None,
    repair_target_voxels_min_dim: int,
    repair_closing_radius: int,
    repair_merge_digits: int,
    repair_keep_area_ratio: float,
    repair_plane_range_tol: float | None,
    repair_verbose: bool,
) -> float:
    return repair_surface_four_open_ends(
        tapered,
        repaired_out,
        pitch=repair_pitch,
        target_voxels_min_dim=repair_target_voxels_min_dim,
        closing_radius=repair_closing_radius,
        merge_digits=repair_merge_digits,
        keep_area_ratio=repair_keep_area_ratio,
        plane_range_tol=repair_plane_range_tol,
        verbose=repair_verbose,
    )


def _cap_four_axis_aligned_ends(
    repaired_open: Path,
    capped_out: Path,
    *,
    merge_digits: int,
    plane_range_tol: float | None,
    verbose: bool,
) -> CapReport:
    return cap_four_open_axis_aligned_ends(
        input_stl=repaired_open,
        output_capped_stl=capped_out,
        merge_digits=merge_digits,
        plane_range_tol=plane_range_tol,
        verbose=verbose,
    )


def run_pipeline(
    tau0: float,
    theta0: float,
    bump_amp: float,
    sigma_t: float = 0.05,
    sigma_theta: float = 0.25,
    bumps: Sequence[BumpSpec | dict] | None = None,
    size_scale: float = 1.0,
    straighten_strength: float = 0.0,
    straighten_exponent: float = 2.0,
    straighten_preserve: int = 4,
    rho_min: float = 1e-3,
    radius_fit_laplacian: float = 1e-3,
    rim_tol: float = 1e-3,
    deform_r1: float = 5.0,
    deform_r2: float = 20.0,
    clip_offset: float = 0.5,
    taper_end: str = "auto",
    taper_length_mm: float = 14.0,
    taper_target_scale: float = 0.5,
    taper_sections: int = 24,
    repair_pitch: float | None = None,
    repair_closing_radius: int = 1,
    repair_target_voxels_min_dim: int = 80,
    repair_merge_digits: int = 6,
    repair_keep_area_ratio: float = 0.01,
    repair_plane_range_tol: float | None = None,
    repair_verbose: bool = True,
    output_dir: Path = Path("."),
    temp_dir: Path | None = None,
    keep_temp_files: bool = True,
    *,
    combined_output: Path | None = None,
    tapered_output: Path | None = None,
    encoding_path: Path = Path("pipeline/sim_conduit/Encoding/encoding.vtm"),
    vcs_map_path: Path = Path("pipeline/sim_conduit/Encoding/vcs_map.vtp"),
    partner_stl: Path = Path("pipeline/not_conduit_extruded_canon.stl"),
    partner_orig_rim: Path = Path("pipeline/basic_loop_canon.vtp"),
) -> tuple[Path, str]:
    """
    Execute full pipeline and return path to the capped combined STL and the parameter hash.
    Intermediates are written to a temp directory (default: per-run folder under output_dir) and
    can be deleted after completion via keep_temp_files=False.

    Steps:
    1) Apply VCS transforms (multi-bump, global scale, straightening) to build a new encoding +
       VCS map + preview PNG.
    2) Extract rim at tau≈0 from the transformed VTP.
    3) Convert transformed VTP to STL.
    4) Deform partner STL to match new rim.
    5) Append partner and transformed STLs.
    6) Clip bottom, rebase to Z=0.
    7) Taper a chosen open end on the clipped STL.
    8) Repair merged surface while keeping the 4 outlets open via voxel remeshing.
    9) Cap the 4 axis-aligned outlets and report watertightness of the final STL.

    If combined_output is provided, a copy of the merged STL from step 6 (pre-clip) is written
    to that path. If tapered_output is provided, a copy of the tapered STL from step 7
    (pre-voxelization) is written to that path.

    Tapering options:
    - taper_end chooses which open end to extrude; use "auto" for the largest perimeter or pick
      among YZ_minX/YZ_maxX/XZ_minY/XZ_maxY/XY_minZ/XY_maxZ.
    - taper_length_mm / taper_target_scale / taper_sections control taper length, final scale, and
      the number of interpolation segments.
    - set keep_temp_files=False to delete the per-run temp folder once the final STL is saved.
    """
    bump_specs = normalize_bump_specs(
        bumps,
        tau0=tau0,
        theta0=theta0,
        bump_amp=bump_amp,
        sigma_t=sigma_t,
        sigma_theta=sigma_theta,
    )

    params = dict(
        tau0=tau0,
        theta0=theta0,
        bump_amp=bump_amp,
        sigma_t=sigma_t,
        sigma_theta=sigma_theta,
        bumps=[spec.__dict__ for spec in bump_specs],
        size_scale=size_scale,
        straighten_strength=straighten_strength,
        straighten_exponent=straighten_exponent,
        straighten_preserve=straighten_preserve,
        rho_min=rho_min,
        radius_fit_laplacian=radius_fit_laplacian,
        rim_tol=rim_tol,
        deform_r1=deform_r1,
        deform_r2=deform_r2,
        clip_offset=clip_offset,
        taper_end=taper_end,
        taper_length_mm=taper_length_mm,
        taper_target_scale=taper_target_scale,
        taper_sections=taper_sections,
        repair_pitch=repair_pitch,
        repair_closing_radius=repair_closing_radius,
        repair_target_voxels_min_dim=repair_target_voxels_min_dim,
        repair_merge_digits=repair_merge_digits,
        repair_keep_area_ratio=repair_keep_area_ratio,
        repair_plane_range_tol=repair_plane_range_tol,
    )
    uid = hash_params(params)

    paths = PipelinePaths.build(uid, output_dir, temp_dir)

    # Build encoding primitives
    cl_meta, rad_meta = _build_encoding_primitives(encoding_path)

    # 1) Apply VCS transformations (multi-bump/scale/straighten) and write encoding artifacts
    _apply_transformations(
        cl_meta,
        rad_meta,
        vcs_map_path,
        paths,
        bump_specs=bump_specs,
        size_scale=size_scale,
        straighten_strength=straighten_strength,
        straighten_exponent=straighten_exponent,
        straighten_preserve=straighten_preserve,
        rho_min=rho_min,
        radius_fit_laplacian=radius_fit_laplacian,
    )

    # 2-3) Rim extraction and STL conversion
    _extract_rim_and_export_stl(
        paths.bumped_vtp,
        paths.rim,
        paths.sim_stl,
        rim_tol=rim_tol,
    )

    # 4) Deform partner STL to match rim
    _deform_partner_to_rim(
        partner_stl,
        partner_orig_rim,
        paths.rim,
        paths.partner_deformed,
        deform_r1=deform_r1,
        deform_r2=deform_r2,
    )

    # 5-6) Combine meshes and clip
    _append_and_clip(
        paths.sim_stl,
        paths.partner_deformed,
        paths.combined_tmp,
        paths.clipped,
        clip_offset=clip_offset,
    )
    if combined_output is not None:
        combined_output = Path(combined_output)
        combined_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(paths.combined_tmp, combined_output)
        except shutil.SameFileError:
            pass

    # 7) Taper the selected open end to smooth into the repair step
    _taper_selected_end(
        paths.clipped,
        paths.tapered,
        taper_end=taper_end,
        taper_length_mm=taper_length_mm,
        taper_target_scale=taper_target_scale,
        taper_sections=taper_sections,
    )
    if tapered_output is not None:
        tapered_output = Path(tapered_output)
        tapered_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(paths.tapered, tapered_output)
        except shutil.SameFileError:
            pass

    # 8) Repair via voxel remeshing
    pitch_used = _repair_and_save(
        paths.tapered,
        paths.repaired,
        repair_pitch=repair_pitch,
        repair_target_voxels_min_dim=repair_target_voxels_min_dim,
        repair_closing_radius=repair_closing_radius,
        repair_merge_digits=repair_merge_digits,
        repair_keep_area_ratio=repair_keep_area_ratio,
        repair_plane_range_tol=repair_plane_range_tol,
        repair_verbose=repair_verbose,
    )
    if repair_verbose:
        print(f"[repair] pitch used: {pitch_used}")

    # 9) Cap the four axis-aligned outlets and report watertightness
    cap_report = _cap_four_axis_aligned_ends(
        paths.repaired,
        paths.capped,
        merge_digits=repair_merge_digits,
        plane_range_tol=repair_plane_range_tol,
        verbose=repair_verbose,
    )
    if not repair_verbose:
        print(
            f"[cap] watertight={cap_report.watertight} "
            f"(boundary_loops={cap_report.boundary_loops}, "
            f"boundary_edges={cap_report.boundary_edges}, "
            f"nonmanifold_edges={cap_report.nonmanifold_edges})"
        )

    # Reload with pyvista for consistent STL writing
    pv.read(str(paths.capped)).save(str(paths.final))

    if not keep_temp_files:
        encoding_candidates = {paths.encoding, paths.encoding.with_suffix("")}
        for enc_path in encoding_candidates:
            if not enc_path.exists():
                continue
            try:
                if enc_path.is_dir():
                    shutil.rmtree(enc_path)
                else:
                    enc_path.unlink()
                if repair_verbose:
                    print(f"[cleanup] Removed encoding artifact at {enc_path}")
            except OSError as exc:
                if repair_verbose:
                    print(f"[cleanup] Failed to remove encoding artifact {enc_path}: {exc}")
        _cleanup_temp_dir(paths.temp_dir, paths.final, verbose=repair_verbose)

    return paths.final, uid
