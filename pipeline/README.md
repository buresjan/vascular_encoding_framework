## VEF Bump-to-Geometry Pipeline (self-contained)

This subfolder holds the minimal building blocks to:
1) rebuild the sim_conduit centerline/radius,
2) apply VCS transforms (multi-bump, global scale, centerline straightening),
3) extract the rim,
4) deform the partner STL, append, clip, repair a combined STL,
5) cap the four axis-aligned outlets for a watertight final mesh.

### Inputs
- Ground truth (unchanged): `sim_conduit/Encoding/encoding.vtm`, `sim_conduit/Encoding/vcs_map.vtp`
- Transform controls: `bumps` (list of bump dicts) or the single `tau0`/`theta0`/`bump_amp` triplet
  (if `bumps` is empty or omitted, the single-bump values are used when `bump_amp != 0`),
  `size_scale`, `straighten_strength`/`straighten_exponent`/`straighten_preserve`, `rho_min`,
  `radius_fit_laplacian`
- Partner assets: `not_conduit_extruded_canon.stl`, `basic_loop_canon.vtp`
- Parameters: `tau0`, `theta0`, `bump_amp` (plus optional widths, rim tolerance, and deformation/clip
  settings)

### Use from Python
```python
from pathlib import Path
from pipeline import run_pipeline

final_path, uid = run_pipeline(
    tau0=0.0,
    theta0=3.0,
    bump_amp=1.0,
    sigma_t=0.05,
    sigma_theta=0.25,
    bumps=None,             # or a list of bump dicts/BumpSpec to apply sequentially
    size_scale=1.0,         # widen/narrow conduit globally
    straighten_strength=0.0,  # 0=no straightening, 1=fully project to the end-to-end chord
    straighten_exponent=2.0,  # farther control points move more
    straighten_preserve=4,    # keep first/last control points fixed (keeps tau=0 rim normal)
    rho_min=1e-3,
    radius_fit_laplacian=1e-3,
    offset_xy=(0.0, 0.0),  # optional XY translation applied before rim extraction
    rim_tol=1e-3,
    deform_r1=5.0,
    deform_r2=20.0,
    clip_offset=0.5,
    repair_pitch=None,  # auto-pick based on geometry if None
    output_dir=Path("outputs"),
    # optional overrides:
    # temp_dir=Path("outputs/tmp_my_run"),
    # keep_temp_files=False,  # delete intermediates after the final STL is saved
    # encoding_path=Path("pipeline/sim_conduit/Encoding/encoding.vtm"),
    # vcs_map_path=Path("pipeline/sim_conduit/Encoding/vcs_map.vtp"),
    # partner_stl=Path("pipeline/not_conduit_extruded_canon.stl"),
    # partner_orig_rim=Path("pipeline/basic_loop_canon.vtp"),
)
print("Wrote:", final_path, "UID:", uid)
```

### Config-driven usage (vef_pipeline.py)
`pipeline/vef_pipeline.py` is the config-oriented entry point meant for external orchestration.
It loads a JSON config, merges it with defaults from `pipeline/vef_pipeline.py`, resolves paths,
and calls `pipeline.run_pipeline`.

Config path resolution order:
1) CLI argument: `python pipeline/vef_pipeline.py /path/to/config.json`
2) `CONFIG_PATH_OVERRIDE` inside `pipeline/vef_pipeline.py` (for embedding)
3) `VEF_CONFIG_PATH` environment variable
4) default: `vef_config.json` at the repo root

Config schema (JSON):
- Provide run parameters at the top level, or wrap them under `run_kwargs`.
- Optional `base_dir` anchors relative paths (outputs and asset paths).
- Relative paths are resolved against `base_dir` if provided, otherwise against the config
  file's directory.
- Unknown keys raise an error to catch typos early.
  If the config lives outside this repo, set `base_dir` to the repo root so that the default
  asset paths still resolve cleanly.

Minimal config example:
```json
{
  "base_dir": ".",
  "run_kwargs": {
    "tau0": 0.0,
    "theta0": 3.7,
    "bump_amp": 0.0
  }
}
```

Full example config: `vef_config.sample.json`

### Parameter reference (vef_pipeline config)
Defaults shown below are the values used by `pipeline/vef_pipeline.py` when a key is omitted. It
merges its own defaults with `pipeline.run_pipeline` defaults.

#### Bump controls and VCS transforms
- `tau0` (float, default 0.0): Center position of a single Gaussian bump along the VCS tau axis.
  Also used as the default for `bumps[*].tau0` when omitted.
- `theta0` (float, default 3.7): Center position of a single Gaussian bump along the VCS theta
  axis (radians). Also used as the default for `bumps[*].theta0` when omitted.
- `bump_amp` (float, default 0.0): Amplitude of the single bump (positive expands, negative
  shrinks). Used for a single bump when `bumps` is empty, and as the default `amp` for any bump
  entries that omit it.
- `sigma_t` (float, default 0.05): Gaussian width for bumps along tau (normalized tau units).
  Also used as the default `sigma_t` for `bumps` entries.
- `sigma_theta` (float, default 0.35): Gaussian width for bumps along theta (radians). Also used
  as the default `sigma_theta` for `bumps` entries.
- `bumps` (list[dict] | null, default two sample bumps): Optional list of bump specs. If provided
  and non-empty, each entry can include `tau0`, `theta0`, `amp`, `sigma_t`, `sigma_theta`. Missing
  fields fall back to the single-bump defaults above, and zero-amplitude bumps are skipped. If
  `bumps` is omitted, null, or an empty list, the pipeline falls back to the single-bump
  parameters (`tau0`/`theta0`/`bump_amp` and widths); if `bump_amp` is 0, no bumps are applied. To
  force no bumps while keeping a non-zero `bump_amp`, pass a non-empty `bumps` list with explicit
  `amp: 0.0` entries.
- `size_scale` (float, default 0.9): Uniform scale factor on the radius field (rho). Must be > 0.
- `straighten_strength` (float, default 0.15): Strength of centerline straightening in [0, 1].
  0 leaves the centerline unchanged; 1 fully projects it to the end-to-end chord.
- `straighten_exponent` (float, default 2.0): Exponent that weights how aggressively off-axis
  control points move during straightening (higher moves farther points more).
- `straighten_preserve` (int, default 5): Number of control points at each end to keep fixed,
  preserving the rim normal at tau=0.
- `rho_min` (float, default 0.001): Clamp on the minimum rho value to avoid degenerate radii.
- `radius_fit_laplacian` (float, default 0.001): Laplacian penalty when fitting a new radius
  spline after applying bumps (higher => smoother radius field).
- `offset_xy` (list[float] | tuple[float], default [0.0, 0.0]): XY translation applied to the
  transformed VTP before rim extraction (does not change the saved encoding output).

#### Rim extraction and partner deformation
- `rim_tol` (float, default 0.001): Tolerance around tau=0 used to extract the rim; increase if
  no rim points are found.
- `deform_r1` (float, default 5.0): Distance from the rim within which deformation is applied at
  full strength.
- `deform_r2` (float, default 20.0): Distance from the rim beyond which deformation fades to zero
  (smoothly interpolated between `deform_r1` and `deform_r2`).

#### Clip and taper
- `clip_offset` (float, default 1.0): Offset above the minimum Z used for the clip plane before
  rebasing to Z=0.
- `taper_end` (str, default "YZ_minX"): Which open end to taper. Use `"auto"` to pick the largest
  perimeter end, or one of `YZ_minX`, `YZ_maxX`, `XZ_minY`, `XZ_maxY`, `XY_minZ`, `XY_maxZ`.
- `taper_length_mm` (float, default 22.0): Extrusion length for the tapered extension.
- `taper_target_scale` (float, default 0.5): Final scale of the tapered outlet (0 < scale <= 1).
- `taper_sections` (int, default 24): Number of interpolation segments along the taper (>= 2).
- `outlet_plane_targets` (dict[str, float] | null, default null): Optional mapping from outlet
  label to absolute plane value. When set, the pipeline extends/snap-aligns the open ends to the
  target planes before repair. Labels follow `YZ_minX`, `XZ_minY`, `XY_minZ`, `XY_maxZ` (and the
  corresponding `*_max*` variants present in the mesh).

#### Repair and cap
- `repair_pitch` (float | null, default 0.125): Voxel size for remeshing during repair. Set to
  null to auto-compute from `repair_target_voxels_min_dim`.
- `repair_target_voxels_min_dim` (int, default 80): Target voxel count along the smallest mesh
  dimension when auto-picking `repair_pitch`.
- `repair_closing_radius` (int, default 1): Morphological closing radius (voxels) used to fill
  small gaps before marching cubes.
- `repair_merge_digits` (int, default 6): Rounding precision used when merging vertices; smaller
  values merge more aggressively.
- `repair_keep_area_ratio` (float, default 0.01): Keep only connected components with area >=
  max_area * ratio before repair; decrease to keep more fragments.
- `repair_plane_range_tol` (float | null, default null): Planarity tolerance for detecting the
  four open ends. Null derives a scale-aware default from the mesh bounds.
- `repair_verbose` (bool, default true): Print repair and cap diagnostics.

#### Output and paths
- `output_dir` (path, default `outputs`): Directory for final outputs and (by default) temp files.
- `temp_dir` (path | null, default null): Explicit temp directory. If null, a per-run folder under
  `output_dir` is created.
- `keep_temp_files` (bool, default false): If false, delete temp artifacts and encoding outputs
  after the final STL is written.
- `combined_output` (path | null, default null): Optional path to write the combined STL before
  clipping.
- `tapered_output` (path | null, default null): Optional path to write the tapered STL before
  voxel repair.
- `encoding_path` (path, default `pipeline/sim_conduit/Encoding/encoding.vtm`): Canonical VCS
  encoding input.
- `vcs_map_path` (path, default `pipeline/sim_conduit/Encoding/vcs_map.vtp`): Canonical VCS map
  input.
- `partner_stl` (path, default `pipeline/not_conduit_extruded_canon.stl`): Partner STL to deform.
- `partner_orig_rim` (path, default `pipeline/basic_loop_canon.vtp`): Original partner rim used
  to drive deformation.

### What it does
1. Loads saved encoding metadata and rebuilds the centerline and radius splines.
2. Applies multi-bump/scale/straighten transforms (preserving the tau=0 rim normal), then applies
   the optional XY offset, writing a transformed VTP + preview PNG (temp). If `keep_temp_files=True`,
   also saves `sim_<hash>_encoding.vtm` to `output_dir`.
3. Extracts the tau≈0 rim from the transformed map using `rim_tol`.
4. Converts the transformed map to STL.
5. Deforms the partner STL so its rim matches the transformed rim (`r1`/`r2` control falloff).
6. Appends both STLs.
7. Clips from the bottom (offset above min Z), rebases to Z=0, triangulates/cleans.
8. Optionally extends/snap-aligns the open ends to `outlet_plane_targets` (if provided).
9. Repairs via voxel remeshing (caps → voxelize → marching cubes → reopen the 4 outlets) to
   produce a cleaned open STL (kept in the temp directory).
10. Caps the four detected axis-aligned outlet loops (`pipeline/cap.py`) and reports watertightness,
    writing the final capped STL `sim_<hash>.stl` in `output_dir`.

Temporary files now live in a per-run temp folder under `output_dir` (prefix
`sim_<hash>_tmp_...`) and are kept for inspection alongside the final capped STL unless you pass
`keep_temp_files=False` to clean them up after the run.
