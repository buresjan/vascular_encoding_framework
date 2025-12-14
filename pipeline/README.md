## VEF Bump-to-Geometry Pipeline (self-contained)

This subfolder holds the minimal building blocks to:
1) rebuild the sim_conduit centerline/radius,
2) apply VCS transforms (multi-bump, global scale, centerline straightening),
3) extract the rim,
4) deform the partner STL, append, clip, repair a combined STL,
5) cap the four axis-aligned outlets for a watertight final mesh.

### Inputs
- Ground truth (unchanged): `sim_conduit/Encoding/encoding.vtm`, `sim_conduit/Encoding/vcs_map.vtp`
- Transform controls: `bumps` (list of bump dicts) or the single `tau0`/`theta0`/`bump_amp` triplet,
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

### What it does
1. Loads saved encoding metadata and rebuilds the centerline and radius splines.
2. Applies multi-bump/scale/straighten transforms (preserving the tau=0 rim normal), writes a
   transformed VTP + preview PNG (temp). If `keep_temp_files=True`, also saves
   `sim_<hash>_encoding.vtm` to `output_dir`.
3. Extracts the tau≈0 rim from the transformed map using `rim_tol`.
4. Converts the transformed map to STL.
5. Deforms the partner STL so its rim matches the transformed rim (`r1`/`r2` control falloff).
6. Appends both STLs.
7. Clips from the bottom (offset above min Z), rebases to Z=0, triangulates/cleans.
8. Repairs via voxel remeshing (caps → voxelize → marching cubes → reopen the 4 outlets) to
   produce a cleaned open STL (kept in the temp directory).
9. Caps the four detected axis-aligned outlet loops (`pipeline/cap.py`) and reports watertightness,
   writing the final capped STL `sim_<hash>.stl` in `output_dir`.

Temporary files now live in a per-run temp folder under `output_dir` (prefix
`sim_<hash>_tmp_...`) and are kept for inspection alongside the final capped STL unless you pass
`keep_temp_files=False` to clean them up after the run.
