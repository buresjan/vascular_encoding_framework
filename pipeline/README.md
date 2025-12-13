## VEF Bump-to-Geometry Pipeline (self-contained)

This subfolder holds the minimal building blocks to:
1) rebuild the sim_conduit centerline/radius,
2) bump the VCS radius at a chosen `(tau0, theta0, bump_amp)`,
3) extract the rim,
4) deform the partner STL, append, optionally taper one outlet, clip, and repair a combined STL.

### Inputs
- Ground truth (unchanged): `sim_conduit/Encoding/encoding.vtm`, `sim_conduit/Encoding/vcs_map.vtp`
- Partner assets: `not_conduit_extruded_canon.stl`, `basic_loop_canon.vtp`
- Parameters: `tau0`, `theta0`, `bump_amp` (plus optional widths, rim tolerance, deformation/clip
  settings, and optional outlet taper controls)

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
    rim_tol=1e-3,
    deform_r1=5.0,
    deform_r2=20.0,
    clip_offset=0.5,
    # Optional outlet tapering:
    taper_enabled=True,
    taper_end="xz_min_y",  # yz_min_x | yz_max_x | xz_min_y | xz_max_y | xy_min_z | xy_max_z
    taper_length=14.0,
    taper_scale=0.75,
    taper_segments=12,
    taper_tol_ratio=0.02,
    repair_pitch=None,  # auto-pick based on geometry if None
    output_dir=Path("outputs"),
    # optional overrides:
    # temp_dir=Path("outputs/tmp_my_run"),
    # encoding_path=Path("pipeline/sim_conduit/Encoding/encoding.vtm"),
    # vcs_map_path=Path("pipeline/sim_conduit/Encoding/vcs_map.vtp"),
    # partner_stl=Path("pipeline/not_conduit_extruded_canon.stl"),
    # partner_orig_rim=Path("pipeline/basic_loop_canon.vtp"),
)
print("Wrote:", final_path, "UID:", uid)
```

### What it does
1. Loads saved encoding metadata and rebuilds the centerline and radius splines.
2. Applies a Gaussian bump in rho at (`tau0`, `theta0`) and writes a bumped VTP plus preview PNG
   (temp).
3. Extracts the tau≈0 rim from the bumped map using `rim_tol`.
4. Converts the bumped map to STL.
5. Deforms the partner STL so its rim matches the bumped rim (`r1`/`r2` control falloff).
6. Appends both STLs.
7. Optionally extrudes one of the 4 axis-aligned outlets and smoothly tapers it (default: +14 mm,
   75% radius at the tip).
8. Clips from the bottom (offset above min Z), rebases to Z=0, triangulates/cleans.
9. Repairs via voxel remeshing (caps → voxelize → marching cubes → reopen the 4 outlets) to
   produce the final open STL `sim_<hash>.stl` in `output_dir`.

Temporary files now live in a per-run temp folder under `output_dir` (prefix
`sim_<hash>_tmp_...`) and are kept for inspection alongside the final STL.
