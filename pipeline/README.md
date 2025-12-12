## VEF Bump-to-Geometry Pipeline (self-contained)

This subfolder holds the minimal building blocks to:
1) rebuild the sim_conduit centerline/radius,
2) bump the VCS radius at a chosen `(tau0, theta0, bump_amp)`,
3) extract the rim,
4) deform the partner STL, append, and clip a combined STL, and
5) run the `mesh_fix.py` repair -> cap -> voxel-rebuild chain to output a watertight surface.

### Inputs
- Ground truth (unchanged): `sim_conduit/Encoding/encoding.vtm`, `sim_conduit/Encoding/vcs_map.vtp`
- Partner assets: `not_conduit_extruded_canon.stl`, `basic_loop_canon.vtp`
- Parameters: `tau0`, `theta0`, `bump_amp` (plus optional widths, rim tolerance, and deformation/clip settings)

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
    hole_size=1000.0,
    voxel_pitch=1.0,
    output_dir=Path("outputs"),
    # optional overrides:
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
7. Clips from the bottom (offset above min Z), rebases to Z=0, triangulates/cleans.
8. Runs the `mesh_fix.py` sequence: repair (keep 4 outlets open), cap those outlets, then rebuild a
   watertight exterior via voxelization (`voxel_pitch` tunable). The final `sim_<hash>.stl` in
   `output_dir` is watertight.

Temporary files live in a temp dir and are deleted when done; only your inputs and the watertight
STL remain.

### Dash playground (interactive)
There is a small Dash app that exposes the same parameters as `vef_pipeline.py` and previews the
resulting STL:
```bash
python -m pipeline.dash_app
```
- Edit the numeric fields and click **Run pipeline** to kick off a run.
- The latest STL is rendered directly in the browser and saved to `outputs/` (or a custom folder).
- Use **Download STL** to fetch the file produced by the most recent run.
