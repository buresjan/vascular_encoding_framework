## VEF Bump-to-Geometry Pipeline

This pipeline glues together the modular pieces already present in the repo to take a bump
specification in VCS space and produce a clipped STL that combines the bumped sim_conduit geometry
with the deformed partner geometry, then caps the four axis-aligned outlets for a watertight result.

### Inputs
- `tau0`, `theta0`, `bump_amp` (and widths `sigma_t`, `sigma_theta`) or an explicit `bumps` list
  for multi-bump runs
- Global transforms: `size_scale` (rho scaling), `straighten_strength`/`straighten_exponent`/
  `straighten_preserve` (centerline straightening while keeping the tau=0 rim normal)
- Radius controls: `rho_min` clamp to avoid degenerate radii, `radius_fit_laplacian` smoothing when
  fitting a spline to the bumped rho field
- Existing ground-truth data: `sim_conduit/Encoding/encoding.vtm`, `sim_conduit/Encoding/vcs_map.vtp`
- Partner assets: `not_conduit_extruded_canon.stl`, `basic_loop_canon.vtp`
- Rim extraction tolerance: `rim_tol`
- Voxel remesh pitch for the repair step: `repair_pitch` (None = auto from size)

### Steps
1. **Build encoding primitives**: load the saved spline metadata and reconstruct the canonical
   centerline and radius surfaces.
2. **Apply VCS transforms**: optional multi-bumps, global rho scale, and centerline straightening
   (tau=0 rim normal preserved) -> transformed VTP, preview PNG, and (if `keep_temp_files=True`) a
   saved encoding `sim_<hash>_encoding.vtm` in `output_dir`.
3. **Extract rim**: pull the tau≈0 rim from the transformed VTP (`rim.extract_rim`) using `rim_tol`.
4. **STL export**: convert the transformed VTP to STL for downstream use.
5. **Deform partner**: deform `not_conduit_extruded_canon.stl` so its rim matches the new rim
   (`deform.deform_rim_to_target`, with tunable `r1`, `r2` falloff).
6. **Append**: merge the transformed STL and deformed partner STL into a combined geometry.
7. **Clip & align**: cut the bottom with a plane parallel to XY (offset above min Z), rebase to
   Z=0, triangulate, and clean.
8. **Taper an outlet**: run `taper_stl_end` on the clipped STL to extrude and shrink a chosen open
   end (e.g., `XZ_minY`, `YZ_maxX`, `XY_minZ`), controlling length, number of segments, and scale.
9. **Align outlets (optional)**: extend/snap-align open ends to `outlet_plane_targets` if provided.
10. **Repair via voxel remesh**: run `pipeline/repair.py` to cap → voxelize → marching cubes and
    reopen the 4 outlets, producing a cleaned open STL in the temp folder.
11. **Cap outlets**: run `pipeline/cap.py` to fan-cap the four detected axis-aligned loops and
    report watertightness, writing the final capped STL `sim_<hash>.stl` in `output_dir`.

Temporary artifacts live in a per-run temp directory under `output_dir` (prefix `sim_<hash>_tmp_`)
and are kept alongside the final capped STL unless `keep_temp_files=False` is passed to clean them
up after the run completes.

### Usage (from Python)
```python
from pathlib import Path
from vef_pipeline import run_pipeline

final_path, uid = run_pipeline(
    tau0=0.0,
    theta0=3.0,
    bump_amp=1.0,
    sigma_t=0.05,
    sigma_theta=0.25,
    # Multi-bump example (optional):
    bumps=[
        {"tau0": 0.12, "theta0": 1.8, "amp": 0.6, "sigma_t": 0.05, "sigma_theta": 0.3},
        {"tau0": 0.48, "theta0": 5.2, "amp": -0.45, "sigma_t": 0.06, "sigma_theta": 0.25},
    ],
    size_scale=1.0,           # widen/narrow conduit globally (rho scale)
    straighten_strength=0.0,  # 0=no straightening, 1=project fully to the end-to-end chord
    straighten_exponent=2.0,  # moves the most off-axis control points more than near-axis ones
    straighten_preserve=4,    # keep first/last control points fixed (preserves rim normal)
    rho_min=1e-3,             # clamp rho to avoid degenerate geometry
    radius_fit_laplacian=1e-3,
    rim_tol=1e-3,
    deform_r1=5.0,
    deform_r2=20.0,
    clip_offset=0.5,
    taper_end="YZ_minX",  # or autoYZ_minX/YZ_maxX/XZ_minY/XZ_maxY/XY_minZ/XY_maxZ
    taper_length_mm=14.0,
    taper_target_scale=0.5,
    taper_sections=24,
    repair_pitch=None,  # auto-pick based on geometry if None
    output_dir=Path("outputs"),
    # keep_temp_files=False,  # drop intermediates once the final STL is written
)
print("wrote", final_path, "uid", uid)
```
