## VEF Bump-to-Geometry Pipeline

This pipeline glues together the modular pieces already present in the repo to take a bump
specification in VCS space and produce a clipped STL that combines the bumped sim_conduit geometry
with the deformed partner geometry.

### Inputs
- `tau0`, `theta0`, `bump_amp` (and widths `sigma_t`, `sigma_theta`)
- Existing ground-truth data: `sim_conduit/Encoding/encoding.vtm`, `sim_conduit/Encoding/vcs_map.vtp`
- Partner assets: `not_conduit_extruded_canon.stl`, `basic_loop_canon.vtp`
- Rim extraction tolerance: `rim_tol`
- Optional outlet taper: `taper_enabled`, `taper_end`, `taper_length`, `taper_scale`,
  `taper_segments`, `taper_tol_ratio`
- Voxel remesh pitch for the repair step: `repair_pitch` (None = auto from size)

### Steps
1. **Build encoding primitives**: load the saved spline metadata and reconstruct the centerline and
   radius surfaces.
2. **Bump radius spline**: evaluate a smooth Gaussian bump in rho at (`tau0`, `theta0`) to create a
   bumped VTP plus a preview PNG (both written to a temp dir).
3. **Extract rim**: pull the tau≈0 rim from the bumped VTP (`rim.extract_rim`) using `rim_tol`.
4. **STL export**: convert the bumped VTP to STL for downstream use.
5. **Deform partner**: deform `not_conduit_extruded_canon.stl` so its rim matches the bumped rim
   (`deform.deform_rim_to_target`, with tunable `r1`, `r2` falloff).
6. **Append**: merge the bumped STL and deformed partner STL into a combined geometry.
7. **(Optional) Taper one outlet**: extrude + taper a chosen axis-aligned rim by a set length and
   scale (default: +14 mm, 75% radius).
8. **Clip & align**: cut the bottom with a plane parallel to XY (offset above min Z), rebase to
   Z=0, triangulate, and clean.
9. **Repair via voxel remesh**: run `pipeline/repair.py` to cap → voxelize → marching cubes and
   reopen the 4 outlets, producing the final open STL `sim_<hash>.stl` in `output_dir`.

Temporary artifacts live in a per-run temp directory under `output_dir` (prefix `sim_<hash>_tmp_`)
and are kept alongside the final clipped STL.

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
    rim_tol=1e-3,
    deform_r1=5.0,
    deform_r2=20.0,
    clip_offset=0.5,
    # Optional outlet taper:
    taper_enabled=True,
    taper_end="xz_min_y",  # yz_min_x | yz_max_x | xz_min_y | xz_max_y | xy_min_z | xy_max_z
    taper_length=14.0,
    taper_scale=0.75,
    taper_segments=12,
    taper_tol_ratio=0.02,
    repair_pitch=None,  # auto-pick based on geometry if None
    output_dir=Path("outputs"),
)
print("wrote", final_path, "uid", uid)
```
