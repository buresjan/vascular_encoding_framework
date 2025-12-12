## VEF Bump-to-Geometry Pipeline

This pipeline glues together the modular pieces already present in the repo to take a bump
specification in VCS space and produce a clipped + watertight STL that combines the bumped
sim_conduit geometry with the deformed partner geometry.

### Inputs
- `tau0`, `theta0`, `bump_amp` (and widths `sigma_t`, `sigma_theta`)
- `voxel_pitch` for the watertight voxel rebuild
- Existing ground-truth data: `sim_conduit/Encoding/encoding.vtm`, `sim_conduit/Encoding/vcs_map.vtp`
- Partner assets: `not_conduit_extruded_canon.stl`, `basic_loop_canon.vtp`
- Rim extraction tolerance: `rim_tol`

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
7. **Clip & align**: cut the bottom with a plane parallel to XY (offset above min Z), rebase to
   Z=0, triangulate, and clean.
8. **Repair/cap/voxel-rebuild**: run the `mesh_fix.py` pipeline to patch small cracks (keeping the
   4 primary outlets open), cap those outlets, and rebuild a watertight exterior using marching
   cubes from a voxelization with pitch `voxel_pitch`. The watertight STL is saved as
   `sim_<hash>.stl` to `output_dir`.

Temporary artifacts live in a temp directory and are removed after the run; the output directory
only receives the watertight combined STL.

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
    hole_size=1000.0,
    voxel_pitch=1.0,
    output_dir=Path("outputs"),
)
print("wrote", final_path, "uid", uid)
```
