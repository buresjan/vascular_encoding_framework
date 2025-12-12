#!/usr/bin/env python
"""
Run the VEF pipeline (which now includes mesh_fix repair/cap/voxel rebuild), then optionally write
an additional trimesh-based repair copy for comparison/debugging.

Outputs:
- Final watertight STL from the pipeline.
- Extra repaired STL (trimesh cleanup) if you want to experiment further.
"""

from pathlib import Path

from pipeline import run_pipeline
from pipeline.mesh_ops import repair_stl


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Run the baseline pipeline (already watertight)
    final_path, uid = run_pipeline(
        tau0=0.0,
        theta0=3.0,
        bump_amp=1.0,
        sigma_t=0.05,
        sigma_theta=0.25,
        deform_r1=5.0,
        deform_r2=20.0,
        clip_offset=1.0,
        hole_size=1000.0,
        output_dir=output_dir,
    )

    # Optional: apply a secondary STL repair pass for experimentation
    repaired_path = output_dir / f"sim_{uid}_repaired.stl"
    repair_log = repair_stl(final_path, repaired_path)

    print(f"Pipeline output (watertight): {final_path}")
    print(f"Secondary repair copy:        {repaired_path}")
    print(f"Repair log:                   {repair_log}")


if __name__ == "__main__":
    main()
