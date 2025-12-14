from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to sys.path so package imports work when executed directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import run_pipeline

DEFAULT_RUN_KWARGS: Dict[str, Any] = {
    "tau0": 0.0,
    "theta0": 3.7,
    "bump_amp": 0.0,
    "sigma_t": 0.05,
    "sigma_theta": 0.35,
    "bumps": [
        {"tau0": 0.0, "theta0": 3.7, "amp": 2.3, "sigma_t": 0.05, "sigma_theta": 0.35},
        {"tau0": 0.0, "theta0": 0.7, "amp": -3.0, "sigma_t": 0.05, "sigma_theta": 0.55},
    ],
    "size_scale": 0.9,
    "straighten_strength": 0.15,
    "straighten_exponent": 2.0,
    "straighten_preserve": 5,
    "rho_min": 1e-3,
    "radius_fit_laplacian": 1e-3,
    "rim_tol": 1e-3,
    "deform_r1": 5.0,
    "deform_r2": 20.0,
    "clip_offset": 1.0,
    "taper_end": "YZ_minX",
    "taper_length_mm": 22.0,
    "taper_target_scale": 0.5,
    "taper_sections": 24,
    "repair_pitch": 0.125,
    "output_dir": Path("outputs"),
    "encoding_path": Path("pipeline/sim_conduit/Encoding/encoding.vtm"),
    "vcs_map_path": Path("pipeline/sim_conduit/Encoding/vcs_map.vtp"),
    "partner_stl": Path("pipeline/not_conduit_extruded_canon.stl"),
    "partner_orig_rim": Path("pipeline/basic_loop_canon.vtp"),
    "keep_temp_files": False,
}


def run_with_defaults(**overrides: Any):
    """
    Run the pipeline using the default parameters, allowing selective overrides.
    """
    kwargs = {**DEFAULT_RUN_KWARGS, **overrides}
    return run_pipeline(**kwargs)


def main():
    final_path, uid = run_with_defaults()
    print("Wrote:", final_path, "UID:", uid)


if __name__ == "__main__":
    main()
