from __future__ import annotations

"""
Generate final STL outputs for all bump/scale/straighten combinations using vef_pipeline defaults.
"""

import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from pipeline import run_pipeline
from pipeline.mesh_ops import hash_params
from pipeline.vef_pipeline import DEFAULT_RUN_KWARGS

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "geometry_combinations"
INDEX_PATH = OUTPUT_DIR / "index.json"
FAILURES_PATH = OUTPUT_DIR / "failures.json"

BUMP1_THETA0 = (2.9, 3.7, 4.5)
BUMP1_AMP = (0.0, 1.5, 2.3)
BUMP2_THETA0 = (0.7, 1.4)
BUMP2_AMP = (-3.0, -1.5, 0.0, 2.0)
SIZE_SCALES = (0.9, 1.0, 1.1)
STRAIGHTEN_STRENGTHS = (0.0, 0.09, 0.15)

BUMP1_SIGMA_T = 0.05
BUMP1_SIGMA_THETA = 0.35
BUMP2_SIGMA_T = 0.05
BUMP2_SIGMA_THETA = 0.55

TOTAL_COMBINATIONS = (
    len(BUMP1_THETA0)
    * len(BUMP1_AMP)
    * len(BUMP2_THETA0)
    * len(BUMP2_AMP)
    * len(SIZE_SCALES)
    * len(STRAIGHTEN_STRENGTHS)
)


def load_index() -> Dict[str, Any]:
    if INDEX_PATH.exists():
        with INDEX_PATH.open() as f:
            return json.load(f)
    return {}


def save_index(index: Dict[str, Any]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def load_failures() -> Dict[str, Any]:
    if FAILURES_PATH.exists():
        with FAILURES_PATH.open() as f:
            return json.load(f)
    return {}


def save_failures(failures: Dict[str, Any]) -> None:
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FAILURES_PATH.open("w") as f:
        json.dump(failures, f, indent=2, sort_keys=True)


def bump_pair(
    theta1: float, amp1: float, theta2: float, amp2: float
) -> Tuple[Dict[str, float], Dict[str, float]]:
    bump1 = {
        "name": "bump1",
        "tau0": 0.0,
        "theta0": float(theta1),
        "amp": float(amp1),
        "sigma_t": BUMP1_SIGMA_T,
        "sigma_theta": BUMP1_SIGMA_THETA,
    }
    bump2 = {
        "name": "bump2",
        "tau0": 0.0,
        "theta0": float(theta2),
        "amp": float(amp2),
        "sigma_t": BUMP2_SIGMA_T,
        "sigma_theta": BUMP2_SIGMA_THETA,
    }
    return bump1, bump2


def combo_uid(combo: Dict[str, Any]) -> str:
    # Length 12 keeps names short while avoiding collisions across 648 combos.
    return hash_params(combo, length=12)


def iter_combinations() -> Iterable[Tuple[str, Dict[str, Any]]]:
    for theta1, amp1, theta2, amp2, size_scale, straighten_strength in product(
        BUMP1_THETA0, BUMP1_AMP, BUMP2_THETA0, BUMP2_AMP, SIZE_SCALES, STRAIGHTEN_STRENGTHS
    ):
        bump1, bump2 = bump_pair(theta1, amp1, theta2, amp2)
        combo = {
            "bump1": bump1,
            "bump2": bump2,
            "size_scale": float(size_scale),
            "straighten_strength": float(straighten_strength),
        }
        yield combo_uid(combo), combo


def strip_labels(bump: Dict[str, Any]) -> Dict[str, float]:
    return {k: v for k, v in bump.items() if k != "name"}


def run_combo(
    uid: str,
    combo: Dict[str, Any],
    base_kwargs: Dict[str, Any],
    index: Dict[str, Any],
    failures: Dict[str, Any],
) -> None:
    target_path = OUTPUT_DIR / f"sim_{uid}.stl"
    existing_entry = index.get(uid)

    if existing_entry and target_path.exists():
        print(f"[skip] {uid} -> {target_path.name}")
        if uid in failures:
            failures.pop(uid, None)
            save_failures(failures)
        return

    run_kwargs = dict(base_kwargs)
    run_kwargs.update(
        bumps=[strip_labels(combo["bump1"]), strip_labels(combo["bump2"])],
        size_scale=combo["size_scale"],
        straighten_strength=combo["straighten_strength"],
    )

    print(f"[run] {uid} -> {target_path.name}")
    try:
        final_path, pipeline_uid = run_pipeline(**run_kwargs)
    except Exception as exc:  # noqa: BLE001 - we want to keep going through combos
        failures[uid] = {
            "combo": combo,
            "error": str(exc),
        }
        save_failures(failures)
        print(f"[error] {uid}: {exc} | combo={combo}")
        return

    final_path = Path(final_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if final_path != target_path:
        final_path.replace(target_path)

    if uid in failures:
        failures.pop(uid, None)
        save_failures(failures)

    index[uid] = {
        "file": target_path.name,
        "pipeline_uid": pipeline_uid,
        "bump1": combo["bump1"],
        "bump2": combo["bump2"],
        "size_scale": combo["size_scale"],
        "straighten_strength": combo["straighten_strength"],
    }
    save_index(index)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()
    failures = load_failures()

    base_kwargs: Dict[str, Any] = dict(DEFAULT_RUN_KWARGS)
    base_kwargs.update(
        output_dir=OUTPUT_DIR,
        keep_temp_files=False,
        repair_verbose=False,
    )

    print(f"Generating up to {TOTAL_COMBINATIONS} combinations into {OUTPUT_DIR}")
    for uid, combo in iter_combinations():
        run_combo(uid, combo, base_kwargs, index, failures)

    save_failures(failures)
    print(f"Finished. Successes: {len(index)}; failures logged: {len(failures)}")


if __name__ == "__main__":
    main()
