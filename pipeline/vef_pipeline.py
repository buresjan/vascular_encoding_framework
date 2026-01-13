from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

# Add project root to sys.path so package imports work when executed directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import run_pipeline

CONFIG_ENV_VAR = "VEF_CONFIG_PATH"
DEFAULT_CONFIG_PATH = ROOT / "vef_config.json"
# Set this when embedding to control how the config is sourced.
CONFIG_PATH_OVERRIDE: Path | None = None

RUN_KWARG_KEYS = set(inspect.signature(run_pipeline).parameters)
PATH_KEYS = {
    "output_dir",
    "temp_dir",
    "combined_output",
    "tapered_output",
    "encoding_path",
    "vcs_map_path",
    "partner_stl",
    "partner_orig_rim",
}

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
    "offset_xy": (0.0, 0.0),
    "rim_tol": 1e-3,
    "deform_r1": 5.0,
    "deform_r2": 20.0,
    "clip_offset": 1.0,
    "taper_end": "YZ_minX",
    "taper_length_mm": 22.0,
    "taper_target_scale": 0.5,
    "taper_sections": 24,
    "outlet_plane_targets": None,
    "repair_pitch": 0.125,
    "output_dir": Path("outputs"),
    "encoding_path": ROOT / "pipeline" / "sim_conduit" / "Encoding" / "encoding.vtm",
    "vcs_map_path": ROOT / "pipeline" / "sim_conduit" / "Encoding" / "vcs_map.vtp",
    "partner_stl": ROOT / "pipeline" / "not_conduit_extruded_canon.stl",
    "partner_orig_rim": ROOT / "pipeline" / "basic_loop_canon.vtp",
    "keep_temp_files": False,
}


def resolve_config_path(cli_arg: str | None = None) -> Path:
    """
    Resolve where the configuration should be loaded from.
    """
    if cli_arg:
        return Path(cli_arg).expanduser()
    if CONFIG_PATH_OVERRIDE is not None:
        return Path(CONFIG_PATH_OVERRIDE).expanduser()
    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_CONFIG_PATH


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """
    Load a JSON config from disk.
    """
    config_path = config_path.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object at the top level.")
    return data


def _resolve_base_dir(base_dir: str | Path | None, config_path: Path | None) -> Path | None:
    if base_dir is None:
        if config_path is None:
            return None
        return config_path.parent.resolve()
    base_dir = Path(base_dir).expanduser()
    if not base_dir.is_absolute():
        anchor = config_path.parent if config_path is not None else Path.cwd()
        base_dir = anchor / base_dir
    return base_dir.resolve()


def _coerce_path(value: str | Path, base_dir: Path | None) -> Path:
    path_value = value if isinstance(value, Path) else Path(value)
    path_value = path_value.expanduser()
    if base_dir is not None and not path_value.is_absolute():
        path_value = base_dir / path_value
    return path_value


def _validate_run_kwargs(run_kwargs: Mapping[str, Any]) -> None:
    extra_keys = set(run_kwargs) - RUN_KWARG_KEYS
    if extra_keys:
        extras = ", ".join(sorted(extra_keys))
        raise ValueError(f"Unknown pipeline config keys: {extras}")


def _split_config(config: Mapping[str, Any]) -> tuple[Dict[str, Any], str | Path | None]:
    if "run_kwargs" in config:
        run_kwargs = config.get("run_kwargs") or {}
        if not isinstance(run_kwargs, Mapping):
            raise ValueError("Config 'run_kwargs' must be a mapping.")
        _validate_run_kwargs(run_kwargs)
        base_dir_value = config.get("base_dir")
        return dict(run_kwargs), base_dir_value
    run_kwargs = dict(config)
    base_dir_value = run_kwargs.pop("base_dir", None)
    _validate_run_kwargs(run_kwargs)
    return run_kwargs, base_dir_value


def _apply_path_overrides(
    kwargs: MutableMapping[str, Any],
    base_dir: Path | None,
    keys_to_resolve: set[str],
) -> None:
    for key in PATH_KEYS & keys_to_resolve:
        value = kwargs.get(key)
        if value is None:
            continue
        kwargs[key] = _coerce_path(value, base_dir)


def build_run_kwargs(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
    config_path: Path | None = None,
) -> Dict[str, Any]:
    """
    Normalize config into kwargs for run_pipeline, resolving relative paths as needed.
    """
    run_kwargs, config_base_dir = _split_config(config)
    resolved_base_dir = _resolve_base_dir(base_dir or config_base_dir, config_path)
    kwargs = {**DEFAULT_RUN_KWARGS, **run_kwargs}
    _apply_path_overrides(kwargs, resolved_base_dir, set(run_kwargs))
    return kwargs


def run_from_config(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
):
    """
    Run the pipeline from a config mapping plus optional overrides.
    """
    run_kwargs, config_base_dir = _split_config(config)
    resolved_base_dir = _resolve_base_dir(base_dir or config_base_dir, config_path)
    kwargs = {**DEFAULT_RUN_KWARGS, **run_kwargs}
    _apply_path_overrides(kwargs, resolved_base_dir, set(run_kwargs))
    if overrides:
        override_kwargs = dict(overrides)
        _apply_path_overrides(override_kwargs, resolved_base_dir, set(override_kwargs))
        kwargs.update(override_kwargs)
    return run_pipeline(**kwargs)


def run_with_defaults(**overrides: Any):
    """
    Run the pipeline using the default parameters, allowing selective overrides.
    """
    kwargs = {**DEFAULT_RUN_KWARGS, **overrides}
    _apply_path_overrides(kwargs, None, set(overrides))
    return run_pipeline(**kwargs)


def main(config_path: str | Path | None = None):
    resolved_path = resolve_config_path(str(config_path) if config_path else None)
    config = load_config_file(resolved_path)
    final_path, uid = run_from_config(config, config_path=resolved_path)
    print("Wrote:", final_path, "UID:", uid)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
