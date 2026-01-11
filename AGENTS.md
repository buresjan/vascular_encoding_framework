# Agent Guide

## Repo overview
- Python package lives in `vascular_encoding_framework/`.
- Pipeline utilities and the config-driven entry point live in `pipeline/`.
- Optional CLI packaging lives in `vef_scripts/`.

## Environment
- Recommended: `conda env create -f environment.yml` (Python 3.10) or `pip install -e .`.
- Runtime depends on `pyvista` and `vtk`; avoid running the pipeline without a full env.

## Pipeline usage
- Config runner: `python pipeline/vef_pipeline.py path/to/config.json`.
- Config resolution order: CLI arg > `CONFIG_PATH_OVERRIDE` in `pipeline/vef_pipeline.py`
  > `VEF_CONFIG_PATH` env var > `vef_config.json` at repo root.
- Schema docs: `pipeline/README.md`. Sample config: `vef_config.sample.json`.

## Outputs and assets
- Pipeline expects assets under `pipeline/` (encoding.vtm, vcs_map.vtp, STL, VTP).
- Outputs default to `outputs/`; avoid overwriting sample outputs unless requested.
- Do not modify large/binary assets unless the task explicitly requires it.

## Style and linting
- Python 3.9+.
- Ruff config in `pyproject.toml`; run `ruff check` or `ruff format` when touching Python code.

## Tests
- No formal test suite. Ad-hoc script: `scripts/size_scale_test.py`.
- If you run scripts/tests, report results.

## Documentation hygiene
- If you add or change pipeline parameters, update `pipeline/README.md` and
  `vef_config.sample.json` together.
