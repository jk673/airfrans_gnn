# Repository Guidelines

## Project Context
AirfRANS GNN is a physics-informed GNN training pipeline for 2D airfoil CFD surrogates (`u`, `v`, `Cp`, `nu_t`) on AirfRANS meshes using PyTorch Geometric and custom RANS-derived losses. The project has migrated from notebooks to script-based workflows; prefer `src/` + `scripts/` changes over notebook edits.

## Project Structure & Module Organization
- `src/`: core library code (physics loss, preprocessing helpers, metrics, trainer, models, utilities).
- `scripts/`: production entry points (`train.py`, `train_multiscale.py`, `eval_cp.py`, `optuna_hpo.py`, `run_experiment.py`, `score_benchmark.py`).
- `preprocessing/`: dataset conversion/downsampling/edge construction.
- `tests/`: pytest targets for loss correctness and batching behavior.
- `docs/` and `docs/benchmark/`: experiment docs, scoring rules, FLOW-GLIDE reference metrics.
- `Dataset/`, `downsampled_graphs/`, `prebuilt_edges_v2/`: local data artifacts.
- `checkpoint/`: model snapshots produced by experiments.

## Build, Test, and Development Commands
- `chmod +x setup_env.sh && ./setup_env.sh`: installs uv + Python 3.11, dependencies, and runs import checks.
- `source .venv/bin/activate`: enable the local environment.
- `uv sync --python 3.11`: install/update dependencies.
- `python preprocessing/downsample_airfrans.py --root Dataset --task scarce --out-dir downsampled_graphs`
- `python preprocessing/build_edges_from_downsampled.py --in-dir downsampled_graphs --out-dir prebuilt_edges_v2 --task scarce`
- `python scripts/train.py --help` / `python scripts/train_multiscale.py --help`: verify training options.
- `python scripts/train.py` / `python scripts/train_multiscale.py`: run training.
- `python scripts/eval_cp.py --checkpoint checkpoints/best.pt --split test` and `python scripts/score_benchmark.py --checkpoint checkpoints/best.pt --task scarce`.
- `pytest tests/ -v`: run full test suite; use `pytest tests/test_continuity_loss.py -k "continuity" -q` for targeted runs.

## Coding Style & Naming Conventions
- Use 4-space indentation and explicit, descriptive names.
- Follow existing module style with clear function boundaries and docstrings for non-trivial logic.
- Naming: `snake_case` for functions/files, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Prefer CLI-first implementation patterns in scripts and keep defaults reproducible.

## Testing Guidelines
- Tests must follow discovery rules: `python_files = test_*.py`, `python_functions = test_*`.
- New physics/math changes should include at least one targeted unit test in `tests/`.
- For refactors, add or update a smoke test and document expected metrics/behavior in PR description.

## Commit & Pull Request Guidelines
- Keep commits concise and imperative (e.g., `fix: normalize edge attr ordering`, `feat: add v2 preprocessing path`).
- PRs should describe objective, changed files, validation commands/results, and any dataset/compute assumptions.

## Agent-Specific Instructions
- For framework/API behavior questions, anchor answers to official docs; when version-specific, note the exact PyTorch/PyG version.
- Prefer deterministic workflows and avoid committing large artifacts or secrets (`.env`, API keys, full datasets, checkpoints).
