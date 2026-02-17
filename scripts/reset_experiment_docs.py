"""Utility script to reset experiment tracking and Optuna documentation files.

Usage:
  python scripts/reset_experiment_docs.py [--no-backup] [--skip-experiment-log] [--skip-optuna-doc] [--skip-results-json]

The script rewrites two files with a clean, reproducible template:
- experiments/EXPERIMENT_LOG.md
- docs/optuna/EXAMPLES_OPTUNA.md
그리고 `experiments/results` 아래의 모든 `*.json` 파일을 삭제합니다.

By default, previous versions are backed up with a UTC timestamp suffix.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_LOG = ROOT / "experiments" / "EXPERIMENT_LOG.md"
DEFAULT_EXPERIMENT_DOC = ROOT / "docs" / "optuna" / "EXAMPLES_OPTUNA.md"
DEFAULT_EXPERIMENT_RESULTS_DIR = ROOT / "experiments" / "results"


def _timestamp() -> str:
    """Return an ISO-like UTC timestamp string for generated headers."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _backup_if_exists(path: Path) -> Path | None:
    if not path.exists():
        return None

    backup = path.with_suffix(path.suffix + ".bak.")
    backup = Path(f"{backup}{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    path.replace(backup)
    return backup


def _experiment_log_template(now: str) -> str:
    return (
        "# AirfRANS 2D Airfoil - GNN Surrogate\n\n"
        f"Auto-generated benchmark comparison. Updated: {now}\n\n"
        "No experiments have been logged yet. Run training with benchmark scoring\n"
        "to append entries to this file automatically.\n\n"
        "---\n\n"
        "## Experiment Details\n\n"
    )


def _optuna_examples_template() -> str:
    return (
        "# Optuna HPO Examples\n\n"
        "Practical examples for running hyperparameter optimization with `optuna_hpo.py`.\n\n"

        "## Example 1: Quick Test Run (5 trials)\n\n"
        "```bash\n"
        "python scripts/optuna_hpo.py \\\n"
        "    --n-trials 5 \\\n"
        "    --trial-epochs 5 \\\n"
        "    --limit-train 50 \\\n"
        "    --limit-val 5\n"
        "```\n\n"

        "## Example 2: Standard HPO Run\n\n"
        "```bash\n"
        "python scripts/optuna_hpo.py \\\n"
        "    --study-name standard-hpo \\\n"
        "    --storage sqlite:///optuna_standard.db \\\n"
        "    --n-trials 50 \\\n"
        "    --trial-epochs 20 \\n"
        "    --limit-train 180 \\\n"
        "    --limit-val 20\n"
        "```\n\n"

        "## Notes\n\n"
        "- Start with small settings first (5-10 trials).\n"
        "- Keep `--storage` for long optimization runs.\n"
        "- Set `--visualize-only` to regenerate HTML outputs from finished studies.\n"
    )


def _write(path: Path, content: str, do_backup: bool, dry_run: bool) -> Path | str | None:
    if dry_run:
        return "(dry-run)"

    if do_backup:
        backup = _backup_if_exists(path)
    else:
        backup = None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return backup


def _delete_results_jsons(results_dir: Path, dry_run: bool) -> list[Path]:
    if not results_dir.exists():
        return []

    removed: list[Path] = []
    for path in sorted(results_dir.rglob("*.json")):
        if dry_run:
            removed.append(path)
            continue
        path.unlink()
        removed.append(path)
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset experiment log and Optuna docs to templates.")
    parser.add_argument(
        "--experiment-log-path",
        type=Path,
        default=DEFAULT_EXPERIMENT_LOG,
        help="Target path for EXPERIMENT_LOG.md",
    )
    parser.add_argument(
        "--optuna-doc-path",
        type=Path,
        default=DEFAULT_EXPERIMENT_DOC,
        help="Target path for EXAMPLES_OPTUNA.md",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup existing files before overwriting",
    )
    parser.add_argument(
        "--skip-experiment-log",
        action="store_true",
        help="Skip resetting experiments/EXPERIMENT_LOG.md",
    )
    parser.add_argument(
        "--skip-optuna-doc",
        action="store_true",
        help="Skip resetting docs/optuna/EXAMPLES_OPTUNA.md",
    )
    parser.add_argument(
        "--skip-results-json",
        action="store_true",
        help="Skip deleting experiments/results/*.json",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_RESULTS_DIR,
        help="Directory containing experiment result JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without modifying files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backup_enabled = not args.no_backup
    now = _timestamp()

    if not args.skip_experiment_log:
        backup = _write(
            args.experiment_log_path,
            _experiment_log_template(now),
            do_backup=backup_enabled,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(f"[dry-run] would reset {args.experiment_log_path}")
        else:
            print(f"reset: {args.experiment_log_path}")
            if backup:
                print(f" backup: {backup}")

    if not args.skip_optuna_doc:
        backup = _write(
            args.optuna_doc_path,
            _optuna_examples_template(),
            do_backup=backup_enabled,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(f"[dry-run] would reset {args.optuna_doc_path}")
        else:
            print(f"reset: {args.optuna_doc_path}")
            if backup:
                print(f" backup: {backup}")

    if not args.skip_results_json:
        removed_json = _delete_results_jsons(args.results_dir, dry_run=args.dry_run)
        if args.dry_run:
            print("[dry-run] would delete experiment result JSON files:")
            for path in removed_json:
                print(f" - {path}")
        else:
            print(f"deleted: {len(removed_json)} json file(s) under {args.results_dir}")


if __name__ == "__main__":
    main()
