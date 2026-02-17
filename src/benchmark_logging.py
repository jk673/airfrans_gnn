"""Benchmark scoring and experiment logging after training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

from src.training_common import DataBundle, NormalizedDataset, collate_pyg
from src.utils import _prep_graph_for_norm
from src.experiment_table import (
    FLOW_GLIDE_TABLE_PATH,
    _update_flow_glide_comparison_table,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_benchmark_and_log_experiment(
    model,
    data_bundle: DataBundle,
    scfg,
    device,
    train_summary: dict,
    training_duration_sec: float,
    notes: str | None = None,
    tracker=None,
):
    """Compute FLOW-GLIDE benchmark metrics, log experiment, and update table."""
    from experiment_tracker import ExperimentTracker
    from scripts.score_benchmark import score_test_set

    tracker = tracker or ExperimentTracker(
        log_dir=PROJECT_ROOT / "experiments",
        project_name="AirfRANS 2D Airfoil - GNN Surrogate",
        reference_path=PROJECT_ROOT / "benchmark/benchmark_reference.json",
    )

    try:
        test_graphs = data_bundle.val_graphs
        if not isinstance(test_graphs, list) or len(test_graphs) == 0:
            print("[experiment log] No test graphs found in data_bundle.val_graphs; skipping experiment log.")
            return None

        test_prepped = [_prep_graph_for_norm(g) for g in test_graphs]
        test_ds = NormalizedDataset(test_prepped, data_bundle.x_scaler, data_bundle.y_scaler)
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pyg,
        )

        benchmark_metrics = score_test_set(
            test_loader,
            model,
            data_bundle.x_scaler,
            data_bundle.y_scaler,
            device,
            verbose=False,
        )
    except Exception as exc:
        print(f"[experiment log] Benchmark scoring failed: {exc}")
        benchmark_metrics = {}

    run_config = asdict(scfg)
    train_metrics = {
        "status": "completed",
        "best_val_loss": train_summary.get("best_val", float("nan")),
        "best_epoch": train_summary.get("best_epoch", -1),
        "final_train_loss": train_summary.get("final_train_loss", float("nan")),
        "final_val_loss": train_summary.get("final_val_loss", float("nan")),
        "artifacts_uploaded": train_summary.get("artifacts_uploaded", 0),
    }

    model_name = (
        getattr(scfg, "wandb_name", None)
        or getattr(scfg, "wandb_run_name", None)
        or f"{scfg.task}-h{scfg.hidden}-l{scfg.layers}"
    )

    try:
        exp_id = tracker.log_experiment(
            config=run_config,
            metrics=train_metrics,
            model=model,
            model_name=model_name,
            notes=notes or "",
            duration_sec=float(training_duration_sec),
            benchmark_metrics=benchmark_metrics,
        )
        try:
            _update_flow_glide_comparison_table()
        except Exception as table_exc:
            print(f"[experiment log] Flow-Glide comparison table update failed: {table_exc}")
        print(f"[experiment log] Logged experiment {exp_id} -> experiments/EXPERIMENT_LOG.md")
        print(f"[experiment log] Flow-Glide comparison table updated -> {FLOW_GLIDE_TABLE_PATH}")
        print(f"[experiment log] Benchmark metrics: "
              f"volume_rel_l2={benchmark_metrics.get('volume_rel_l2', float('nan')):.4f}, "
              f"surface_rel_l2={benchmark_metrics.get('surface_rel_l2', float('nan')):.4f}, "
              f"cd_relative_error={benchmark_metrics.get('cd_relative_error', float('nan')):.4f}, "
              f"cl_relative_error={benchmark_metrics.get('cl_relative_error', float('nan')):.4f}, "
              f"rho_d={benchmark_metrics.get('rho_d', float('nan')):.4f}, "
              f"rho_l={benchmark_metrics.get('rho_l', float('nan')):.4f}")

        return exp_id
    except Exception as exc:
        print(f"[experiment log] Experiment tracker write failed: {exc}")
        return None
