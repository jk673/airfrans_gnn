#!/usr/bin/env python3
"""Run a training experiment with benchmark scoring and experiment tracking.

Usage:
    python scripts/run_experiment.py --epochs 100 --hidden 32 --layers 8 --model-name "Test A"
    python scripts/run_experiment.py --epochs 100 --hidden 64 --layers 8 --model-name "Test B" --lr 1e-3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from src.training_common import (
    SmokeCfg, load_and_prepare_data, collate_pyg, NormalizedDataset,
    set_seed, train_epoch, run_epoch, create_lr_scheduler,
)
from src.global_context_processor import EnhancedCFDModelWithGlobalContext
from src.navier_stokes_physics_loss import NavierStokesPhysicsLoss
from src.utils import _prep_graph_for_norm
from scripts.score_benchmark import score_test_set
from src.experiment_tracker import ExperimentTracker


def parse_args():
    p = argparse.ArgumentParser(description="Train + benchmark score + log experiment")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--task", type=str, default="scarce")
    p.add_argument("--model-name", type=str, default="Ours")
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="disabled")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # --- Config ---
    scfg = SmokeCfg()
    scfg.task = args.task
    scfg.hidden = args.hidden
    scfg.layers = args.layers
    scfg.lr = args.lr
    scfg.batch_size = args.batch_size
    scfg.epochs = args.epochs
    scfg.wandb_mode = args.wandb_mode

    run_config = asdict(scfg)
    print(f"Config: epochs={scfg.epochs}, hidden={scfg.hidden}, layers={scfg.layers}, lr={scfg.lr}")

    # --- Data ---
    print("Loading data...")
    data_bundle = load_and_prepare_data(scfg)
    train_loader = data_bundle.train_loader
    val_loader = data_bundle.val_loader

    # --- Model ---
    model = EnhancedCFDModelWithGlobalContext(
        node_feat_dim=7, edge_feat_dim=5,
        hidden_dim=scfg.hidden, output_dim=4,
        num_mp_layers=scfg.layers,
        dropout_p=0.1, config=scfg,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # --- Physics loss ---
    loss_fn = NavierStokesPhysicsLoss(
        continuity_loss_weight=scfg.continuity_loss_weight,
        continuity_target_weight=scfg.continuity_target_weight,
        momentum_loss_weight=scfg.momentum_loss_weight,
        momentum_target_weight=scfg.momentum_target_weight,
        bc_loss_weight=scfg.bc_loss_weight,
        curriculum_ramp_steps=scfg.ramp_epochs,
        ramp_start_step=scfg.ramp_start_epoch,
        ramp_mode=scfg.ramp_mode,
        chord_length=scfg.chord_length,
        nu_molecular=scfg.nu_molecular,
        dynamic_uref_from_data=scfg.dynamic_uref_from_data,
        dynamic_re_from_data=scfg.dynamic_re_from_data,
        uinf_from=scfg.uinf_from,
        use_huber_for_physics=scfg.use_huber_for_physics,
        huber_delta=scfg.huber_delta,
        use_perimeter_norm_for_div=scfg.use_perimeter_norm_for_div,
        div_area_floor_factor=scfg.div_area_floor_factor,
        div_min_degree=scfg.div_min_degree,
    )

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scfg.lr,
        weight_decay=scfg.weight_decay, betas=scfg.betas, eps=scfg.eps,
    )
    scheduler = create_lr_scheduler(optimizer, scfg)
    scaler = GradScaler("cuda", enabled=True) if scfg.amp and torch.cuda.is_available() else None

    # --- Training ---
    print(f"\nTraining for {scfg.epochs} epochs...")
    start_time = time.time()
    global_step = 0
    best_val = float("inf")
    best_epoch = -1
    train_losses = []
    val_losses = []

    for epoch in range(scfg.epochs):
        train_total, train_logs, global_step = train_epoch(
            train_loader, model, optimizer, device, scaler,
            desc=f"train[{epoch}]", loss_fn=loss_fn,
            global_step_start=global_step,
        )
        val_total, val_logs = run_epoch(val_loader, model, device, loss_fn=loss_fn)

        train_losses.append(float(train_total))
        val_losses.append(float(val_total))

        if val_total < best_val:
            best_val = val_total
            best_epoch = epoch
            # Save best checkpoint
            os.makedirs(scfg.ckpt_dir, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": val_total},
                os.path.join(scfg.ckpt_dir, "best.pt"),
            )

        # Step scheduler
        if scheduler is not None:
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(val_total)

        if epoch % 10 == 0 or epoch == scfg.epochs - 1:
            print(f"  Epoch {epoch}: train={train_total:.4f}  val={val_total:.4f}  best_val={best_val:.4f}@{best_epoch}")

    duration = time.time() - start_time
    print(f"\nTraining done in {duration:.1f}s. Best val={best_val:.4f} at epoch {best_epoch}")

    # --- Benchmark Scoring ---
    print("\nRunning benchmark scoring...")

    # Load best checkpoint
    ckpt = torch.load(os.path.join(scfg.ckpt_dir, "best.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    # Build test loader from val_graphs (test split for prebuilt_edges)
    test_graphs = data_bundle.val_graphs
    test_prepped = [_prep_graph_for_norm(g) for g in test_graphs]
    test_ds = NormalizedDataset(test_prepped, data_bundle.x_scaler, data_bundle.y_scaler)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_pyg)
    print(f"Test graphs: {len(test_ds)}")

    benchmark_metrics = score_test_set(
        test_loader, model, data_bundle.x_scaler, data_bundle.y_scaler, device,
    )

    print(f"\nBenchmark Results:")
    print(f"  Volume Rel. L2:    {benchmark_metrics['volume_rel_l2']:.6f}")
    print(f"  Surface Rel. L2:   {benchmark_metrics['surface_rel_l2']:.6f}")
    print(f"  CD Relative Error: {benchmark_metrics['cd_relative_error']:.6f}")
    print(f"  CL Relative Error: {benchmark_metrics['cl_relative_error']:.6f}")
    print(f"  rho_D (Spearman):  {benchmark_metrics['rho_d']:.4f}")
    print(f"  rho_L (Spearman):  {benchmark_metrics['rho_l']:.4f}")

    # --- Log to Experiment Tracker ---
    tracker = ExperimentTracker(
        log_dir="experiments",
        project_name="AirfRANS 2D Airfoil - GNN Surrogate",
    )

    train_metrics = {
        "status": "completed",
        "final_train_loss": train_losses[-1] if train_losses else float("nan"),
        "final_val_loss": val_losses[-1] if val_losses else float("nan"),
        "best_val_loss": float(best_val),
        "best_epoch": best_epoch,
    }

    exp_id = tracker.log_experiment(
        config=run_config,
        metrics=train_metrics,
        model=model,
        model_name=args.model_name,
        notes=args.notes or f"epochs={scfg.epochs}, hidden={scfg.hidden}, layers={scfg.layers}, lr={scfg.lr}",
        duration_sec=duration,
        benchmark_metrics=benchmark_metrics,
    )

    print(f"\nExperiment {exp_id} saved.")
    print(f"  JSON: experiments/results/{exp_id}.json")
    print(f"  Log:  experiments/EXPERIMENT_LOG.md")


if __name__ == "__main__":
    main()
