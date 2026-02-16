#!/usr/bin/env python3
"""
scripts/train.py — Main training pipeline for AirfRANS GNN.
Converted from 01_trainer.ipynb with CLI argument support.

Usage:
    python scripts/train.py [OPTIONS]
    python scripts/train.py --help

Examples:
    # Basic training with default settings
    python scripts/train.py

    # Train with custom hyperparameters
    python scripts/train.py --batch-size 4 --epochs 200 --lr 3e-4 --hidden 256

    # Train with different task and scheduler
    python scripts/train.py --task full --lr-scheduler cosine --cosine-T-max 100

    # Disable global context tokens
    python scripts/train.py --no-global-tokens

    # Train with custom physics loss weights
    python scripts/train.py --continuity-target-weight 0.3 --momentum-target-weight 0.3
"""

from __future__ import annotations

import argparse
import os
import math
import gc
import time
import sys
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from matplotlib.tri import Triangulation
from matplotlib.path import Path
import matplotlib.pyplot as plt
import wandb

from src.training_common import (
    SmokeCfg, DataBundle,
    apply_config_dict,
    DEFAULT_CONFIG_PATH,
    set_seed, get_lr,
    load_and_prepare_data,
    load_config_file,
    run_epoch, train_epoch, create_lr_scheduler, init_wandb,
)
from experiment_tracker import ExperimentTracker

from src.navier_stokes_physics_loss import NavierStokesPhysicsLoss
from src.global_context_processor import EnhancedCFDModelWithGlobalContext
from src.utils import get_surface_mask, with_pos2, ensure_edge_features, _prep_graph_for_norm


# ---------------------------------------------------------------------------
# Training loop with checkpointing + W&B artifacts
# ---------------------------------------------------------------------------

def train_with_scheduler(model, optim, scheduler, train_loader, val_loader,
                         scfg, device, physics_loss_fn):
    scaler = torch.amp.GradScaler('cuda', enabled=(scfg.amp and torch.cuda.is_available()))  # type: ignore[attr-defined]
    global_step = 0
    best_val = float('inf')
    best_epoch = -1
    final_train_total = float('nan')
    final_val_total = float('nan')

    USE_WANDB_ARTIFACTS = getattr(scfg, "use_wandb_artifacts", False)
    ARTIFACT_SAVE_BEST_ONLY = getattr(scfg, "artifact_save_best_only", True)
    ARTIFACT_SAVE_INTERVAL = getattr(scfg, "artifact_save_interval", 20)
    EPOCHS = getattr(scfg, "epochs", 50)
    ckpt_dir = getattr(scfg, "ckpt_dir", "checkpoints")
    ckpt_interval = max(1, getattr(scfg, "ckpt_interval", 5))

    artifact_history = {'best_uploaded': False, 'last_periodic_epoch': -1, 'total_artifacts': 0}

    for epoch in range(EPOCHS):
        train_total, train_logs, global_step = train_epoch(
            train_loader, model, optim, device, scaler,
            amp_enabled=scfg.amp,
            desc=f"train[{epoch}]",
            loss_fn=physics_loss_fn,
            global_step_start=global_step,
            scheduler=scheduler,
            scheduler_step_mode=("step" if getattr(scfg, "scheduler_step_per_batch", False) else "epoch"),
            log_every_n_steps=getattr(scfg, "log_every_n_steps", 25),
        )

        val_total, val_logs = run_epoch(
            val_loader, model, device,
            amp_enabled=scfg.amp,
            loss_fn=physics_loss_fn,
        )
        final_train_total = float(train_total)
        final_val_total = float(val_total)

        # W&B epoch-level logging
        log_epoch = {
            "epoch": epoch,
            "train/total_epoch": train_logs['total_loss'],
            "train/mse_epoch": train_logs['mse_loss'],
            "train/continuity_epoch": train_logs.get('continuity_loss', float('nan')),
            "train/momentum_epoch": train_logs.get('momentum_loss', float('nan')),
            "train/bc_epoch": train_logs.get('bc_loss', float('nan')),
            "val/total_epoch": val_logs.get('total_loss', float('nan')),
            "val/mse_epoch": val_logs.get('mse_loss', float('nan')),
            "val/continuity_epoch": val_logs.get('continuity_loss', float('nan')),
            "val/momentum_epoch": val_logs.get('momentum_loss', float('nan')),
            "val/bc_epoch": val_logs.get('bc_loss', float('nan')),
        }
        if 'cont_weight_used' in train_logs:
            log_epoch["weight/cont_used_epoch"] = train_logs['cont_weight_used']
        if 'mom_weight_used' in train_logs:
            log_epoch["weight/mom_used_epoch"] = train_logs['mom_weight_used']
        lr_now = get_lr(optim)
        if lr_now is not None:
            log_epoch["lr_epoch"] = lr_now
        wandb.log(log_epoch, step=global_step, commit=True)

        # LR scheduler step
        if scheduler is not None and not getattr(scfg, "scheduler_step_per_batch", False):
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_total)
            else:
                scheduler.step()

        # Checkpointing
        os.makedirs(ckpt_dir, exist_ok=True)

        is_best = val_total < best_val
        if is_best:
            best_val = val_total
            best_epoch = epoch
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "train_logs": train_logs,
                "val_logs": val_logs,
            }, best_path)

            if USE_WANDB_ARTIFACTS:
                try:
                    art = wandb.Artifact(
                        name="model-best", type="model",
                        description=f"Best model at epoch {epoch} with val_loss={val_total:.4f}",
                        metadata={"epoch": epoch, "val_loss": val_total, "train_loss": train_total, "best_val": best_val},
                    )
                    art.add_file(best_path)
                    if wandb.run is not None:
                        wandb.run.log_artifact(art)
                    artifact_history['best_uploaded'] = True
                    artifact_history['total_artifacts'] += 1
                except Exception as e:
                    print(f"  Failed to upload W&B artifact: {e}")

        if (epoch + 1) % ckpt_interval == 0:
            ep_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
            }, ep_path)

            if USE_WANDB_ARTIFACTS and not ARTIFACT_SAVE_BEST_ONLY:
                if (epoch + 1) % ARTIFACT_SAVE_INTERVAL == 0:
                    try:
                        art = wandb.Artifact(
                            name="model-checkpoint", type="model",
                            description=f"Checkpoint at epoch {epoch + 1}",
                            metadata={"epoch": epoch + 1, "val_loss": val_total, "train_loss": train_total},
                        )
                        art.add_file(ep_path)
                        if wandb.run is not None:
                            wandb.run.log_artifact(art, aliases=[f"epoch-{epoch + 1}"])
                        artifact_history['last_periodic_epoch'] = epoch + 1
                        artifact_history['total_artifacts'] += 1
                    except Exception as e:
                        print(f"  Failed to upload periodic artifact: {e}")

        print(f"Epoch {epoch:3d} | Train: total={train_total:.4f} mse={train_logs['mse_loss']:.4f} "
              f"cont={train_logs.get('continuity_loss', 0):.2e} mom={train_logs.get('momentum_loss', 0):.2e} "
              f"bc={train_logs.get('bc_loss', 0):.2e} | "
              f"Val: total={val_total:.4f} bc={val_logs.get('bc_loss', 0):.2e}"
              f" {'[BEST]' if is_best else ''}")

    wandb.finish()
    return {
        'best_val': best_val,
        'best_epoch': best_epoch,
        'final_train_loss': final_train_total,
        'final_val_loss': final_val_total,
        'artifacts_uploaded': artifact_history['total_artifacts'],
    }


def run_benchmark_and_log_experiment(
    model,
    data_bundle: DataBundle,
    scfg,
    device,
    train_summary: dict,
    training_duration_sec: float,
    notes: str | None = None,
    tracker: ExperimentTracker | None = None,
):
    """Compute FLOW-GLIDE benchmark metrics and append to experiments/EXPERIMENT_LOG.md."""
    from scripts.score_benchmark import score_test_set
    tracker = tracker or ExperimentTracker(
        log_dir="experiments",
        project_name="AirfRANS 2D Airfoil - GNN Surrogate",
        reference_path="benchmark/benchmark_reference.json",
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
        return None

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
    except Exception as exc:
        print(f"[experiment log] Experiment tracker write failed: {exc}")
        return None

    print(f"[experiment log] Logged experiment {exp_id} -> experiments/EXPERIMENT_LOG.md")
    print(f"[experiment log] Benchmark metrics: "
          f"volume_rel_l2={benchmark_metrics.get('volume_rel_l2', float('nan')):.4f}, "
          f"surface_rel_l2={benchmark_metrics.get('surface_rel_l2', float('nan')):.4f}, "
          f"cd_relative_error={benchmark_metrics.get('cd_relative_error', float('nan')):.4f}, "
          f"cl_relative_error={benchmark_metrics.get('cl_relative_error', float('nan')):.4f}, "
          f"rho_d={benchmark_metrics.get('rho_d', float('nan')):.4f}, "
          f"rho_l={benchmark_metrics.get('rho_l', float('nan')):.4f}")

    return exp_id


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _surface_volume_masks_from_orig(d: Data):
    x = d.x
    if x is not None and x.size(1) >= 5:
        nxy = x[:, 3:5]
        surf = (nxy.abs().sum(dim=1) > 0)
    else:
        surf = get_surface_mask(d)
    vol = ~surf
    return surf, vol


@torch.no_grad()
def mse_per_channel(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor | None = None):
    dev = y_pred.device
    y_t = y_true.to(dev)
    if mask is not None:
        m = mask.to(dev)
        y_p = y_pred[m]
        y_t = y_t[m]
    else:
        y_p = y_pred
    err = (y_p - y_t) ** 2
    if err.numel() == 0:
        return [float('nan')] * y_true.size(1)
    return [float(err[:, i].mean().item()) for i in range(y_true.size(1))]


@torch.no_grad()
def _predict_one_local(d: Data, model, device, x_scaler, y_scaler, amp_enabled: bool = False):
    dm = d.clone()
    assert dm.x is not None, 'x is None'
    if dm.x.size(1) == 5 or not getattr(dm, 'pos2_appended', False):
        dm = with_pos2(dm)
    assert hasattr(dm, 'edge_index') and dm.edge_index is not None, 'edge_index missing'
    dm = ensure_edge_features(dm, want_dim=5)
    dm.x_orig = dm.x

    dm_norm = dm.clone()
    dm_norm.x = dm.x
    dm_norm.y = dm.y
    dm_norm.x_orig = dm.x
    dm_norm.x_norm_params = {'mean': x_scaler.mean.clone(), 'scale': x_scaler.std.clone()}
    dm_norm.y_norm_params = {'mean': y_scaler.mean.clone(), 'scale': y_scaler.std.clone()}

    dm_run = dm_norm.to(device)
    with torch.autocast(device_type='cuda', enabled=(amp_enabled and torch.cuda.is_available())):
        y_pred_norm = model(dm_run).detach().cpu()
    return dm_norm, y_pred_norm


def evaluate_model(model, device, data_bundle: DataBundle, scfg):
    """Evaluate model on validation graphs and print MSE statistics."""
    x_scaler = data_bundle.x_scaler
    y_scaler = data_bundle.y_scaler
    val_edges = data_bundle.val_graphs
    train_edges = data_bundle.train_graphs

    if isinstance(val_edges, list) and len(val_edges) > 0:
        d_eval = val_edges[0]
    else:
        d_eval = train_edges[0] if len(train_edges) > 0 else None

    if d_eval is None:
        print('No sample available for MSE computation')
        return

    d_orig = d_eval
    dm_eval_norm, y_pred_eval_norm = _predict_one_local(d_orig, model, device, x_scaler, y_scaler, scfg.amp)
    surf_mask, vol_mask = _surface_volume_masks_from_orig(d_orig)

    assert isinstance(dm_eval_norm.y, torch.Tensor), 'y must be a Tensor'
    names = ['u', 'v', 'p_over_rho', 'nu_t']
    mse_all = mse_per_channel(y_pred_eval_norm, dm_eval_norm.y, None)
    mse_surf = mse_per_channel(y_pred_eval_norm, dm_eval_norm.y, surf_mask)
    mse_vol = mse_per_channel(y_pred_eval_norm, dm_eval_norm.y, vol_mask)

    print('[MSE | ALL   | normalized]', {n: f'{v:.4e}' for n, v in zip(names, mse_all)})
    print('[MSE | SURF  | normalized]', {n: f'{v:.4e}' for n, v in zip(names, mse_surf)})
    print('[MSE | VOLUME| normalized]', {n: f'{v:.4e}' for n, v in zip(names, mse_vol)})


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_one_for_viz(d: Data, model, device, x_scaler, y_scaler):
    dm = d.clone()
    assert dm.x is not None, 'x is None'
    if dm.x.size(1) == 5 or not getattr(dm, 'pos2_appended', False):
        dm = with_pos2(dm)
    assert hasattr(dm, 'edge_index') and dm.edge_index is not None, 'edge_index missing'
    dm = ensure_edge_features(dm, want_dim=5)
    dm_norm = dm.clone()
    dm_norm.x = x_scaler.transform(dm.x)
    dm_norm.y = y_scaler.transform(dm.y)
    dm_run = dm_norm.to(device)

    model.eval()
    if dm_run.x is not None:
        dm_run.x = dm_run.x.float()
    if dm_run.edge_attr is not None:
        dm_run.edge_attr = dm_run.edge_attr.float()

    y_pred_norm = model(dm_run).detach().cpu()
    return dm_norm, y_pred_norm


@torch.no_grad()
def plot_pred_vs_gt(dm: Data, y_pred: torch.Tensor, y_scaler, x_scaler,
                    channel: int = 2, show_mesh: bool = True, mask_airfoil: bool = True,
                    titles: tuple = ('Ground Truth', 'Prediction', 'Abs Error'),
                    cmap: str = 'viridis', denormalize: bool = True,
                    save_dir: str = "figures", save: bool = True):
    assert isinstance(dm.y, torch.Tensor), 'dm.y must be a Tensor'
    assert isinstance(dm.x, torch.Tensor), 'dm.x must be a Tensor'
    if denormalize:
        gt = y_scaler.inverse(dm.y.detach().cpu())
        pr = y_scaler.inverse(y_pred.detach().cpu())
    else:
        gt = dm.y.detach().cpu()
        pr = y_pred.detach().cpu()

    pos_src = dm.pos if dm.pos is not None else dm.x
    xy = pos_src[:, :2].detach().cpu().float().numpy()
    tri = Triangulation(xy[:, 0], xy[:, 1])

    dm_x_cpu = dm.x.detach().cpu()
    x_phys = x_scaler.inverse(dm_x_cpu)
    vx = float(x_phys[0, 0])
    vy = float(x_phys[0, 1])
    v = math.sqrt(vx * vx + vy * vy)
    q = 0.5 * v * v

    gt_c = gt[:, channel].detach().cpu().float().numpy() / q
    pr_c = pr[:, channel].detach().cpu().float().numpy() / q
    err = np.abs(pr_c - gt_c)
    vmin = -2.0
    vmax = 1.0

    if mask_airfoil:
        try:
            x_np = x_phys.detach().cpu().numpy()
            pos_src2 = dm.pos if dm.pos is not None else dm.x
            assert isinstance(pos_src2, torch.Tensor)
            pos_np = pos_src2[:, :2].detach().cpu().numpy()
            surf_mask = None
            wall = x_np[:, 2] if x_np.shape[1] >= 3 else None
            nxy = x_np[:, 3:5] if x_np.shape[1] >= 5 else None
            if nxy is not None:
                surf_mask = (np.abs(nxy).sum(axis=1) > 1e-8)
                if wall is not None:
                    surf_mask = np.logical_or(surf_mask, (wall < 1e-6))
            elif wall is not None:
                surf_mask = (wall < 1e-6)

            if surf_mask is not None and np.any(surf_mask):
                pts = pos_np[surf_mask]
                if pts.shape[0] >= 3:
                    c = pts.mean(axis=0)
                    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
                    order = np.argsort(ang)
                    poly = Path(pts[order], closed=True)
                    tris = tri.triangles
                    centers = np.mean(np.stack([tri.x[tris], tri.y[tris]], axis=-1), axis=1)
                    inside = poly.contains_points(centers, radius=-1e-6)
                    if inside is not None and inside.any():
                        tri.set_mask(inside.astype(bool))
        except Exception:
            pass

    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    a1, a2, a3 = ax
    a1.tricontourf(tri, gt_c, levels=50, vmin=vmin, vmax=vmax, cmap=cmap)
    a2.tricontourf(tri, pr_c, levels=50, vmin=vmin, vmax=vmax, cmap=cmap)
    a3.tricontourf(tri, err, levels=50, cmap='magma')
    if show_mesh:
        for a in (a1, a2, a3):
            a.triplot(tri, color='k', lw=0.25, alpha=0.35)
    for a, t in zip((a1, a2, a3), titles):
        a.set_aspect('equal', 'box')
        a.set_title(t)
        a.set_xlabel('x')
        a.set_ylabel('y')
    m1 = plt.cm.ScalarMappable(cmap=cmap)
    m1.set_clim(vmin, vmax)
    fig.colorbar(m1, ax=[a1, a2], fraction=0.046, pad=0.04, label='Cp')
    fig.colorbar(plt.cm.ScalarMappable(cmap='magma'), ax=a3, fraction=0.046, pad=0.04, label='abs error')

    if save:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"pred_vs_gt_channel{channel}_{timestamp}.png")
        plt.savefig(filename, dpi=200)
        print(f"[Saved] {filename}")

    plt.show()


@torch.no_grad()
def plot_surface_pressure(d: Data, model, device, x_scaler, y_scaler, amp_enabled: bool = False):
    assert isinstance(d.x, torch.Tensor), 'd.x must be a Tensor'
    dm_norm, y_pred_norm = _predict_one_local(d, model, device, x_scaler, y_scaler, amp_enabled)
    assert isinstance(dm_norm.y, torch.Tensor), 'dm_norm.y must be a Tensor'
    y_pred_denorm = y_scaler.inverse(y_pred_norm.cpu())
    y_true_denorm = y_scaler.inverse(dm_norm.y.cpu())

    xvars = d.x.cpu()
    pos = d.pos.cpu() if isinstance(d.pos, torch.Tensor) else d.x[:, :2].cpu()

    normals = xvars[:, 3:5]
    surf_mask = (normals.abs().sum(dim=1) > 0)
    surf_idx = torch.nonzero(surf_mask, as_tuple=False).squeeze(-1).long()

    if surf_idx.numel() < 5:
        print("Not enough surface nodes to plot.")
        return

    pos_surf = pos[surf_idx]
    ord_idx = torch.argsort(pos_surf[:, 0])
    ordered_pos = pos_surf[ord_idx]

    p_true = torch.index_select(y_true_denorm, 0, surf_idx)[:, 2]
    p_true_ordered = p_true[ord_idx]
    p_pred = torch.index_select(y_pred_denorm, 0, surf_idx)[:, 2]
    p_pred_ordered = p_pred[ord_idx]

    _, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ordered_pos[:, 0], p_true_ordered, 'o-', label='Ground Truth Pressure', markersize=4)
    ax.plot(ordered_pos[:, 0], p_pred_ordered, 'x-', label='Predicted Pressure', markersize=4)
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Pressure / Density (p/rho)')
    ax.set_title('Surface Pressure Distribution')
    ax.legend()
    ax.grid(True)
    ax.invert_yaxis()
    plt.show()


# ---------------------------------------------------------------------------
# Force coefficients
# ---------------------------------------------------------------------------

def order_surface(pos_surf: torch.Tensor):
    c = pos_surf.mean(dim=0)
    rel = pos_surf - c
    angles = torch.atan2(rel[:, 1], rel[:, 0])
    return torch.argsort(angles)


def panel_lengths(pos_ordered: torch.Tensor):
    rolled = torch.roll(pos_ordered, -1, 0)
    seg = rolled - pos_ordered
    ds = seg.norm(dim=1)
    return ds, seg


def compute_force_coefficients_local(data: Data, eps=1e-9):
    xvars = data.x.cpu() if isinstance(data.x, torch.Tensor) else torch.tensor(data.x)
    yvars = data.y.cpu() if isinstance(data.y, torch.Tensor) else torch.tensor(data.y)
    pos = data.pos.cpu() if data.pos is not None and isinstance(data.pos, torch.Tensor) else None

    if pos is None:
        N = xvars.size(0)
        pos = torch.zeros(N, 2)
        pos[:, 0] = torch.linspace(0, 1, N)

    normals = xvars[:, 3:5]
    surf_mask = (normals.abs().sum(dim=1) > 0)
    surf_idx = torch.nonzero(surf_mask, as_tuple=False).squeeze(-1).long()
    if surf_idx.numel() < 5:
        return None

    pos_surf = pos[surf_idx]
    ord_idx = order_surface(pos_surf).long()
    ordered_pos = pos_surf[ord_idx]
    ds, _ = panel_lengths(ordered_pos)
    n_nodes = normals[surf_idx[ord_idx]]
    n_norm = n_nodes.norm(dim=1, keepdim=True).clamp_min(eps)
    n_hat = n_nodes / n_norm

    if hasattr(data, 'x_norm_params'):
        mean = data.x_norm_params['mean']
        scale = data.x_norm_params['scale']
        u_inf = xvars[0, 0] * scale[0] + mean[0]
        v_inf = xvars[0, 1] * scale[1] + mean[1]
    else:
        u_inf = xvars[0, 0]
        v_inf = xvars[0, 1]
    U_inf = float(torch.sqrt(u_inf ** 2 + v_inf ** 2)) + eps

    temp = torch.index_select(yvars, 0, surf_idx)[:, 2]
    p_over_rho = temp[ord_idx]
    Fp = (-p_over_rho.unsqueeze(1) * n_hat) * ds.unsqueeze(1)
    F = Fp.sum(dim=0)

    chord = float(pos_surf[:, 0].max() - pos_surf[:, 0].min() + eps)
    q_ref = 0.5 * U_inf ** 2
    CD = -F[0] / (q_ref * chord + eps)
    CL = F[1] / (q_ref * chord + eps)
    return {'CD': float(CD), 'CL': float(CL)}


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train AirfRANS GNN model with physics-informed loss',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data configuration
    parser.add_argument('--task', type=str, default='scarce',
                        choices=['full', 'scarce', 'medium'],
                        help='AirfRANS task variant')
    parser.add_argument('--root', type=str, default='Dataset',
                        help='Path to AirfRANS dataset root')
    parser.add_argument('--limit-train', type=int, default=None,
                        help='Limit number of training graphs (None = use all)')
    parser.add_argument('--limit-val', type=int, default=None,
                        help='Limit number of validation graphs (None = use all)')

    # Model architecture
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--layers', type=int, default=14,
                        help='Number of message passing layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')

    # Global context & attention
    parser.add_argument('--use-global-tokens', action='store_true', default=True,
                        help='Enable global context tokens')
    parser.add_argument('--no-global-tokens', dest='use_global_tokens', action='store_false',
                        help='Disable global context tokens')
    parser.add_argument('--num-global-tokens', type=int, default=2,
                        help='Number of global tokens')
    parser.add_argument('--attention-heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--attention-layers', type=int, default=2,
                        help='Number of attention layers')
    parser.add_argument('--attention-dropout', type=float, default=0.0,
                        help='Attention dropout rate')
    parser.add_argument('--use-cross-attention', action='store_true', default=True,
                        help='Use cross-attention between nodes and global tokens')
    parser.add_argument('--global-pooling-type', type=str, default='attention',
                        choices=['attention', 'mean', 'max', 'set2set'],
                        help='Global pooling mechanism')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        help='AdamW weight decay')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision (AMP)')

    # Learning rate scheduler
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=['cosine', 'cosine_warm_restarts', 'reduce_on_plateau', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--cosine-T-max', type=int, default=80,
                        help='T_max for cosine annealing')
    parser.add_argument('--cosine-eta-min', type=float, default=1e-6,
                        help='Minimum LR for cosine annealing')

    # Physics loss configuration
    parser.add_argument('--data-loss-weight', type=float, default=1.0,
                        help='Weight for data MSE loss')
    parser.add_argument('--continuity-loss-weight', type=float, default=0.05,
                        help='Initial continuity loss weight')
    parser.add_argument('--continuity-target-weight', type=float, default=0.20,
                        help='Target continuity loss weight after ramp')
    parser.add_argument('--momentum-loss-weight', type=float, default=0.05,
                        help='Initial momentum loss weight')
    parser.add_argument('--momentum-target-weight', type=float, default=0.20,
                        help='Target momentum loss weight after ramp')
    parser.add_argument('--bc-loss-weight', type=float, default=0.1,
                        help='Boundary condition loss weight')
    parser.add_argument('--ramp-start-epoch', type=int, default=40,
                        help='Epoch to start physics loss curriculum ramp')
    parser.add_argument('--ramp-epochs', type=int, default=60,
                        help='Number of epochs for curriculum ramp')
    parser.add_argument('--ramp-mode', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Curriculum ramp schedule mode')

    # Physics parameters
    parser.add_argument('--chord-length', type=float, default=1.0,
                        help='Reference chord length')
    parser.add_argument('--nu-molecular', type=float, default=1.5e-5,
                        help='Molecular viscosity')
    parser.add_argument('--dynamic-uref', action='store_true', default=True,
                        help='Dynamically compute U_ref from data')
    parser.add_argument('--dynamic-re', action='store_true', default=True,
                        help='Dynamically compute Reynolds number from data')
    parser.add_argument('--use-huber-physics', action='store_true', default=True,
                        help='Use Huber loss for physics terms')
    parser.add_argument('--huber-delta', type=float, default=0.05,
                        help='Huber loss delta parameter')

    # Checkpointing
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--ckpt-interval', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Weights & Biases
    parser.add_argument('--wandb-project', type=str, default='airfrans-gnn',
                        help='W&B project name')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--wandb-mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='W&B logging mode')
    parser.add_argument('--enable-wandb', dest='enable_wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--disable-wandb', dest='enable_wandb', action='store_false',
                        help='Disable W&B logging')
    parser.set_defaults(enable_wandb=False)
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=None,
                        help='W&B tags for this run')
    parser.add_argument('--use-wandb-artifacts', action='store_true', default=False,
                        help='Upload model checkpoints as W&B artifacts')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto-detect if not specified)')
    parser.add_argument('--log-every-n-steps', type=int, default=-1,
                        help='Log to W&B every N steps (-1 for epoch-only)')
    parser.add_argument('--no-viz', action='store_true', default=False,
                        help='Skip visualization at the end of training')
    parser.add_argument(
        '--config',
        type=str,
        default=os.environ.get("AIRFRANS_TRAIN_CONFIG", str(DEFAULT_CONFIG_PATH)),
        help=f'Path to JSON config file (defaults to {DEFAULT_CONFIG_PATH})'
    )

    return parser.parse_args(argv)


def create_config_from_args(args):
    """Create SmokeCfg from command line arguments"""
    parser_defaults = vars(parse_args([]))
    cfg = SmokeCfg()
    for key, value in parser_defaults.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    apply_config_dict(cfg, load_config_file(getattr(args, 'config', None)))

    arg_values = vars(args)
    for key, value in arg_values.items():
        if key == 'config' or not hasattr(cfg, key):
            continue
        if key in {'limit_train', 'limit_val'} and value is None:
            continue
        if value != parser_defaults.get(key):
            if key == 'lr_scheduler':
                value = None if value in ('', 'none', 'None') else value
            setattr(cfg, key, value)

    return cfg


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    from dotenv import load_dotenv
    load_dotenv()

    # Parse command line arguments
    args = parse_args()
    scfg = create_config_from_args(args)

    set_seed(scfg.seed)

    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 80)
    print('AirfRANS GNN Training')
    print('=' * 80)
    print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}')
    print(f'Task: {scfg.task}')
    print(f'Batch size: {scfg.batch_size} | Epochs: {scfg.epochs} | LR: {scfg.lr}')
    print(f'Hidden: {scfg.hidden} | Layers: {scfg.layers}')
    print(f'Global tokens: {scfg.use_global_tokens} (num={scfg.num_global_tokens})')
    print(f'LR scheduler: {scfg.lr_scheduler}')
    print('=' * 80)
    training_start = time.time()

    # --- Data ---
    data_bundle = load_and_prepare_data(scfg)

    # --- Model ---
    node_dim = 7
    edge_dim = 5

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = EnhancedCFDModelWithGlobalContext(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=scfg.hidden,
        output_dim=4,
        num_mp_layers=scfg.layers,
        dropout_p=scfg.dropout,
        config=scfg,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scfg.lr,
        weight_decay=scfg.weight_decay, betas=scfg.betas, eps=scfg.eps,
    )
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # --- Physics loss ---
    steps_per_epoch = len(data_bundle.train_loader)
    loss_fn = NavierStokesPhysicsLoss(
        data_loss_weight=scfg.data_loss_weight,
        continuity_loss_weight=scfg.continuity_loss_weight,
        continuity_target_weight=scfg.continuity_target_weight,
        momentum_loss_weight=scfg.momentum_loss_weight,
        momentum_target_weight=scfg.momentum_target_weight,
        curriculum_ramp_steps=scfg.ramp_epochs * steps_per_epoch,
        ramp_start_step=scfg.ramp_start_epoch * steps_per_epoch,
        ramp_mode=scfg.ramp_mode,
        bc_loss_weight=scfg.bc_loss_weight,
        chord_length=scfg.chord_length,
        dynamic_uref_from_data=scfg.dynamic_uref_from_data,
        dynamic_re_from_data=scfg.dynamic_re_from_data,
        nu_molecular=scfg.nu_molecular,
        uinf_from=scfg.uinf_from,
        use_huber_for_physics=scfg.use_huber_for_physics,
        huber_delta=scfg.huber_delta,
        use_perimeter_norm_for_div=scfg.use_perimeter_norm_for_div,
        div_area_floor_factor=scfg.div_area_floor_factor,
        div_min_degree=scfg.div_min_degree,
        debug=scfg.physics_debug,
        debug_level=scfg.physics_debug_level,
        debug_every=scfg.physics_debug_every,
    )
    print(f"Physics loss initialized: cont {scfg.continuity_loss_weight:.3f}->{scfg.continuity_target_weight:.3f}, "
          f"mom {scfg.momentum_loss_weight:.3f}->{scfg.momentum_target_weight:.3f}")

    # --- LR Scheduler ---
    lr_scheduler = create_lr_scheduler(optimizer, scfg)

    # --- W&B ---
    init_wandb(scfg, loss_fn)

    # --- Train ---
    train_summary = train_with_scheduler(
        model, optimizer, lr_scheduler,
        data_bundle.train_loader, data_bundle.val_loader,
        scfg, device, loss_fn,
    )
    training_duration = time.time() - training_start

    # --- Evaluate ---
    evaluate_model(model, device, data_bundle, scfg)

    # --- Visualize ---
    if not args.no_viz:
        val_edges = data_bundle.val_graphs
        train_edges = data_bundle.train_graphs
        d_vis = val_edges[0] if len(val_edges) > 0 else (train_edges[0] if len(train_edges) > 0 else None)
        if d_vis is not None:
            dm_vis_n, y_pred_vis_n = _predict_one_for_viz(
                d_vis, model, device, data_bundle.x_scaler, data_bundle.y_scaler)
            plot_pred_vs_gt(
                dm_vis_n, y_pred_vis_n,
                data_bundle.y_scaler, data_bundle.x_scaler,
                channel=2, show_mesh=False, denormalize=True,
            )

    # --- Benchmark + experiment tracker ---
    run_benchmark_and_log_experiment(
        model=model,
        data_bundle=data_bundle,
        scfg=scfg,
        device=device,
        train_summary=train_summary,
        training_duration_sec=training_duration,
        notes=f"task={scfg.task}, hidden={scfg.hidden}, layers={scfg.layers}, lr={scfg.lr}, scheduler={scfg.lr_scheduler}",
    )

    print('\n' + '=' * 80)
    print('Training complete!')
    print('=' * 80)
    print(f'Best checkpoint saved to: {os.path.join(scfg.ckpt_dir, "best.pt")}')


if __name__ == '__main__':
    main()
