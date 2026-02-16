#!/usr/bin/env python3
"""
scripts/train_multi_scale.py — Multi-scale model training with turbulence physics.
Converted from 02_trainer_multi_scale.ipynb.

Usage:
    python scripts/train_multi_scale.py
"""

from __future__ import annotations

import os
import math
import contextlib
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Data
from matplotlib.tri import Triangulation
from matplotlib.path import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb

from src.training_common import (
    SmokeCfg, DataBundle,
    set_seed, get_lr, collate_pyg,
    load_and_prepare_data, run_epoch,
    create_lr_scheduler, init_wandb,
)
from src.turbulent_modeling_physics_loss import EnhancedPhysicsLoss
from src.global_context_processor import UltraEnhancedCFDModel
from src.utils import get_surface_mask, with_pos2, ensure_edge_features


# ---------------------------------------------------------------------------
# Enhanced loss computation (8-term loss)
# ---------------------------------------------------------------------------

def compute_enhanced_loss(predictions, targets, data, loss_fn=None, step=None):
    """Compute loss using enhanced physics loss with NaN/Inf handling."""
    if loss_fn is not None:
        try:
            if torch.isnan(predictions).any():
                print("Warning: NaN in predictions, using MSE fallback")
                mse_loss = F.mse_loss(predictions[~torch.isnan(predictions)],
                                      targets[~torch.isnan(predictions)])
                if torch.isnan(mse_loss):
                    mse_loss = torch.tensor(1.0, device=predictions.device)
                return mse_loss, {
                    'mse_loss': float(mse_loss.item()), 'total_loss': float(mse_loss.item()),
                    'continuity_loss': 0.0, 'momentum_loss': 0.0, 'bc_loss': 0.0,
                    'turbulence_production_loss': 0.0, 'turbulence_dissipation_loss': 0.0,
                    'smoothness_loss': 0.0, 'wall_function_loss': 0.0,
                }

            loss_dict = loss_fn(predictions, targets, data=data, step=step)
            total_loss = loss_dict['total_loss']

            if torch.isnan(total_loss).any():
                print("Warning: NaN in physics loss, falling back to MSE")
                mse_loss = F.mse_loss(predictions, targets)
                return mse_loss, {
                    'mse_loss': float(mse_loss.item()), 'total_loss': float(mse_loss.item()),
                    'continuity_loss': 0.0, 'momentum_loss': 0.0, 'bc_loss': 0.0,
                    'turbulence_production_loss': 0.0, 'turbulence_dissipation_loss': 0.0,
                    'smoothness_loss': 0.0, 'wall_function_loss': 0.0,
                }

            log_dict = {}
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    val = v.detach().item() if v.numel() == 1 else v.detach().mean().item()
                    log_dict[k] = float(val) if not torch.isnan(v).any() else 0.0
                else:
                    log_dict[k] = float(v) if not (isinstance(v, float) and math.isnan(v)) else 0.0

            return total_loss, log_dict

        except Exception as e:
            print(f"Warning: Enhanced physics loss failed ({e}), falling back to MSE")
            mse_loss = F.mse_loss(predictions, targets)
            return mse_loss, {
                'mse_loss': float(mse_loss.item()), 'total_loss': float(mse_loss.item()),
                'continuity_loss': 0.0, 'momentum_loss': 0.0, 'bc_loss': 0.0,
                'turbulence_production_loss': 0.0, 'turbulence_dissipation_loss': 0.0,
                'smoothness_loss': 0.0, 'wall_function_loss': 0.0,
            }
    else:
        mse_loss = F.mse_loss(predictions, targets)
        return mse_loss, {
            'mse_loss': float(mse_loss.item()), 'total_loss': float(mse_loss.item()),
            'continuity_loss': 0.0, 'momentum_loss': 0.0, 'bc_loss': 0.0,
            'turbulence_production_loss': 0.0, 'turbulence_dissipation_loss': 0.0,
            'smoothness_loss': 0.0, 'wall_function_loss': 0.0,
        }


# ---------------------------------------------------------------------------
# Enhanced training epoch with gradient stability
# ---------------------------------------------------------------------------

def train_epoch_enhanced(loader, model, optim, device, scaler, desc='train',
                         loss_fn=None, global_step_start=0):
    model.train()
    losses_tracker = {
        'total': [], 'mse': [], 'continuity': [], 'momentum': [],
        'turbulence_production': [], 'turbulence_dissipation': [],
        'smoothness': [], 'wall_function': [], 'bc': [],
    }

    global_step = global_step_start
    pbar = tqdm(total=len(loader), desc=desc, leave=False)

    skip_count = 0
    max_skips = len(loader) // 2

    for batch in loader:
        if batch is None:
            pbar.update(1)
            global_step += 1
            continue

        b = batch.to(device)
        optim.zero_grad(set_to_none=True)

        try:
            with autocast(enabled=scaler is not None):
                out = model(b)

                if torch.isnan(out).any() or torch.isinf(out).any():
                    print(f"Warning: NaN/Inf in model output at step {global_step}")
                    skip_count += 1
                    if skip_count > max_skips:
                        print(f"Too many skips ({skip_count}), stopping training")
                        break
                    pbar.update(1)
                    global_step += 1
                    continue

                out = torch.clamp(out, -10.0, 10.0)
                loss, loss_dict = compute_enhanced_loss(out, b.y, b, loss_fn, step=global_step)

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000.0:
                print(f"Warning: Invalid/Large loss at step {global_step}: {loss.item()}")
                skip_count += 1
                if skip_count > max_skips:
                    break
                pbar.update(1)
                global_step += 1
                continue

            # Backward with aggressive gradient clipping
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 5.0:
                    skip_count += 1
                    if skip_count > max_skips:
                        break
                    optim.zero_grad(set_to_none=True)
                    pbar.update(1)
                    global_step += 1
                    continue
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 5.0:
                    skip_count += 1
                    if skip_count > max_skips:
                        break
                    optim.zero_grad(set_to_none=True)
                    pbar.update(1)
                    global_step += 1
                    continue
                optim.step()

            # Track losses
            for key in losses_tracker:
                loss_key = f'{key}_loss' if key != 'total' else 'total_loss'
                if loss_key in loss_dict:
                    val = loss_dict[loss_key]
                    if not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                        losses_tracker[key].append(val)

        except Exception as e:
            print(f"Error in training step {global_step}: {e}")
            skip_count += 1
            if skip_count > max_skips:
                break
            pbar.update(1)
            global_step += 1
            continue

        pbar.set_postfix({
            'total': f"{loss_dict.get('total_loss', 0):.4e}",
            'grad': f"{grad_norm:.2e}" if 'grad_norm' in dir() else "N/A",
            'skips': skip_count,
        })
        pbar.update(1)
        global_step += 1

    pbar.close()

    if skip_count > 0:
        print(f"Training epoch completed with {skip_count} skipped batches")

    avg_losses = {}
    for k, v in losses_tracker.items():
        key = f'{k}_loss' if k != 'total' else 'total_loss'
        if v:
            avg = np.nanmean(v)
            avg_losses[key] = avg if not np.isnan(avg) else 0.0
        else:
            avg_losses[key] = 0.0

    if 'total_loss' not in avg_losses or avg_losses['total_loss'] == 0.0:
        if 'mse_loss' in avg_losses and avg_losses['mse_loss'] > 0:
            avg_losses['total_loss'] = avg_losses['mse_loss']
        else:
            avg_losses['total_loss'] = 1.0

    return avg_losses['total_loss'], avg_losses, global_step


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def init_weights_carefully(m):
    """Initialize weights conservatively to prevent gradient explosion."""
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
# Evaluation helpers (same as train.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _surface_volume_masks_from_orig(d: Data):
    x = d.x
    if x is not None and x.size(1) >= 5:
        nxy = x[:, 3:5]
        surf = (nxy.abs().sum(dim=1) > 0)
    else:
        surf = get_surface_mask(d)
    return surf, ~surf


@torch.no_grad()
def mse_per_channel(y_pred, y_true, mask=None):
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
def _predict_one_local(d, model, device, x_scaler, y_scaler, amp_enabled=False):
    dm = Data(**{k: v for k, v in d})
    if dm.x.size(1) == 5 or not getattr(dm, 'pos2_appended', False):
        dm = with_pos2(dm)
    assert hasattr(dm, 'edge_index') and dm.edge_index is not None
    dm = ensure_edge_features(dm, want_dim=5)

    dm_norm = Data(**{k: v for k, v in dm})
    dm_norm.x = x_scaler.transform(dm.x)
    dm_norm.y = y_scaler.transform(dm.y)
    dm_norm.x_norm_params = {'mean': x_scaler.mean.clone(), 'scale': x_scaler.std.clone()}
    dm_norm.y_norm_params = {'mean': y_scaler.mean.clone(), 'scale': y_scaler.std.clone()}

    dm_run = dm_norm.to(device)
    with (torch.amp.autocast(device_type='cuda', enabled=(amp_enabled and torch.cuda.is_available()))
          if torch.cuda.is_available() else contextlib.nullcontext()):
        y_pred_norm = model(dm_run).detach().cpu()
    return dm_norm, y_pred_norm


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_pred_vs_gt(dm, y_pred, y_scaler, x_scaler,
                    channel=2, show_mesh=True, mask_airfoil=True,
                    cmap='viridis', denormalize=True,
                    save_dir="figures", save=True):
    if denormalize and hasattr(dm, 'y'):
        gt = y_scaler.inverse(dm.y.detach().cpu())
        pr = y_scaler.inverse(y_pred.detach().cpu())
    else:
        gt = dm.y.detach().cpu()
        pr = y_pred.detach().cpu()

    xy = (dm.pos if hasattr(dm, 'pos') and dm.pos is not None else dm.x)[:, :2].detach().cpu().float().numpy()
    tri = Triangulation(xy[:, 0], xy[:, 1])

    x_phys = x_scaler.inverse(dm.x.detach().cpu())
    vx = float(x_phys[0, 0])
    vy = float(x_phys[0, 1])
    v = math.sqrt(vx * vx + vy * vy)
    q = 0.5 * v * v

    gt_c = gt[:, channel].numpy() / q
    pr_c = pr[:, channel].numpy() / q
    err = np.abs(pr_c - gt_c)
    vmin, vmax = -2.0, 1.0

    if mask_airfoil:
        try:
            x_np = x_phys.numpy()
            pos_np = xy
            wall = x_np[:, 2] if x_np.shape[1] >= 3 else None
            nxy = x_np[:, 3:5] if x_np.shape[1] >= 5 else None
            surf_mask = None
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
    for a, t in zip((a1, a2, a3), ('Ground Truth', 'Prediction', 'Abs Error')):
        a.set_aspect('equal', 'box')
        a.set_title(t)
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    scfg = SmokeCfg()
    env_wandb = os.getenv("AIRFRANS_ENABLE_WANDB", "").strip().lower()
    scfg.enable_wandb = env_wandb not in {"0", "false", "f", "off", "no"}
    if not scfg.enable_wandb:
        print("[wandb] disabled (set AIRFRANS_ENABLE_WANDB=1 to enable)")
    set_seed(scfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}')
    print('Config:', asdict(scfg))

    # --- Data ---
    data_bundle = load_and_prepare_data(scfg)

    # --- Enhanced Physics Loss ---
    steps_per_epoch = len(data_bundle.train_loader)
    loss_fn = EnhancedPhysicsLoss(
        data_loss_weight=scfg.data_loss_weight,
        continuity_loss_weight=scfg.continuity_loss_weight,
        continuity_target_weight=scfg.continuity_target_weight,
        momentum_loss_weight=scfg.momentum_loss_weight,
        momentum_target_weight=scfg.momentum_target_weight,
        bc_loss_weight=scfg.bc_loss_weight,
        turbulence_loss_weight=0.05,
        rans_loss_weight=0.05,
        smoothness_weight=0.01,
        wall_function_weight=0.02,
        curriculum_ramp_steps=scfg.ramp_epochs * steps_per_epoch,
        ramp_start_step=scfg.ramp_start_epoch * steps_per_epoch,
        ramp_mode=scfg.ramp_mode,
        nu_molecular=scfg.nu_molecular,
        chord_length=scfg.chord_length,
        use_adaptive_weights=False,
        debug=scfg.physics_debug,
    )
    print(f"Enhanced Physics Loss initialized:")
    print(f"  Continuity: {scfg.continuity_loss_weight:.3f} -> {scfg.continuity_target_weight:.3f}")
    print(f"  Momentum: {scfg.momentum_loss_weight:.3f} -> {scfg.momentum_target_weight:.3f}")

    # --- UltraEnhanced Model ---
    enhanced_model = UltraEnhancedCFDModel(
        node_feat_dim=7, edge_feat_dim=5,
        hidden_dim=scfg.hidden, output_dim=4,
        num_mp_layers=scfg.layers,
        num_scales=3,
        dropout_p=0.1,
        config=scfg,
    ).to(device)

    # Differential LR groups
    optimizer_groups = [
        {'params': enhanced_model.base_model.parameters(), 'lr': scfg.lr},
        {'params': enhanced_model.multi_scale_convs.parameters(), 'lr': scfg.lr * 0.5},
        {'params': enhanced_model.output_head.parameters(), 'lr': scfg.lr * 1.5},
    ]
    enhanced_optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=scfg.weight_decay,
        betas=scfg.betas, eps=scfg.eps,
    )

    # Careful weight init
    enhanced_model.apply(init_weights_carefully)
    if hasattr(enhanced_model, 'residual_weight'):
        nn.init.constant_(enhanced_model.residual_weight, 0.01)

    # Reduce initial LR
    for param_group in enhanced_optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.01
    print(f"Initial LR reduced to {enhanced_optimizer.param_groups[0]['lr']:.2e}")

    print(f'Model parameters: {sum(p.numel() for p in enhanced_model.parameters() if p.requires_grad):,}')

    # --- LR Scheduler ---
    enhanced_scheduler = create_lr_scheduler(enhanced_optimizer, scfg)

    # --- W&B ---
    init_wandb(scfg, loss_fn)

    # --- Training loop ---
    history = []
    best_val = float('inf')

    for epoch in range(scfg.epochs):
        # Train
        try:
            train_loss, train_metrics, _ = train_epoch_enhanced(
                data_bundle.train_loader, enhanced_model, enhanced_optimizer,
                device, None, desc=f'Enhanced Train [{epoch}]', loss_fn=loss_fn,
            )

            if math.isnan(train_loss) or math.isinf(train_loss):
                print(f"Training became unstable at epoch {epoch}, reducing learning rate")
                for param_group in enhanced_optimizer.param_groups:
                    param_group['lr'] *= 0.5
                train_loss = 1.0
                train_metrics = {'mse_loss': 1.0, 'total_loss': 1.0,
                                 'continuity_loss': 0.0, 'momentum_loss': 0.0, 'bc_loss': 0.0,
                                 'turbulence_production_loss': 0.0, 'turbulence_dissipation_loss': 0.0,
                                 'smoothness_loss': 0.0, 'wall_function_loss': 0.0}
        except Exception as e:
            print(f"Training failed at epoch {epoch}: {e}")
            for param_group in enhanced_optimizer.param_groups:
                param_group['lr'] *= 0.1
            train_loss = 1.0
            train_metrics = {'mse_loss': 1.0, 'total_loss': 1.0,
                             'continuity_loss': 0.0, 'momentum_loss': 0.0, 'bc_loss': 0.0,
                             'turbulence_production_loss': 0.0, 'turbulence_dissipation_loss': 0.0,
                             'smoothness_loss': 0.0, 'wall_function_loss': 0.0}

        # Validation
        try:
            val_loss, val_metrics = run_epoch(
                data_bundle.val_loader, enhanced_model, device,
                amp_enabled=scfg.amp,
                desc=f'Enhanced Val [{epoch}]', loss_fn=loss_fn,
            )
        except Exception as e:
            print(f"Validation failed at epoch {epoch}: {e}")
            val_loss = 1.0
            val_metrics = {'mse_loss': 1.0, 'total_loss': 1.0}

        # Scheduler step
        if enhanced_scheduler is not None:
            if isinstance(enhanced_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                enhanced_scheduler.step(val_loss)
            else:
                enhanced_scheduler.step()

        # Best model checkpoint
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': enhanced_model.state_dict(),
                'optimizer_state_dict': enhanced_optimizer.state_dict(),
                'scheduler_state_dict': enhanced_scheduler.state_dict() if enhanced_scheduler else None,
                'best_val': best_val,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, 'checkpoints/best_enhanced_model.pt')

        # W&B logging
        current_lr = enhanced_optimizer.param_groups[0]['lr']
        wandb.log({
            'epoch': epoch, 'lr': current_lr,
            'train/total': train_loss,
            'train/mse': train_metrics.get('mse_loss', 0),
            'train/continuity': train_metrics.get('continuity_loss', 0),
            'train/momentum': train_metrics.get('momentum_loss', 0),
            'train/bc': train_metrics.get('bc_loss', 0),
            'train/turbulence_production': train_metrics.get('turbulence_production_loss', 0),
            'train/turbulence_dissipation': train_metrics.get('turbulence_dissipation_loss', 0),
            'train/smoothness': train_metrics.get('smoothness_loss', 0),
            'train/wall_function': train_metrics.get('wall_function_loss', 0),
            'val/total': val_loss,
            'val/mse': val_metrics.get('mse_loss', 0),
            'val/continuity': val_metrics.get('continuity_loss', 0),
            'val/momentum': val_metrics.get('momentum_loss', 0),
            'val/bc': val_metrics.get('bc_loss', 0),
        }, step=epoch)

        print(f"Epoch {epoch:3d} | LR: {current_lr:.2e} | "
              f"Train: {train_loss:.4f} "
              f"(mse: {train_metrics.get('mse_loss', 0):.4f}, "
              f"cont: {train_metrics.get('continuity_loss', 0):.2e}, "
              f"mom: {train_metrics.get('momentum_loss', 0):.2e}, "
              f"bc: {train_metrics.get('bc_loss', 0):.2e}, "
              f"turb_prod: {train_metrics.get('turbulence_production_loss', 0):.2e}) | "
              f"Val: {val_loss:.4f} {'[BEST]' if is_best else ''}")

        history.append({
            'epoch': epoch, 'lr': current_lr,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_metrics': train_metrics, 'val_metrics': val_metrics,
            'is_best': is_best,
        })

    print(f"\nTraining complete. Best Val Loss: {best_val:.4f}")
    wandb.finish()

    # --- Evaluate ---
    val_edges = data_bundle.val_graphs
    train_edges = data_bundle.train_graphs
    x_scaler = data_bundle.x_scaler
    y_scaler = data_bundle.y_scaler

    d_eval = val_edges[0] if len(val_edges) > 0 else (train_edges[0] if len(train_edges) > 0 else None)
    if d_eval is not None:
        dm_eval, y_pred = _predict_one_local(d_eval, enhanced_model, device, x_scaler, y_scaler, scfg.amp)
        surf_mask, vol_mask = _surface_volume_masks_from_orig(d_eval)
        names = ['u', 'v', 'p_over_rho', 'nu_t']
        print('[MSE | ALL   ]', {n: f'{v:.4e}' for n, v in zip(names, mse_per_channel(y_pred, dm_eval.y, None))})
        print('[MSE | SURF  ]', {n: f'{v:.4e}' for n, v in zip(names, mse_per_channel(y_pred, dm_eval.y, surf_mask))})
        print('[MSE | VOLUME]', {n: f'{v:.4e}' for n, v in zip(names, mse_per_channel(y_pred, dm_eval.y, vol_mask))})

    # --- Visualize ---
    d_vis = val_edges[0] if len(val_edges) > 0 else (train_edges[0] if len(train_edges) > 0 else None)
    if d_vis is not None:
        dm_norm = Data(**{k: v for k, v in d_vis})
        if dm_norm.x.size(1) == 5 or not getattr(dm_norm, 'pos2_appended', False):
            dm_norm = with_pos2(dm_norm)
        dm_norm = ensure_edge_features(dm_norm, want_dim=5)
        dm_viz = Data(**{k: v for k, v in dm_norm})
        dm_viz.x = x_scaler.transform(dm_norm.x)
        dm_viz.y = y_scaler.transform(dm_norm.y)
        dm_run = dm_viz.to(device)
        if hasattr(dm_run, 'x'):
            dm_run.x = dm_run.x.float()
        if hasattr(dm_run, 'edge_attr'):
            dm_run.edge_attr = dm_run.edge_attr.float()
        enhanced_model.eval()
        y_pred_viz = enhanced_model(dm_run).detach().cpu()
        plot_pred_vs_gt(dm_viz, y_pred_viz, y_scaler, x_scaler,
                        channel=2, show_mesh=False, denormalize=True)


if __name__ == '__main__':
    main()
