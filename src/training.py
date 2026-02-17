"""Training loop utilities: epoch routines, loss computation, LR scheduler, W&B init.

Split from training_common.py — training-specific code only.
Config lives in config.py, data loading in data.py.
"""

from __future__ import annotations

import os
import random
import contextlib
from typing import Optional, Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import wandb


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_lr(optim):
    return optim.param_groups[0].get('lr', None)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

_mse_loss_fn = nn.MSELoss()


def compute_loss_with_physics(predictions, targets, data, loss_fn=None, *, step: int | None = None):
    """Compute loss using physics-informed loss function or fallback to MSE.
    Returns a differentiable scalar loss tensor and a dict of float metrics.
    """
    if loss_fn is not None:
        try:
            loss_dict = loss_fn(predictions, targets, data=data, step=step)

            total_loss = loss_dict.get('total_loss')
            if not isinstance(total_loss, torch.Tensor):
                total_loss = torch.as_tensor(total_loss, dtype=predictions.dtype, device=predictions.device)

            log_dict = {}
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    try:
                        log_dict[k] = float(v.detach().item())
                    except Exception:
                        log_dict[k] = float(v.detach().mean().item())
                else:
                    log_dict[k] = float(v)

            return total_loss, log_dict
        except Exception as e:
            print(f"Warning: Physics loss failed ({e}), falling back to MSE")
            mse_loss = _mse_loss_fn(predictions, targets)
            return mse_loss, {
                'mse_loss': float(mse_loss.detach().item()),
                'continuity_loss': 0.0,
                'momentum_loss': 0.0,
                'bc_wall_loss': 0.0,
                'bc_inlet_loss': 0.0,
                'bc_outlet_loss': 0.0,
                'bc_farfield_loss': 0.0,
                'bc_loss': 0.0,
                'total_loss': float(mse_loss.detach().item())
            }
    else:
        mse_loss = _mse_loss_fn(predictions, targets)
        return mse_loss, {
            'mse_loss': float(mse_loss.detach().item()),
            'bc_wall_loss': 0.0,
            'bc_inlet_loss': 0.0,
            'bc_outlet_loss': 0.0,
            'bc_farfield_loss': 0.0,
            'bc_loss': 0.0,
            'total_loss': float(mse_loss.detach().item())
        }


# ---------------------------------------------------------------------------
# Epoch routines
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_epoch(loader, model, device, *, amp_enabled: bool = False, desc: str = 'val', loss_fn=None):
    model.eval()
    total_losses, mse_losses, continuity_losses, momentum_losses = [], [], [], []
    bc_losses = []
    bc_wall_losses = []
    bc_inlet_losses = []
    bc_outlet_losses = []
    bc_farfield_losses = []
    cont_w_used_hist, mom_w_used_hist = [], []

    if loader is None or (isinstance(loader, list) and len(loader) == 0):
        return float('nan'), {}

    steps = len(loader)
    pbar = tqdm(total=steps, desc=desc, leave=False)

    for batch in loader:
        try:
            if batch is None:
                pbar.update(1)
                continue

            b = batch.to(device)
            with (autocast(enabled=(amp_enabled and torch.cuda.is_available()))
                  if torch.cuda.is_available() else contextlib.nullcontext()):
                out = model(b)
                _, loss_dict = compute_loss_with_physics(out, b.y, b, loss_fn=loss_fn, step=None)

            total_losses.append(loss_dict['total_loss'])
            mse_losses.append(loss_dict['mse_loss'])
            continuity_losses.append(loss_dict.get('continuity_loss', 0.0))
            momentum_losses.append(loss_dict.get('momentum_loss', 0.0))
            bc_losses.append(loss_dict.get('bc_loss', 0.0))
            bc_wall_losses.append(loss_dict.get('bc_wall_loss', 0.0))
            bc_inlet_losses.append(loss_dict.get('bc_inlet_loss', 0.0))
            bc_outlet_losses.append(loss_dict.get('bc_outlet_loss', 0.0))
            bc_farfield_losses.append(loss_dict.get('bc_farfield_loss', 0.0))
            if 'cont_weight_used' in loss_dict:
                cont_w_used_hist.append(loss_dict['cont_weight_used'])
            if 'mom_weight_used' in loss_dict:
                mom_w_used_hist.append(loss_dict['mom_weight_used'])

            postfix = {"total": f"{loss_dict['total_loss']:.4e}"}
            if 'continuity_loss' in loss_dict:
                postfix["cont"] = f"{loss_dict['continuity_loss']:.4e}"
            if 'momentum_loss' in loss_dict:
                postfix["momentum"] = f"{loss_dict['momentum_loss']:.4e}"
            if 'bc_loss' in loss_dict:
                postfix["bc"] = f"{loss_dict['bc_loss']:.4e}"
            if 'bc_wall_loss' in loss_dict:
                postfix["bc_wall"] = f"{loss_dict['bc_wall_loss']:.4e}"
            if 'bc_inlet_loss' in loss_dict:
                postfix["bc_inlet"] = f"{loss_dict['bc_inlet_loss']:.4e}"
            if 'bc_outlet_loss' in loss_dict:
                postfix["bc_out"] = f"{loss_dict['bc_outlet_loss']:.4e}"
            if 'bc_farfield_loss' in loss_dict:
                postfix["bc_far"] = f"{loss_dict['bc_farfield_loss']:.4e}"
            pbar.set_postfix(postfix)
        finally:
            pbar.update(1)

    pbar.close()

    avg_losses = {
        'total_loss': np.mean(total_losses) if total_losses else float('nan'),
        'mse_loss': np.mean(mse_losses) if mse_losses else float('nan'),
        'continuity_loss': np.mean(continuity_losses) if continuity_losses else float('nan'),
        'momentum_loss': np.mean(momentum_losses) if momentum_losses else float('nan'),
        'bc_loss': np.mean(bc_losses) if bc_losses else float('nan'),
        'bc_wall_loss': np.mean(bc_wall_losses) if bc_wall_losses else float('nan'),
        'bc_inlet_loss': np.mean(bc_inlet_losses) if bc_inlet_losses else float('nan'),
        'bc_outlet_loss': np.mean(bc_outlet_losses) if bc_outlet_losses else float('nan'),
        'bc_farfield_loss': np.mean(bc_farfield_losses) if bc_farfield_losses else float('nan'),
    }
    if cont_w_used_hist:
        avg_losses['cont_weight_used'] = float(np.mean(cont_w_used_hist))
    if mom_w_used_hist:
        avg_losses['mom_weight_used'] = float(np.mean(mom_w_used_hist))
    return avg_losses['total_loss'], avg_losses


def train_epoch(loader, model, optim, device, scaler, *,
                amp_enabled: bool = False,
                desc: str = 'train',
                loss_fn=None,
                global_step_start: int = 0,
                scheduler=None,
                scheduler_step_mode: str = "epoch",
                log_every_n_steps: int = -1):
    model.train()
    total_losses, mse_losses, continuity_losses, momentum_losses = [], [], [], []
    bc_losses = []
    bc_wall_losses = []
    bc_inlet_losses = []
    bc_outlet_losses = []
    bc_farfield_losses = []
    cont_w_used_hist, mom_w_used_hist = [], []

    global_step = global_step_start
    steps = len(loader)
    pbar = tqdm(total=steps, desc=desc, leave=False)

    for batch_idx, batch in enumerate(loader):
        try:
            if batch is None:
                pbar.update(1)
                global_step += 1
                continue

            b = batch.to(device)
            optim.zero_grad(set_to_none=True)

            use_scaler = (scaler is not None) and getattr(scaler, "is_enabled", lambda: False)()

            if use_scaler:
                with autocast(enabled=torch.cuda.is_available()):
                    out = model(b)
                    loss, loss_dict = compute_loss_with_physics(out, b.y, b, loss_fn=loss_fn, step=global_step)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                with contextlib.nullcontext():
                    out = model(b)
                    loss, loss_dict = compute_loss_with_physics(out, b.y, b, loss_fn=loss_fn, step=global_step)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            if scheduler is not None and scheduler_step_mode == "step":
                try:
                    scheduler.step()
                except TypeError:
                    pass

            total_losses.append(loss_dict['total_loss'])
            mse_losses.append(loss_dict['mse_loss'])
            continuity_losses.append(loss_dict.get('continuity_loss', 0.0))
            momentum_losses.append(loss_dict.get('momentum_loss', 0.0))
            bc_losses.append(loss_dict.get('bc_loss', 0.0))
            bc_wall_losses.append(loss_dict.get('bc_wall_loss', 0.0))
            bc_inlet_losses.append(loss_dict.get('bc_inlet_loss', 0.0))
            bc_outlet_losses.append(loss_dict.get('bc_outlet_loss', 0.0))
            bc_farfield_losses.append(loss_dict.get('bc_farfield_loss', 0.0))
            if 'cont_weight_used' in loss_dict:
                cont_w_used_hist.append(loss_dict['cont_weight_used'])
            if 'mom_weight_used' in loss_dict:
                mom_w_used_hist.append(loss_dict['mom_weight_used'])

            if log_every_n_steps > 0 and (batch_idx % max(1, log_every_n_steps)) == 0:
                log_payload = {
                    "step": global_step,
                    "train/total": loss_dict['total_loss'],
                    "train/mse": loss_dict['mse_loss'],
                    "train/continuity": loss_dict.get('continuity_loss', 0.0),
                    "train/momentum": loss_dict.get('momentum_loss', 0.0),
                    "train/bc": loss_dict.get('bc_loss', 0.0),
                    "train/bc_wall": loss_dict.get('bc_wall_loss', 0.0),
                    "train/bc_inlet": loss_dict.get('bc_inlet_loss', 0.0),
                    "train/bc_outlet": loss_dict.get('bc_outlet_loss', 0.0),
                    "train/bc_farfield": loss_dict.get('bc_farfield_loss', 0.0),
                }
                if 'cont_weight_used' in loss_dict:
                    log_payload["weight/cont_used"] = loss_dict['cont_weight_used']
                if 'mom_weight_used' in loss_dict:
                    log_payload["weight/mom_used"] = loss_dict['mom_weight_used']
                lr_now = get_lr(optim)
                if lr_now is not None:
                    log_payload["lr"] = lr_now
                wandb.log(log_payload, step=global_step, commit=False)

            postfix = {"total": f"{loss_dict['total_loss']:.4e}",
                       "lr": f"{get_lr(optim):.2e}" if get_lr(optim) is not None else "n/a"}
            if 'continuity_loss' in loss_dict:
                postfix["cont"] = f"{loss_dict['continuity_loss']:.4e}"
            if 'momentum_loss' in loss_dict:
                postfix["momentum"] = f"{loss_dict['momentum_loss']:.4e}"
            if 'bc_loss' in loss_dict:
                postfix["bc"] = f"{loss_dict['bc_loss']:.4e}"
            if 'bc_wall_loss' in loss_dict:
                postfix["bc_wall"] = f"{loss_dict['bc_wall_loss']:.4e}"
            if 'bc_inlet_loss' in loss_dict:
                postfix["bc_inlet"] = f"{loss_dict['bc_inlet_loss']:.4e}"
            if 'bc_outlet_loss' in loss_dict:
                postfix["bc_out"] = f"{loss_dict['bc_outlet_loss']:.4e}"
            if 'bc_farfield_loss' in loss_dict:
                postfix["bc_far"] = f"{loss_dict['bc_farfield_loss']:.4e}"
            pbar.set_postfix(postfix)
        finally:
            pbar.update(1)
            global_step += 1

    pbar.close()

    avg_losses = {
        'total_loss': np.mean(total_losses) if total_losses else float('nan'),
        'mse_loss': np.mean(mse_losses) if mse_losses else float('nan'),
        'continuity_loss': np.mean(continuity_losses) if continuity_losses else float('nan'),
        'momentum_loss': np.mean(momentum_losses) if momentum_losses else float('nan'),
        'bc_loss': np.mean(bc_losses) if bc_losses else float('nan'),
        'bc_wall_loss': np.mean(bc_wall_losses) if bc_wall_losses else float('nan'),
        'bc_inlet_loss': np.mean(bc_inlet_losses) if bc_inlet_losses else float('nan'),
        'bc_outlet_loss': np.mean(bc_outlet_losses) if bc_outlet_losses else float('nan'),
        'bc_farfield_loss': np.mean(bc_farfield_losses) if bc_farfield_losses else float('nan'),
    }
    if cont_w_used_hist:
        avg_losses['cont_weight_used'] = float(np.mean(cont_w_used_hist))
    if mom_w_used_hist:
        avg_losses['mom_weight_used'] = float(np.mean(mom_w_used_hist))

    return avg_losses['total_loss'], avg_losses, global_step


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------

def create_lr_scheduler(optimizer, config):
    """Create an LR scheduler based on config.lr_scheduler."""
    if config.lr_scheduler is None:
        print("No learning rate scheduler (constant LR)")
        return None

    elif config.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.cosine_T_max, eta_min=config.cosine_eta_min)
        print(f"LR scheduler: CosineAnnealingLR (T_max={config.cosine_T_max}, eta_min={config.cosine_eta_min})")
        return scheduler

    elif config.lr_scheduler == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.wr_T_0, T_mult=config.wr_T_mult, eta_min=config.wr_eta_min)
        print(f"LR scheduler: CosineAnnealingWarmRestarts (T_0={config.wr_T_0})")
        return scheduler

    elif config.lr_scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.rop_factor,
            patience=config.rop_patience, min_lr=config.rop_min_lr)
        print(f"LR scheduler: ReduceLROnPlateau (factor={config.rop_factor}, patience={config.rop_patience})")
        return scheduler

    else:
        print(f"Unknown scheduler: {config.lr_scheduler}, using None")
        return None


# ---------------------------------------------------------------------------
# W&B initialization
# ---------------------------------------------------------------------------

def init_wandb(scfg, loss_fn=None):
    """Initialize W&B run and configure epoch-only logging."""
    enable_wandb = bool(getattr(scfg, "enable_wandb", True))
    wandb_env = os.getenv("AIRFRANS_ENABLE_WANDB", "").strip().lower()
    if wandb_env in {"0", "false", "f", "off", "no"}:
        enable_wandb = False
    elif wandb_env in {"1", "true", "t", "on", "yes"}:
        enable_wandb = True

    if not enable_wandb:
        print("[wandb] disabled (use --enable-wandb or AIRFRANS_ENABLE_WANDB=1)")

    wandb_init_kwargs: dict[str, Any] = dict(
        project=getattr(scfg, "wandb_project", "airfrans-gnn"),
        name=getattr(scfg, "wandb_run_name", getattr(scfg, "wandb_name", None)),
        tags=getattr(scfg, "wandb_tags", None),
        mode="disabled" if not enable_wandb else getattr(scfg, "wandb_mode", "online"),
        settings=wandb.Settings(start_method="thread"),
        config={
            "epochs": getattr(scfg, "epochs", None),
            "batch_size": getattr(scfg, "batch_size", None),
            "lr": getattr(scfg, "lr", None),
            "optimizer": getattr(scfg, "optimizer", "adam"),
            "scheduler": getattr(scfg, "scheduler", None),
            "amp": getattr(scfg, "amp", False),
        }
    )
    if loss_fn is not None:
        config = cast(dict[str, Any], wandb_init_kwargs["config"])
        config.update({
            "cont_w0": getattr(loss_fn, "cont_w0", None),
            "cont_w_target": getattr(loss_fn, "cont_w_target", None),
            "mom_w0": getattr(loss_fn, "mom_w0", None),
            "mom_w_target": getattr(loss_fn, "mom_w_target", None),
            "ramp_steps": getattr(loss_fn, "curr_steps", None),
            "ramp_start_step": getattr(loss_fn, "ramp_start_step", 0),
        })

    wandb_run = wandb.init(**cast(dict[str, Any], wandb_init_kwargs))

    # Epoch-only logging: swallow per-step logs (commit=False)
    if getattr(scfg, "log_epoch_only", True):
        try:
            _wandb_orig_log = wandb.log

            def _log_epoch_only(data=None, step=None, commit=None, *args, **kwargs):
                if commit is None:
                    commit = True
                if commit is False:
                    return
                if data is None:
                    return
                return _wandb_orig_log(cast(dict[str, Any], data), step=step, commit=commit, *args, **kwargs)

            wandb.log = _log_epoch_only
        except Exception:
            pass

    return wandb_run
