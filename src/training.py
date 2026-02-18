"""Training utilities, epoch routines, loss computation, LR scheduler, W&B init, and full training loop.

Config lives in config.py, data loading in data.py.
"""

from __future__ import annotations

import math
import os
import random
import contextlib
from pathlib import Path
from typing import Optional, Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import wandb

from src.ddp_utils import _unwrap_model


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


# ---------------------------------------------------------------------------
# Loss signal tracking & full training loop (merged from train_loop.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOSS_SIGNALS_DIR = PROJECT_ROOT / "experiments" / "loss_signals"
LOSS_SIGNALS_PLOT_PATH = LOSS_SIGNALS_DIR / "loss_signals_epoch_vs_loss.png"

LOSS_SIGNAL_KEYS = (
    "total_loss",
    "mse_loss",
    "continuity_loss",
    "momentum_loss",
    "bc_loss",
    "learning_rate",
)
LOSS_SIGNAL_LABELS = {
    "total_loss": "Total Loss",
    "mse_loss": "MSE Loss",
    "continuity_loss": "Continuity Loss",
    "momentum_loss": "Momentum Loss",
    "bc_loss": "Boundary Condition Loss",
    "learning_rate": "Learning Rate",
}


def _safe_float_loss_value(v) -> float:
    if isinstance(v, torch.Tensor):
        if v.numel() != 1:
            return float("nan")
        v = v.item()
    try:
        fv = float(v)
    except Exception:
        return float("nan")
    if isinstance(fv, float) and not (fv == fv):
        return float("nan")
    return fv


def _append_loss_history(loss_history: dict, loss_type: str, logs: dict, step: int):
    values = {}
    for key in LOSS_SIGNAL_KEYS:
        values[key] = _safe_float_loss_value(logs.get(key, float("nan")))
    loss_history[loss_type]["epoch"].append(step)
    for key, value in values.items():
        loss_history[loss_type][key].append(value)


def _save_loss_signal_plots(loss_history: dict) -> None:
    LOSS_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    all_epochs = loss_history["train"]["epoch"]
    if not all_epochs:
        return

    def _finite_series(values):
        arr = np.array(values, dtype=float)
        return np.isfinite(arr), arr

    active_keys = []
    for key in LOSS_SIGNAL_KEYS:
        train_mask, train_vals = _finite_series(loss_history["train"][key])
        val_mask, val_vals = _finite_series(loss_history["val"][key])
        if np.any(train_mask) or np.any(val_mask):
            active_keys.append(key)

    if not active_keys:
        return

    n_cols = 3
    n_rows = math.ceil(len(active_keys) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    epochs = np.array(all_epochs, dtype=int)

    for idx, key in enumerate(active_keys):
        ax = axes[idx // n_cols][idx % n_cols]
        if key == "learning_rate":
            # LR is recorded only in train history (single line, no train/val split)
            lr_mask, lr_vals = _finite_series(loss_history["train"][key])
            if np.any(lr_mask):
                ax.plot(
                    epochs[lr_mask],
                    lr_vals[lr_mask],
                    label="lr",
                    marker="o",
                    linewidth=1.2,
                    color="tab:purple",
                )
            ax.set_ylabel("LR")
        else:
            train_mask, train_vals = _finite_series(loss_history["train"][key])
            val_mask, val_vals = _finite_series(loss_history["val"][key])
            if np.any(train_mask):
                ax.plot(
                    epochs[train_mask],
                    train_vals[train_mask],
                    label="train",
                    marker="o",
                    linewidth=1.2,
                )
            if np.any(val_mask):
                ax.plot(
                    epochs[val_mask],
                    val_vals[val_mask],
                    label="val",
                    marker="x",
                    linewidth=1.2,
                )
            ax.set_ylabel("Loss")
            if key in {"continuity_loss", "bc_loss"}:
                ax.set_ylim(0.0, 0.1)
            else:
                ax.set_ylim(0, 1)
        ax.set_title(LOSS_SIGNAL_LABELS.get(key, key))
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend()

    for idx in range(len(active_keys), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.tight_layout()
    fig.suptitle("Loss History", y=1.02)
    plt.savefig(LOSS_SIGNALS_PLOT_PATH, dpi=200)
    plt.close(fig)


def train_with_scheduler(model, optim, scheduler, train_loader, val_loader,
                         scfg, device, physics_loss_fn,
                         *, train_sampler=None, is_main=True):
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
    loss_history = {
        "train": {"epoch": []},
        "val": {"epoch": []},
    }
    for key in LOSS_SIGNAL_KEYS:
        loss_history["train"][key] = []
        loss_history["val"][key] = []

    for epoch in range(EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

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

        if is_main:
            train_logs["learning_rate"] = get_lr(optim) or float("nan")
            _append_loss_history(loss_history, "train", train_logs, epoch + 1)
            _append_loss_history(loss_history, "val", val_logs, epoch + 1)
            _save_loss_signal_plots(loss_history)

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

        # Track best val on all ranks (needed for curriculum/scheduler consistency)
        is_best = val_total < best_val
        if is_best:
            best_val = val_total
            best_epoch = epoch

        # Checkpointing (rank 0 only)
        if is_main:
            os.makedirs(ckpt_dir, exist_ok=True)
            model_state = _unwrap_model(model).state_dict()

            if is_best:
                best_path = os.path.join(ckpt_dir, "best.pt")
                torch.save({
                    "model": model_state,
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
                    "model": model_state,
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

    if is_main:
        wandb.finish()
    return {
        'best_val': best_val,
        'best_epoch': best_epoch,
        'final_train_loss': final_train_total,
        'final_val_loss': final_val_total,
        'artifacts_uploaded': artifact_history['total_artifacts'],
    }
