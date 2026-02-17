"""Training loop with checkpointing, W&B artifacts, and loss signal tracking."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from src.training_common import train_epoch, run_epoch, get_lr
from src.ddp_utils import _unwrap_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOSS_SIGNALS_DIR = PROJECT_ROOT / "experiments" / "loss_signals"
LOSS_SIGNALS_PLOT_PATH = LOSS_SIGNALS_DIR / "loss_signals_epoch_vs_loss.png"

LOSS_SIGNAL_KEYS = (
    "total_loss",
    "mse_loss",
    "continuity_loss",
    "momentum_loss",
    "bc_loss",
    "bc_wall_loss",
    "bc_inlet_loss",
    "bc_outlet_loss",
    "bc_farfield_loss",
)
LOSS_SIGNAL_LABELS = {
    "total_loss": "Total Loss",
    "mse_loss": "MSE Loss",
    "continuity_loss": "Continuity Loss",
    "momentum_loss": "Momentum Loss",
    "bc_loss": "Boundary Condition Loss",
    "bc_wall_loss": "BC Wall Loss",
    "bc_inlet_loss": "BC Inlet Loss",
    "bc_outlet_loss": "BC Outlet Loss",
    "bc_farfield_loss": "BC Farfield Loss",
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
        ax.set_title(LOSS_SIGNAL_LABELS.get(key, key))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
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
