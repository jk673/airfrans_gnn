"""Training epoch routines and loss computation for AirfRANS GNN."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_lr(optim):
    return optim.param_groups[0].get('lr', None)


def _unwrap_model(model):
    """Return the underlying model, unwrapping DDP if needed."""
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

_mse_loss_fn = nn.MSELoss()

_LOSS_KEYS = (
    'total_loss', 'mse_loss', 'continuity_loss', 'momentum_loss',
    'bc_loss', 'bc_wall_loss', 'bc_inlet_loss', 'bc_outlet_loss', 'bc_farfield_loss',
)
_WEIGHT_KEYS = ('cont_weight_used', 'mom_weight_used')


def compute_loss_with_physics(predictions, targets, data, loss_fn=None, *, step=None):
    """Compute loss using physics-informed loss function or fallback to MSE."""
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
            import traceback
            print(f"Warning: Physics loss failed ({e}), falling back to MSE")
            traceback.print_exc()

    mse_loss = _mse_loss_fn(predictions, targets)
    return mse_loss, {
        'mse_loss': float(mse_loss.detach().item()),
        'total_loss': float(mse_loss.detach().item()),
        **{k: 0.0 for k in _LOSS_KEYS if k not in ('total_loss', 'mse_loss')},
    }


# ---------------------------------------------------------------------------
# Loss accumulation helpers
# ---------------------------------------------------------------------------

def _new_accumulators():
    acc = {k: [] for k in _LOSS_KEYS}
    for wk in _WEIGHT_KEYS:
        acc[wk] = []
    return acc


def _collect(acc, loss_dict):
    for k in _LOSS_KEYS:
        acc[k].append(loss_dict.get(k, 0.0))
    for wk in _WEIGHT_KEYS:
        if wk in loss_dict:
            acc[wk].append(loss_dict[wk])


def _average(acc):
    result = {}
    for k in _LOSS_KEYS:
        vals = acc[k]
        result[k] = float(np.mean(vals)) if vals else float('nan')
    for wk in _WEIGHT_KEYS:
        if acc[wk]:
            result[wk] = float(np.mean(acc[wk]))
    return result


def _postfix(loss_dict, lr=None):
    pf = {"total": f"{loss_dict['total_loss']:.4e}"}
    for short, key in [("cont", "continuity_loss"), ("mom", "momentum_loss"), ("bc", "bc_loss")]:
        if key in loss_dict:
            pf[short] = f"{loss_dict[key]:.4e}"
    if lr is not None:
        pf["lr"] = f"{lr:.2e}"
    return pf


# ---------------------------------------------------------------------------
# Epoch routines
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_epoch(loader, model, device, *, amp_enabled=False, desc='val', loss_fn=None):
    model.eval()
    if loader is None or (isinstance(loader, list) and len(loader) == 0):
        return float('nan'), {}

    acc = _new_accumulators()
    pbar = tqdm(total=len(loader), desc=desc, leave=False)

    for batch in loader:
        try:
            if batch is None:
                continue
            b = batch.to(device)
            with torch.autocast(device_type="cuda", enabled=(amp_enabled and torch.cuda.is_available()),
                                dtype=torch.bfloat16):
                out = model(b)
                _, loss_dict = compute_loss_with_physics(out, b.y, b, loss_fn=None)
            _collect(acc, loss_dict)
            pbar.set_postfix(_postfix(loss_dict))
        finally:
            pbar.update(1)

    pbar.close()
    avg = _average(acc)
    return avg['total_loss'], avg


def train_epoch(loader, model, optim, device, scaler, *,
                amp_enabled=False, desc='train', loss_fn=None,
                global_step_start=0, scheduler=None,
                scheduler_step_mode="epoch"):
    model.train()
    acc = _new_accumulators()
    global_step = global_step_start
    pbar = tqdm(total=len(loader), desc=desc, leave=False)

    for batch in loader:
        try:
            if batch is None:
                global_step += 1
                continue

            b = batch.to(device)
            optim.zero_grad(set_to_none=True)

            use_scaler = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()

            if use_scaler:
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    out = model(b)
                    loss, loss_dict = compute_loss_with_physics(out, b.y, b, loss_fn=loss_fn, step=global_step)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=torch.bfloat16):
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

            _collect(acc, loss_dict)
            pbar.set_postfix(_postfix(loss_dict, get_lr(optim)))
        finally:
            pbar.update(1)
            global_step += 1

    pbar.close()
    avg = _average(acc)
    return avg['total_loss'], avg, global_step
