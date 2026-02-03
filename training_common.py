"""
training_common.py — Shared infrastructure for AirfRANS GNN training scripts.

Extracted from 01_trainer.ipynb, 02_optuna_training.ipynb, 02_trainer_multi_scale.ipynb.
All functions take explicit parameters instead of referencing globals.
"""

from __future__ import annotations

import os
import glob
import random
import contextlib
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric.datasets import AirfRANS
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import BaseTransform
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import wandb

from airfrans_utils import prepare_airfrans_graph_for_physics, build_bc_masks_airfrans
from utils import prep_graph, validate_edges, _prep_graph_for_norm


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


def collate_pyg(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return Batch.from_data_list(batch)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SmokeCfg:
    seed: int = 42
    task: str = 'scarce'
    root: str = 'Dataset'
    limit_train: int = 180
    limit_val: int = 20

    # training
    batch_size: int = 2
    epochs: int = 100
    hidden: int = 128
    layers: int = 14
    lr: float = 4e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    amp: bool = False

    # lr scheduler: 'cosine', 'cosine_warm_restarts', 'reduce_on_plateau', or None
    lr_scheduler: str = 'cosine'
    cosine_T_max: int = 80
    cosine_eta_min: float = 1e-6
    wr_T_0: int = 10
    wr_T_mult: int = 1
    wr_eta_min: float = 1e-6
    rop_factor: float = 0.5
    rop_patience: int = 5
    rop_min_lr: float = 1e-6

    # Physics-Informed Loss Configuration
    ramp_start_epoch: int = 40
    ramp_epochs: int = 60
    ramp_mode: str = 'linear'

    data_loss_weight: float = 1.0
    continuity_loss_weight: float = 0.05
    continuity_target_weight: float = 0.20
    momentum_loss_weight: float = 0.05
    momentum_target_weight: float = 0.20
    bc_loss_weight: float = 0.1

    chord_length: float = 1.0
    nu_molecular: float = 1.5e-5
    dynamic_uref_from_data: bool = True
    dynamic_re_from_data: bool = True
    uinf_from: str = 'inlet'

    use_huber_for_physics: bool = True
    huber_delta: float = 0.05
    use_perimeter_norm_for_div: bool = True
    div_area_floor_factor: float = 0.25
    div_min_degree: int = 2

    physics_debug: bool = False
    physics_debug_level: int = 1
    physics_debug_every: int = 50

    # Global Context & Attention
    use_global_tokens: bool = True
    num_global_tokens: int = 2
    attention_heads: int = 2
    attention_layers: int = 2
    attention_dropout: float = 0.0
    use_cross_attention: bool = True
    global_pooling_type: str = 'attention'
    positional_encoding: bool = False
    pos_encoding_max_len: int = 50000
    use_residual_attention: bool = True
    attention_normalization: str = 'layer'
    temperature_scaling: bool = False
    attention_bias: bool = False

    # W&B Artifact management
    use_wandb_artifacts: bool = False
    artifact_save_best_only: bool = True
    artifact_save_interval: int = 50

    # Checkpoint management
    ckpt_dir: str = "checkpoints"
    ckpt_interval: int = 5

    # W&B settings
    wandb_project: str = "airfrans-gnn"
    wandb_mode: str = "online"
    log_every_n_steps: int = -1
    log_epoch_only: bool = True


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, t: torch.Tensor):
        self.mean = t.mean(dim=0)
        self.std = t.std(dim=0).clamp_min(1e-8)
        return self

    def transform(self, t: torch.Tensor):
        return (t - self.mean) / self.std

    def inverse(self, t: torch.Tensor):
        return t * self.std + self.mean


# ---------------------------------------------------------------------------
# NormalizedDataset
# ---------------------------------------------------------------------------

class NormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, x_scaler, y_scaler):
        self.graphs = graphs
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int):
        d = self.graphs[idx]
        dm = Data(**{k: v for k, v in d})
        dm.x = self.x_scaler.transform(d.x)
        if hasattr(d, 'y') and d.y is not None:
            dm.y = self.y_scaler.transform(d.y)
        else:
            dm.y = d.y

        dm.has_norm = True

        # Ensure edge_attr_dxdy is present (needed for physics loss)
        if hasattr(d, 'edge_attr_dxdy'):
            dm.edge_attr_dxdy = d.edge_attr_dxdy
        elif hasattr(d, 'edge_attr'):
            if d.edge_attr.shape[1] >= 2:
                dm.edge_attr_dxdy = d.edge_attr[:, -2:]
            dm.edge_attr = d.edge_attr

        # Build BC masks
        dm = build_bc_masks_airfrans(dm)

        # Fallback BC masks if build_bc_masks_airfrans didn't set individual masks
        if not hasattr(dm, 'bc_mask_dict'):
            num_nodes = dm.x.size(0)
            x_orig = d.x

            if x_orig.size(1) > 2:
                wall_dist = x_orig[:, 2]
                dm.is_wall = (wall_dist < 1e-6)
            else:
                dm.is_wall = torch.zeros(num_nodes, dtype=torch.bool)

            if not hasattr(dm, 'is_inlet'):
                dm.is_inlet = torch.zeros(num_nodes, dtype=torch.bool)
            if not hasattr(dm, 'is_outlet'):
                dm.is_outlet = torch.zeros(num_nodes, dtype=torch.bool)
            if not hasattr(dm, 'is_farfield'):
                dm.is_farfield = torch.zeros(num_nodes, dtype=torch.bool)

            if hasattr(dm, 'pos'):
                x_coords = dm.pos[:, 0]
                y_coords = dm.pos[:, 1]
                dm.is_inlet = (x_coords < -1.0) & ~dm.is_wall
                dm.is_outlet = (x_coords > 2.0) & ~dm.is_wall
                dm.is_farfield = (torch.abs(y_coords) > 1.0) & ~dm.is_wall & ~dm.is_inlet & ~dm.is_outlet
        else:
            for bc_type, mask in dm.bc_mask_dict.items():
                setattr(dm, f'is_{bc_type}', mask)

        return dm


# ---------------------------------------------------------------------------
# PreparePhysics transform
# ---------------------------------------------------------------------------

class PreparePhysics(BaseTransform):
    def forward(self, data):
        return prepare_airfrans_graph_for_physics(data, verbose=False)


# ---------------------------------------------------------------------------
# DataBundle — holds loaders, scalers, graphs
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: object  # DataLoader or empty list
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    train_graphs: list
    val_graphs: list
    train_norm: NormalizedDataset
    val_norm: object  # NormalizedDataset or empty list


# ---------------------------------------------------------------------------
# load_and_prepare_data
# ---------------------------------------------------------------------------

def load_and_prepare_data(scfg: SmokeCfg) -> DataBundle:
    """Load AirfRANS data, build edges, normalize, and return DataBundle."""
    prebuilt_root = f'prebuilt_edges/{scfg.task}'
    prebuilt_train_dir = f"{prebuilt_root}/train"
    prebuilt_test_dir = f"{prebuilt_root}/test"

    # Load prebuilt edge graphs
    train_edge_files = sorted(glob.glob(os.path.join(prebuilt_train_dir, 'graph_*.pt')))
    val_edge_files = sorted(glob.glob(os.path.join(prebuilt_test_dir, 'graph_*.pt')))
    print(f"[prebuilt] found: {len(train_edge_files)} train and {len(val_edge_files)} val graphs under {prebuilt_root}")

    train_edges = []
    for p in train_edge_files:
        d = torch.load(p, map_location='cpu', weights_only=False)
        if not isinstance(d, Data):
            d = Data(**d)
        train_edges.append(prep_graph(d))

    val_edges = []
    for p in val_edge_files:
        d = torch.load(p, map_location='cpu', weights_only=False)
        if not isinstance(d, Data):
            d = Data(**d)
        val_edges.append(prep_graph(d))

    if len(train_edges) > 0:
        print(f"Graphs prepared. Example dims -> x: {train_edges[0].x.shape}  "
              f"edge_attr: {train_edges[0].edge_attr.shape if hasattr(train_edges[0], 'edge_attr') and train_edges[0].edge_attr is not None else None}")

    validate_edges(train_edges, 'train_edges')

    # Train/val split
    if scfg.task == 'scarce':
        n = len(train_edges)
        n_train = int(n * 0.9)
        ids_all = list(range(n))
        random.Random(scfg.seed).shuffle(ids_all)
        ids_train = ids_all[:n_train]
        ids_val = ids_all[n_train:]
        train_edges_subset = [train_edges[i] for i in ids_train]
        val_edges_subset = [train_edges[i] for i in ids_val] if ids_val else []
    else:
        train_edges_subset = train_edges
        val_edges_subset = val_edges

    train_prepped = [_prep_graph_for_norm(g) for g in train_edges_subset]
    val_prepped = [_prep_graph_for_norm(g) for g in val_edges_subset] if isinstance(val_edges_subset, list) else []

    # Fit scalers
    X_train = torch.cat([d.x for d in train_prepped if hasattr(d, 'x') and d.x is not None], dim=0)
    Y_train = torch.cat([d.y for d in train_prepped if hasattr(d, 'y') and d.y is not None], dim=0)

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    # Build normalized datasets
    train_norm = NormalizedDataset(train_prepped, x_scaler, y_scaler)
    val_norm = NormalizedDataset(val_prepped, x_scaler, y_scaler) if isinstance(val_prepped, list) and len(val_prepped) > 0 else []

    print(f'Prepared normalized datasets: {len(train_norm)} train | '
          f'{len(val_norm) if isinstance(val_norm, NormalizedDataset) else len(val_norm)} val')

    # Build DataLoaders
    train_loader = DataLoader(train_norm, batch_size=scfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_pyg)
    val_loader = DataLoader(val_norm, batch_size=scfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pyg) if isinstance(val_norm, NormalizedDataset) else []

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        train_graphs=train_edges,
        val_graphs=val_edges,
        train_norm=train_norm,
        val_norm=val_norm,
    )


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
                'bc_loss': 0.0,
                'total_loss': float(mse_loss.detach().item())
            }
    else:
        mse_loss = _mse_loss_fn(predictions, targets)
        return mse_loss, {
            'mse_loss': float(mse_loss.detach().item()),
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
    wandb_init_kwargs = dict(
        project=getattr(scfg, "wandb_project", "airfrans-gnn"),
        name=getattr(scfg, "wandb_run_name", None),
        tags=getattr(scfg, "wandb_tags", None),
        mode=getattr(scfg, "wandb_mode", "online"),
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
        wandb_init_kwargs["config"].update({
            "cont_w0": getattr(loss_fn, "cont_w0", None),
            "cont_w_target": getattr(loss_fn, "cont_w_target", None),
            "mom_w0": getattr(loss_fn, "mom_w0", None),
            "mom_w_target": getattr(loss_fn, "mom_w_target", None),
            "ramp_steps": getattr(loss_fn, "curr_steps", None),
            "ramp_start_step": getattr(loss_fn, "ramp_start_step", 0),
        })

    wandb_run = wandb.init(**wandb_init_kwargs)

    # Epoch-only logging: swallow per-step logs (commit=False)
    if getattr(scfg, "log_epoch_only", True):
        try:
            _wandb_orig_log = wandb.log

            def _log_epoch_only(data=None, step=None, commit=None, *args, **kwargs):
                if commit is None:
                    commit = True
                if commit is False:
                    return
                return _wandb_orig_log(data, step=step, commit=commit, *args, **kwargs)

            wandb.log = _log_epoch_only
        except Exception:
            pass

    return wandb_run
