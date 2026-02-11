#!/usr/bin/env python3
"""Auto-generated training script from 01_trainer.ipynb (scarce task)"""

# ============================================================
# Cell 2
# ============================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import matplotlib
matplotlib.use('Agg')  # headless backend
import sys, math, json, random, contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric.datasets import AirfRANS
from torch_geometric.data import Data, Batch
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch_geometric.data import Data
from src.navier_stokes_physics_loss import NavierStokesPhysicsLoss
from src.airfrans_utils import prepare_airfrans_graph_for_physics, estimate_node_area, build_bc_masks_airfrans
import contextlib
import wandb  
from torch.cuda.amp import GradScaler, autocast

def get_lr(optim):
    return optim.param_groups[0].get('lr', None)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(42)
print('SmokeTest | PyTorch:', torch.__version__, '| CUDA?', torch.cuda.is_available())

# ============================================================
# Cell 4
# ============================================================
# 2) Configuration (minimal for smoke)
from dataclasses import dataclass, asdict

@dataclass
class SmokeCfg:
    seed: int = 42
    task: str = 'scarce'
    root: str = 'Dataset'
    # subsample graph count for smoke
    limit_train: int = 200
    limit_val: int = 200

    # training
    batch_size: int = 1
    epochs: int = 100
    hidden: int = 128
    layers: int = 14
    lr: float = 4e-4
    weight_decay: float = 1e-2  # typical AdamW wd
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    amp: bool = False

    # lr scheduler: 'cosine', 'cosine_warm_restarts', 'reduce_on_plateau', or None
    lr_scheduler: str = 'cosine'
    # cosine params
    cosine_T_max: int = 80  # epochs
    cosine_eta_min: float = 1e-6
    # warm restarts params
    wr_T_0: int = 10
    wr_T_mult: int = 1
    wr_eta_min: float = 1e-6
    # reduce on plateau params
    rop_factor: float = 0.5
    rop_patience: int = 5
    rop_min_lr: float = 1e-6

    # Physics-Informed Loss Configuration
    # =====================================
    # Curriculum learning schedule
    ramp_start_epoch: int = 40              # Epoch to start ramping physics losses
    ramp_epochs: int = 60                   # Number of epochs to ramp up
    ramp_mode: str = 'linear'               # 'linear' or 'cosine'
    
    # MSE/Data loss
    data_loss_weight: float = 1.0           # Weight for MSE loss (constant)
    
    # Continuity equation loss
    continuity_loss_weight: float = 0.05    # Initial continuity weight
    continuity_target_weight: float = 0.20  # Target continuity weight after ramp
    
    # Momentum equation loss  
    momentum_loss_weight: float = 0.05      # Initial momentum weight
    momentum_target_weight: float = 0.20    # Target momentum weight after ramp
    
    # Boundary condition loss
    bc_loss_weight: float = 0.1              # Weight for boundary condition loss
    
    # Physics parameters
    chord_length: float = 1.0               # Airfoil chord length
    nu_molecular: float = 1.5e-5            # Molecular viscosity
    dynamic_uref_from_data: bool = True     # Compute reference velocity from data
    dynamic_re_from_data: bool = True       # Compute Reynolds number from data
    uinf_from: str = 'inlet'                # 'inlet', 'farfield', or 'robust'
    
    # Stability & outlier control
    use_huber_for_physics: bool = True       # Use Huber loss for physics terms
    huber_delta: float = 0.05                 # Huber loss delta parameter
    use_perimeter_norm_for_div: bool = True   # Normalize divergence by perimeter
    div_area_floor_factor: float = 0.25       # Area floor factor for stability
    div_min_degree: int = 2                     # Minimum node degree for physics loss

    # Debug & monitoring
    physics_debug: bool = False              # Enable physics loss debugging
    physics_debug_level: int = 1              # Debug verbosity (1=summary, 2=detailed)
    physics_debug_every: int = 50           # Log debug info every N steps

    # Global Context & Attention Configuration
    use_global_tokens: bool = True           # Enable/disable global tokens
    num_global_tokens: int = 2               # Number of global tokens
    attention_heads: int = 2                 # Multi-head attention heads
    attention_layers: int = 2               # Number of transformer layers
    attention_dropout: float = 0.0           # Attention dropout rate
    use_cross_attention: bool = True         # Cross-attention between local and global
    global_pooling_type: str = 'attention'   # 'mean', 'max', 'attention', 'set2set'
    positional_encoding: bool = False         # Use positional encoding
    pos_encoding_max_len: int = 50000        # Max sequence length for positional encoding
    # Advanced attention options
    use_residual_attention: bool = True      # Residual connections in attention
    attention_normalization: str = 'layer'   # 'layer', 'batch', 'rms'
    temperature_scaling: bool = False        # Temperature scaling for attention
    attention_bias: bool = False             # Use bias in attention projections

    # W&B Artifact 관리
    use_wandb_artifacts: bool = False        # W&B artifact 사용 여부
    artifact_save_best_only: bool = True     # best 모델만 업로드
    artifact_save_interval: int = 50         # periodic 저장 간격 (epochs)
    
    # Checkpoint 관리
    ckpt_dir: str = "checkpoints"           # 로컬 체크포인트 디렉토리
    ckpt_interval: int = 5                  # 로컬 체크포인트 저장 간격
    
    # W&B 설정
    wandb_project: str = "airfrans-gnn"
    wandb_mode: str = "online"              # "online", "offline", "disabled"
    log_every_n_steps: int = -1             # 로깅 빈도
    log_epoch_only: bool = True             # Epoch 로깅만 사용

scfg = SmokeCfg()
set_seed(scfg.seed)
print('Smoke config:', asdict(scfg))

# ============================================================
# Cell 6
# ============================================================
from torch_geometric.transforms import BaseTransform

class _PreparePhysics(BaseTransform):
    def forward(self, data):
        # edge_attr_dxdy가 이미 있을 경우 build_edge_attr_dxdy는 생략되고 나머지만 수행
        return prepare_airfrans_graph_for_physics(data, verbose=False)

# 3) Load dataset indices (train/val split)
assert os.path.isdir(scfg.root), f"Dataset folder not found: {scfg.root}"
try:
    ds_train = AirfRANS(root=scfg.root, train=True, task=scfg.task, transform=_PreparePhysics())
    ds_test  = AirfRANS(root=scfg.root, train=False, task=scfg.task, transform=_PreparePhysics())
except TypeError:
    ds_train = AirfRANS(root=scfg.root, train=True, task=scfg.task, transform=_PreparePhysics())
    ds_test  = AirfRANS(root=scfg.root, train=False, task=scfg.task, transform=_PreparePhysics())

if scfg.task == 'scarce':
    # Scarce provides train only; create 90/10 split from ds_train
    n = len(ds_train)
    ids_all = list(range(n))
    random.Random(scfg.seed).shuffle(ids_all)
    
    # Split into train and val
    n_train = scfg.limit_train if scfg.limit_train > 0 else int(0.9 * n)
    n_val = scfg.limit_val if scfg.limit_val > 0 else (n - n_train)
    
    ids_train = ids_all[:n_train]
    ids_val = ids_all[n_train:n_train + n_val]
    
    train_raw = Subset(ds_train, ids_train)
    val_raw = Subset(ds_train, ids_val)
    
else:
    ids_train = list(range(min(scfg.limit_train+scfg.limit_val, len(ds_train))))
    ids_val = ids_train[-scfg.limit_val:] if scfg.limit_val>0 else []
    ids_train = ids_train[:scfg.limit_train] if scfg.limit_train>0 else ids_train
    train_raw = Subset(ds_train, ids_train)
    val_raw   = Subset(ds_train, ids_val) if ids_val else []

print('Loaded subset indices:', len(train_raw), 'train |', len(val_raw) if isinstance(val_raw, Subset) else 0, 'val/test')

# ============================================================
# Cell 8
# ============================================================
# 6) Load prebuilt graphs and ensure features (index-aligned with raw)
import glob, os, re
from src.utils import with_pos2, prep_graph, validate_edges, _prep_graph_for_norm

USE_PREBUILT = True
if scfg.task == 'full':
    PREBUILT_ROOT = 'prebuilt_edges/full'
else:
    PREBUILT_ROOT = 'prebuilt_edges/scarce'
    
PREBUILT_TRAIN_DIR = f"{PREBUILT_ROOT}/train"
PREBUILT_TEST_DIR  = f"{PREBUILT_ROOT}/test"
DOWNSAMPLED_ROOT = f"downsampled_graphs/{scfg.task}"

# Load prebuilt edge graphs
train_edge_files = sorted(glob.glob(os.path.join(PREBUILT_TRAIN_DIR, 'graph_*.pt')))
val_edge_files   = sorted(glob.glob(os.path.join(PREBUILT_TEST_DIR,  'graph_*.pt')))
print(f"[prebuilt] found: {len(train_edge_files)} train and {len(val_edge_files)} val graphs under {PREBUILT_ROOT}")

# Load tensors and prepare
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

print(f"Graphs prepared. Example dims -> x: {train_edges[0].x.shape if len(train_edges)>0 else None}  edge_attr: {train_edges[0].edge_attr.shape if (len(train_edges)>0 and hasattr(train_edges[0],'edge_attr') and train_edges[0].edge_attr is not None) else None}")

validate_edges(train_edges, 'train_edges')

# ============================================================
# Cell 10
# ============================================================
if scfg.task == 'scarce':
    n = len(train_edges)
    n_train = int(n * 0.9)
    ids_all = list(range(n))
    random.Random(scfg.seed).shuffle(ids_all)
    ids_train = ids_all[:n_train]
    ids_val = ids_all[n_train:]

    # Use prebuilt graphs, not raw dataset
    train_edges_subset = [train_edges[i] for i in ids_train]
    val_edges_subset = [train_edges[i] for i in ids_val] if ids_val else []
else:
    train_edges_subset = train_edges
    val_edges_subset = val_edges

train_prepped = [_prep_graph_for_norm(g) for g in train_edges_subset]
val_prepped   = [_prep_graph_for_norm(g) for g in val_edges_subset] if isinstance(val_edges_subset, list) else []

# 8b) Fit scalers on train_prepped
if 'StandardScaler' not in globals():
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

# Concatenate node features/targets across train graphs for fitting
X_train = torch.cat([d.x for d in train_prepped if hasattr(d, 'x') and d.x is not None], dim=0)
Y_train = torch.cat([d.y for d in train_prepped if hasattr(d, 'y') and d.y is not None], dim=0)

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(Y_train)

# 8c) Build normalized dataset wrappers
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
            
        # DON'T attach norm params as graph attributes - they cause batching issues
        # Instead, we'll handle denormalization differently
        # dm.x_norm_params = {'mean': self.x_scaler.mean.clone(), 'scale': self.x_scaler.std.clone()}
        # dm.y_norm_params = {'mean': self.y_scaler.mean.clone(), 'scale': self.y_scaler.std.clone()} if dm.y is not None else None
        
        # Store scalers as module-level attributes for physics loss to access
        dm.has_norm = True  # Flag to indicate normalized data
        
        # Ensure edge_attr_dxdy is present (needed for physics loss)
        if hasattr(d, 'edge_attr_dxdy'):
            dm.edge_attr_dxdy = d.edge_attr_dxdy
        elif hasattr(d, 'edge_attr'):
            # If we have edge_attr but not edge_attr_dxdy, use the last 2 dims as dxdy
            if d.edge_attr.shape[1] >= 2:
                dm.edge_attr_dxdy = d.edge_attr[:, -2:]  # Last 2 columns should be dx, dy
            dm.edge_attr = d.edge_attr
        
        # Build BC masks properly
        from src.airfrans_utils import build_bc_masks_airfrans
        dm = build_bc_masks_airfrans(dm)
        
        # Ensure individual BC masks are present as attributes
        if hasattr(dm, 'bc_mask_dict'):
            for bc_type, mask in dm.bc_mask_dict.items():
                setattr(dm, f'is_{bc_type}', mask)
        else:
            # Fallback: create default masks if build_bc_masks_airfrans failed
            num_nodes = dm.x.size(0)
            # Use the normalized x for BC detection
            x_orig = d.x  # Use original (non-normalized) for BC detection
            
            # Wall nodes: distance_wall < threshold (column 2 of original x)
            if x_orig.size(1) > 2:
                wall_dist = x_orig[:, 2]
                dm.is_wall = (wall_dist < 1e-6)
            else:
                dm.is_wall = torch.zeros(num_nodes, dtype=torch.bool)
            
            # For AirfRANS, we typically don't have explicit inlet/outlet/farfield in the features
            # These would need to be inferred from position or other criteria
            dm.is_inlet = torch.zeros(num_nodes, dtype=torch.bool)
            dm.is_outlet = torch.zeros(num_nodes, dtype=torch.bool)
            dm.is_farfield = torch.zeros(num_nodes, dtype=torch.bool)
            
            # Simple heuristics for inlet/outlet/farfield based on position
            if hasattr(dm, 'pos'):
                x_coords = dm.pos[:, 0]
                y_coords = dm.pos[:, 1]
                
                # Inlet: leftmost boundary (x < -1)
                dm.is_inlet = (x_coords < -1.0) & ~dm.is_wall
                
                # Outlet: rightmost boundary (x > 2)
                dm.is_outlet = (x_coords > 2.0) & ~dm.is_wall
                
                # Farfield: top/bottom boundaries (|y| > 1)
                dm.is_farfield = (torch.abs(y_coords) > 1.0) & ~dm.is_wall & ~dm.is_inlet & ~dm.is_outlet
        
        return dm

train_norm = NormalizedDataset(train_prepped, x_scaler, y_scaler)
val_norm   = NormalizedDataset(val_prepped, x_scaler, y_scaler) if isinstance(val_prepped, list) and len(val_prepped) > 0 else []

# Debug BC mask creation for a single sample
test_single = train_norm[0]
print("Single sample BC check:")
print(f"  Total nodes: {test_single.x.size(0)}")

# Check original features that determine BC
if hasattr(test_single, 'x'):
    x_orig = train_prepped[0].x  # Original unnormalized
    print(f"  Original x shape: {x_orig.shape}")
    if x_orig.size(1) > 2:
        wall_dist = x_orig[:, 2]
        print(f"  Wall distance range: [{wall_dist.min():.3e}, {wall_dist.max():.3e}]")
        print(f"  Nodes with wall_dist < 1e-6: {(wall_dist < 1e-6).sum().item()}")

# Check the BC masks
for bc_type in ['wall', 'inlet', 'outlet', 'farfield']:
    mask_name = f'is_{bc_type}'
    if hasattr(test_single, mask_name):
        mask = getattr(test_single, mask_name)
        print(f"  {mask_name}: {mask.sum().item()} nodes ({mask.sum().item()/len(mask)*100:.1f}%)")

# Check position-based criteria if available
if hasattr(test_single, 'pos'):
    pos = test_single.pos
    print(f"\n  Position ranges:")
    print(f"    x: [{pos[:, 0].min():.2f}, {pos[:, 0].max():.2f}]")
    print(f"    y: [{pos[:, 1].min():.2f}, {pos[:, 1].max():.2f}]")



print('Prepared normalized datasets:', len(train_norm), 'train |', (len(val_norm) if isinstance(val_norm, NormalizedDataset) else len(val_norm)), 'val')
if len(train_prepped) > 0:
    print('Example dims -> x:', tuple(train_prepped[0].x.shape), '| edge_attr:', (tuple(train_prepped[0].edge_attr.shape) if hasattr(train_prepped[0], 'edge_attr') and train_prepped[0].edge_attr is not None else None))

# ============================================================
# Cell 13
# ============================================================
# Use true batching with PyG Batch.from_data_list so batch_size>1 works correctly

def collate_pyg(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return Batch.from_data_list(batch)

train_loader = DataLoader(train_norm, batch_size=scfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_pyg)
val_loader   = DataLoader(val_norm,   batch_size=scfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pyg) if isinstance(val_norm, NormalizedDataset) else []
print('Loaders ready:', len(train_norm), 'train samples | batch_size =', scfg.batch_size)
print('Loaders ready:', len(val_norm), 'val samples | batch_size =', scfg.batch_size)

# ============================================================
# Cell 15
# ============================================================
# 13) Enhanced Train/Val epoch routines with Physics Loss

mse_loss_fn = nn.MSELoss()
_CH_NAMES = ['u', 'v', 'p', 'nut']

def _per_channel_mse(predictions, targets):
    """Compute per-channel MSE for logging (returns plain floats)."""
    with torch.no_grad():
        return {f'mse_{cn}': float(F.mse_loss(predictions[:, i], targets[:, i]).item())
                for i, cn in enumerate(_CH_NAMES)}

def compute_loss_with_physics(predictions, targets, data, loss_fn=None, *, step: int | None = None):
    """Compute loss using physics-informed loss function or fallback to MSE
    Returns a differentiable scalar loss tensor for backward as first value,
    and a lightweight dict of float metrics for logging as second value.
    """
    ch_mse = _per_channel_mse(predictions, targets)

    if loss_fn is not None:
        try:
            # Always let the physics loss handle batched Data (PyG batches are a big disjoint graph)
            loss_dict = loss_fn(predictions, targets, data=data, step=step)

            # Ensure total_loss is a Tensor usable for backward
            total_loss = loss_dict.get('total_loss')
            if not isinstance(total_loss, torch.Tensor):
                total_loss = torch.as_tensor(total_loss, dtype=predictions.dtype, device=predictions.device)

            # Prepare a logging-friendly dict (floats only) to avoid holding graph refs
            log_dict = {}
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    try:
                        log_dict[k] = float(v.detach().item())
                    except Exception:
                        # Fallback if it's not 0-dim
                        log_dict[k] = float(v.detach().mean().item())
                else:
                    log_dict[k] = float(v)

            log_dict.update(ch_mse)
            return total_loss, log_dict
        except Exception as e:
            print(f"Warning: Physics loss failed ({e}), falling back to MSE")
            mse_loss = mse_loss_fn(predictions, targets)
            return mse_loss, {
                'mse_loss': float(mse_loss.detach().item()), 
                'continuity_loss': 0.0, 
                'momentum_loss': 0.0,
                'bc_loss': 0.0,
                'total_loss': float(mse_loss.detach().item()),
                **ch_mse,
            }
    else:
        # Fallback to simple MSE
        mse_loss = mse_loss_fn(predictions, targets)
        return mse_loss, {
            'mse_loss': float(mse_loss.detach().item()), 
            'bc_loss': 0.0,
            'total_loss': float(mse_loss.detach().item()),
            **ch_mse,
        }


@torch.no_grad()
def run_epoch(loader, model, device, scaler=None, desc: str = 'val', loss_fn=None):
    model.eval()
    total_losses = []; mse_losses = []; continuity_losses = []; momentum_losses = []
    bc_losses = []
    per_ch_losses = {cn: [] for cn in _CH_NAMES}
    cont_w_used_hist, mom_w_used_hist = [], []

    if loader is None or (isinstance(loader, list) and len(loader)==0):
        return float('nan'), {}

    steps = len(loader)
    pbar = tqdm(total=steps, desc=desc, leave=False)

    for batch in loader:
        try:
            if batch is None:
                pbar.update(1); continue

            b = batch.to(device)
            with (autocast(enabled=(scfg.amp and torch.cuda.is_available()))
                  if torch.cuda.is_available() else contextlib.nullcontext()):
                out = model(b)
                _, loss_dict = compute_loss_with_physics(out, b.y, b, loss_fn=loss_fn, step=None)

            total_losses.append(loss_dict['total_loss'])
            mse_losses.append(loss_dict['mse_loss'])
            continuity_losses.append(loss_dict.get('continuity_loss', 0.0))
            momentum_losses.append(loss_dict.get('momentum_loss', 0.0))
            bc_losses.append(loss_dict.get('bc_loss', 0.0))
            for cn in _CH_NAMES:
                per_ch_losses[cn].append(loss_dict.get(f'mse_{cn}', 0.0))
            if 'cont_weight_used' in loss_dict: cont_w_used_hist.append(loss_dict['cont_weight_used'])
            if 'mom_weight_used'  in loss_dict: mom_w_used_hist.append(loss_dict['mom_weight_used'])

            postfix = {"total": f"{loss_dict['total_loss']:.4e}"}
            if 'continuity_loss' in loss_dict: postfix["cont"] = f"{loss_dict['continuity_loss']:.4e}"
            if 'momentum_loss' in loss_dict:   postfix["momentum"] = f"{loss_dict['momentum_loss']:.4e}"
            if 'bc_loss' in loss_dict:         postfix["bc"] = f"{loss_dict['bc_loss']:.4e}"
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
    for cn in _CH_NAMES:
        avg_losses[f'mse_{cn}'] = np.mean(per_ch_losses[cn]) if per_ch_losses[cn] else float('nan')
    if cont_w_used_hist: avg_losses['cont_weight_used'] = float(np.mean(cont_w_used_hist))
    if mom_w_used_hist:  avg_losses['mom_weight_used']  = float(np.mean(mom_w_used_hist))
    return avg_losses['total_loss'], avg_losses



def train_epoch(loader, model, optim, device, scaler, desc: str = 'train',
                loss_fn=None, global_step_start: int = 0, scheduler=None, scheduler_step_mode: str = "epoch",
                log_every_n_steps: int = -1):  # -1로 설정하면 step 로깅 비활성화
    model.train()
    total_losses, mse_losses, continuity_losses, momentum_losses = [], [], [], []
    bc_losses = []
    per_ch_losses = {cn: [] for cn in _CH_NAMES}
    cont_w_used_hist, mom_w_used_hist = [], []

    global_step = global_step_start
    steps = len(loader)
    pbar = tqdm(total=steps, desc=desc, leave=False)

    for batch_idx, batch in enumerate(loader):
        try:
            if batch is None:
                pbar.update(1); global_step += 1; continue

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

            # 집계
            total_losses.append(loss_dict['total_loss'])
            mse_losses.append(loss_dict['mse_loss'])
            continuity_losses.append(loss_dict.get('continuity_loss', 0.0))
            momentum_losses.append(loss_dict.get('momentum_loss', 0.0))
            bc_losses.append(loss_dict.get('bc_loss', 0.0))
            for cn in _CH_NAMES:
                per_ch_losses[cn].append(loss_dict.get(f'mse_{cn}', 0.0))
            if 'cont_weight_used' in loss_dict: cont_w_used_hist.append(loss_dict['cont_weight_used'])
            if 'mom_weight_used'  in loss_dict: mom_w_used_hist.append(loss_dict['mom_weight_used'])

            # === Step-level 로깅 ===
            if log_every_n_steps > 0 and (batch_idx % max(1, log_every_n_steps)) == 0:
                log_payload = {
                    "step": global_step,
                    "train/total": loss_dict['total_loss'],
                    "train/mse": loss_dict['mse_loss'],
                    "train/continuity": loss_dict.get('continuity_loss', 0.0),
                    "train/momentum": loss_dict.get('momentum_loss', 0.0),
                    "train/bc": loss_dict.get('bc_loss', 0.0),
                }
                for cn in _CH_NAMES:
                    log_payload[f"train/mse_{cn}"] = loss_dict.get(f'mse_{cn}', 0.0)
                if 'cont_weight_used' in loss_dict: log_payload["weight/cont_used"] = loss_dict['cont_weight_used']
                if 'mom_weight_used'  in loss_dict: log_payload["weight/mom_used"]  = loss_dict['mom_weight_used']
                lr_now = get_lr(optim)
                if lr_now is not None:
                    log_payload["lr"] = lr_now
                wandb.log(log_payload, step=global_step, commit=False)

            postfix = {"total": f"{loss_dict['total_loss']:.4e}",
                       "lr": f"{get_lr(optim):.2e}" if get_lr(optim) is not None else "n/a"}
            if 'continuity_loss' in loss_dict: postfix["cont"] = f"{loss_dict['continuity_loss']:.4e}"
            if 'momentum_loss' in loss_dict:   postfix["momentum"] = f"{loss_dict['momentum_loss']:.4e}"
            if 'bc_loss' in loss_dict:         postfix["bc"] = f"{loss_dict['bc_loss']:.4e}"
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
    for cn in _CH_NAMES:
        avg_losses[f'mse_{cn}'] = np.mean(per_ch_losses[cn]) if per_ch_losses[cn] else float('nan')
    if cont_w_used_hist: avg_losses['cont_weight_used'] = float(np.mean(cont_w_used_hist))
    if mom_w_used_hist:  avg_losses['mom_weight_used']  = float(np.mean(mom_w_used_hist))

    return avg_losses['total_loss'], avg_losses, global_step

# ============================================================
# Cell 17
# ============================================================
# Import and setup physics-informed loss
from importlib import reload
from typing import Dict, Optional
from src.navier_stokes_physics_loss import NavierStokesPhysicsLoss

steps_per_epoch = len(train_loader)

loss_fn = NavierStokesPhysicsLoss(
    # Data loss
    data_loss_weight=scfg.data_loss_weight,

    # Continuity loss with curriculum
    continuity_loss_weight=scfg.continuity_loss_weight,
    continuity_target_weight=scfg.continuity_target_weight,

    # Momentum loss with curriculum
    momentum_loss_weight=scfg.momentum_loss_weight,
    momentum_target_weight=scfg.momentum_target_weight,

    # Curriculum schedule
    curriculum_ramp_steps=scfg.ramp_epochs * steps_per_epoch,
    ramp_start_step=scfg.ramp_start_epoch * steps_per_epoch,
    cont_curriculum_ramp_steps=-1,  # Use shared schedule
    mom_curriculum_ramp_steps=-1,   # Use shared schedule
    ramp_mode=scfg.ramp_mode,

    # Boundary conditions
    bc_loss_weight=scfg.bc_loss_weight,
    chord_length=scfg.chord_length,
    
    # Physics parameters
    dynamic_uref_from_data=scfg.dynamic_uref_from_data,
    dynamic_re_from_data=scfg.dynamic_re_from_data,
    nu_molecular=scfg.nu_molecular,
    uinf_from=scfg.uinf_from,

    # Stability controls
    use_huber_for_physics=scfg.use_huber_for_physics,
    huber_delta=scfg.huber_delta,
    use_perimeter_norm_for_div=scfg.use_perimeter_norm_for_div,
    div_area_floor_factor=scfg.div_area_floor_factor,
    div_min_degree=scfg.div_min_degree,
    
    # Debug settings
    debug=scfg.physics_debug,
    debug_level=scfg.physics_debug_level,
    debug_every=scfg.physics_debug_every,
)

print(f"✅ Physics loss initialized:")
print(f"   Curriculum: start at epoch {scfg.ramp_start_epoch}, ramp for {scfg.ramp_epochs} epochs")
print(f"   Continuity: {scfg.continuity_loss_weight:.3f} → {scfg.continuity_target_weight:.3f}")
print(f"   Momentum: {scfg.momentum_loss_weight:.3f} → {scfg.momentum_target_weight:.3f}")
print(f"   BC weight: {scfg.bc_loss_weight:.3f} (constant)")
print(f"   ├─ Wall: No-slip condition (u=v=0)")
print(f"   ├─ Inlet: Prescribed velocity")
print(f"   ├─ Outlet: Zero-gradient")
print(f"   └─ Farfield: Freestream conditions")

# ============================================================
# Cell 21
# ============================================================
# Define model using EnhancedCFDModelWithGlobalContext (safe device placement)
from src.global_context_processor import EnhancedCFDModelWithGlobalContext
import torch
import gc

# Clean up memory before model instantiation
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    # Dimensions: utils._prep_graph_for_norm() appends pos2 -> node_dim = 7, edge_dim ensured = 5
    node_dim = 7
    edge_dim = 5

    # Set device: use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate model with current smoke config (scfg)
    model = EnhancedCFDModelWithGlobalContext(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=scfg.hidden,
        output_dim=4,
        num_mp_layers=scfg.layers,
        dropout_p=0.1,
        config=scfg  # pass explicitly to avoid module-scope dependency
    )

    # Move to target device
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scfg.lr,
        weight_decay=scfg.weight_decay,
        betas=scfg.betas,
        eps=scfg.eps,
    )

    print('🚀 Enhanced CFD Model with Global Context ready!')
    print(f'Configuration: global_tokens={scfg.use_global_tokens}, num_tokens={scfg.num_global_tokens}')
    print(f'Attention: heads={scfg.attention_heads}, layers={scfg.attention_layers}')
    print(f'Cross-attention={scfg.use_cross_attention}, pooling={scfg.global_pooling_type}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(f'Node dim: {node_dim} | Edge dim: {edge_dim}')
    print(f"[DEVICE] Model placed on: {device}")

except torch.cuda.OutOfMemoryError:
    print("CUDA out of memory. Kernel will not crash.")
    print("Try reducing batch size or model size.")
    # Free up memory
    del model
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    # Free up memory
    if 'model' in locals():
        del model
    if 'optimizer' in locals():
        del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# Cell 24
# ============================================================
# Learning Rate Scheduler 설정
def create_lr_scheduler(optimizer, config):
    """Configuration에 따라 적절한 LR scheduler를 생성합니다."""
    
    if config.lr_scheduler is None:
        print("🚫 Learning rate scheduler: None (constant LR)")
        return None
    
    elif config.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.cosine_T_max,
            eta_min=config.cosine_eta_min
        )
        print(f"📊 Learning rate scheduler: CosineAnnealingLR")
        print(f"   T_max: {config.cosine_T_max}, eta_min: {config.cosine_eta_min}")
        return scheduler
    
    elif config.lr_scheduler == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.wr_T_0,
            T_mult=config.wr_T_mult,
            eta_min=config.wr_eta_min
        )
        print(f"🔄 Learning rate scheduler: CosineAnnealingWarmRestarts")
        print(f"   T_0: {config.wr_T_0}, T_mult: {config.wr_T_mult}, eta_min: {config.wr_eta_min}")
        return scheduler
    
    elif config.lr_scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # validation loss를 minimize
            factor=config.rop_factor,
            patience=config.rop_patience,
            min_lr=config.rop_min_lr,
        )
        print(f"📉 Learning rate scheduler: ReduceLROnPlateau")
        print(f"   factor: {config.rop_factor}, patience: {config.rop_patience}, min_lr: {config.rop_min_lr}")
        return scheduler
    
    else:
        print(f"❌ Unknown scheduler: {config.lr_scheduler}, using None")
        return None

# Scheduler 생성
lr_scheduler = create_lr_scheduler(optimizer, scfg)

print(f"\n🎯 Current configuration:")
print(f"   Initial LR: {scfg.lr}")
print(f"   Scheduler: {scfg.lr_scheduler}")
print(f"   Epochs: {scfg.epochs}")

# 스케줄러별 LR 변화 시뮬레이션 (시각화용)
def simulate_lr_schedule(config, num_epochs=20):
    """LR 스케줄 변화를 시뮬레이션합니다."""
    import copy
    
    # 임시 optimizer 생성
    temp_param = torch.nn.Parameter(torch.randn(1))
    temp_opt = torch.optim.AdamW([temp_param], lr=config.lr)
    temp_scheduler = create_lr_scheduler(temp_opt, config)
    
    lrs = []
    val_losses = [1.0, 0.9, 0.85, 0.8, 0.85, 0.82, 0.81, 0.80, 0.82, 0.79,
                  0.78, 0.77, 0.78, 0.76, 0.75, 0.76, 0.74, 0.73, 0.74, 0.72]  # 가상의 validation loss
    
    for epoch in range(num_epochs):
        lrs.append(temp_opt.param_groups[0]['lr'])
        
        if temp_scheduler is not None:
            if config.lr_scheduler == 'reduce_on_plateau':
                # ReduceLROnPlateau의 경우 validation loss 필요
                val_loss = val_losses[epoch] if epoch < len(val_losses) else val_losses[-1]
                temp_scheduler.step(val_loss)
            else:
                # 다른 스케줄러는 epoch만 필요
                temp_scheduler.step()
    
    return lrs

# LR 스케줄 시각화
if scfg.lr_scheduler is not None:
    lrs = simulate_lr_schedule(scfg, scfg.epochs)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(lrs, 'o-', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'LR Schedule: {scfg.lr_scheduler}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(lrs, 'o-', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.title(f'LR Schedule: {scfg.lr_scheduler} (Log)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lr_schedule_preview.png', dpi=100)
    plt.close()
    print("LR schedule plot saved to lr_schedule_preview.png")
    
    print(f"📈 LR Schedule Preview:")
    for i, lr in enumerate(lrs[:10]):  # 처음 10 epoch만 표시
        print(f"   Epoch {i+1:2d}: {lr:.2e}")
    if len(lrs) > 10:
        print(f"   ...     : ...")
        print(f"   Epoch {len(lrs):2d}: {lrs[-1]:.2e}")
else:
    print("📊 No scheduler selected - constant learning rate will be used.")

# ============================================================
# Cell 26
# ============================================================
# === Weights & Biases init ===

# Type normalization helpers
def _to_list(v):
    """Convert to list if not already: str -> [str], iterable -> list, None -> None"""
    if v is None:
        return None
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return v
    try:
        return list(v)
    except (TypeError, ValueError):
        return None

def _validate_choice(value, valid_choices, default=None):
    """Return value if in valid_choices, else default"""
    return value if value in valid_choices else default

# Gather config dict
wandb_config = {
    "epochs": getattr(scfg, "epochs", None),
    "batch_size": getattr(scfg, "batch_size", None),
    "lr": getattr(scfg, "lr", None),
    "optimizer": getattr(scfg, "optimizer", "adam"),
    "scheduler": getattr(scfg, "scheduler", None),
    "amp": getattr(scfg, "amp", False),
    "cont_w0": getattr(loss_fn, "cont_w0", None),
    "cont_w_target": getattr(loss_fn, "cont_w_target", None),
    "mom_w0": getattr(loss_fn, "mom_w0", None),
    "mom_w_target": getattr(loss_fn, "mom_w_target", None),
    "ramp_steps": getattr(loss_fn, "curr_steps", None),
    "ramp_start_step": getattr(loss_fn, "ramp_start_step", 0),
}

# Prepare wandb.init arguments
wandb_run = wandb.init(
    project=getattr(scfg, "wandb_project", "airfrans-gnn"),
    name=getattr(scfg, "wandb_run_name", None),
    tags=_to_list(getattr(scfg, "wandb_tags", None)),
    mode=_validate_choice(getattr(scfg, "wandb_mode", "online"), 
                          ("online", "offline", "disabled", "shared"), "online"),
    settings=wandb.Settings(start_method="thread"),
    config=wandb_config,
    config_include_keys=_to_list(getattr(scfg, "wandb_config_include_keys", None)),
    config_exclude_keys=_to_list(getattr(scfg, "wandb_config_exclude_keys", None)),
    allow_val_change=getattr(scfg, "wandb_allow_val_change", None),
    reinit=_validate_choice(getattr(scfg, "wandb_reinit", None),
                            (True, False, "default", "return_previous", "finish_previous", "create_new", None)),
    resume=_validate_choice(getattr(scfg, "wandb_resume", None),
                            (True, False, "allow", "never", "must", "auto", None)),
    force=getattr(scfg, "wandb_force", None),
    save_code=getattr(scfg, "wandb_save_code", None),
    tensorboard=getattr(scfg, "wandb_tensorboard", None),
    sync_tensorboard=getattr(scfg, "wandb_sync_tensorboard", None),
    monitor_gym=getattr(scfg, "wandb_monitor_gym", None),
)

# ============================================================
# Cell 28
# ============================================================
# Enhanced Training Loop with Optimized W&B Artifact Management
def train_with_scheduler(model, optim, scheduler, train_loader, val_loader, 
                        scfg, device, scaler=None, physics_loss_fn=None):
    scaler = GradScaler(enabled=(scfg.amp and torch.cuda.is_available()))
    global_step = 0
    best_val = float('inf')
    
    # Artifact 관리 설정
    USE_WANDB_ARTIFACTS = getattr(scfg, "use_wandb_artifacts", False)  # 기본값 False
    ARTIFACT_SAVE_BEST_ONLY = getattr(scfg, "artifact_save_best_only", True)  # best만 저장
    ARTIFACT_SAVE_INTERVAL = getattr(scfg, "artifact_save_interval", 20)  # 20 epochs마다
    
    EPOCHS = getattr(scfg, "epochs", 50)
    ckpt_dir = getattr(scfg, "ckpt_dir", "checkpoints")
    ckpt_interval = max(1, getattr(scfg, "ckpt_interval", 5))
    
    # Artifact 히스토리 추적
    artifact_history = {
        'best_uploaded': False,
        'last_periodic_epoch': -1,
        'total_artifacts': 0
    }

    for epoch in range(EPOCHS):
        train_total, train_logs, global_step = train_epoch(
            train_loader, model, optim, device, scaler,
            desc=f"train[{epoch}]", loss_fn=physics_loss_fn,
            global_step_start=global_step,
            scheduler=scheduler,
            scheduler_step_mode=("step" if getattr(scfg, "scheduler_step_per_batch", False) else "epoch"),
            log_every_n_steps=getattr(scfg, "log_every_n_steps", 25)
        )

        val_total, val_logs = run_epoch(val_loader, model, device, loss_fn=physics_loss_fn)

        # === wandb epoch-level logging (항상 수행) ===
        log_epoch = {
            "epoch": epoch,
            "train/total_epoch": train_logs['total_loss'],
            "train/mse_epoch": train_logs['mse_loss'],
            "train/continuity_epoch": train_logs.get('continuity_loss', float('nan')),
            "train/momentum_epoch": train_logs.get('momentum_loss', float('nan')),
            "train/bc_epoch": train_logs.get('bc_loss', float('nan')),
            "val/total_epoch": val_logs['total_loss'],
            "val/mse_epoch": val_logs['mse_loss'],
            "val/continuity_epoch": val_logs.get('continuity_loss', float('nan')),
            "val/momentum_epoch": val_logs.get('momentum_loss', float('nan')),
            "val/bc_epoch": val_logs.get('bc_loss', float('nan')),
        }
        for cn in _CH_NAMES:
            log_epoch[f"train/mse_{cn}_epoch"] = train_logs.get(f'mse_{cn}', float('nan'))
            log_epoch[f"val/mse_{cn}_epoch"] = val_logs.get(f'mse_{cn}', float('nan'))
        if 'cont_weight_used' in train_logs: 
            log_epoch["weight/cont_used_epoch"] = train_logs['cont_weight_used']
        if 'mom_weight_used' in train_logs: 
            log_epoch["weight/mom_used_epoch"] = train_logs['mom_weight_used']
        lr_now = get_lr(optim)
        if lr_now is not None:
            log_epoch["lr_epoch"] = lr_now
        wandb.log(log_epoch, step=global_step, commit=True)

        # === Learning Rate Scheduler Step ===
        if scheduler is not None and not getattr(scfg, "scheduler_step_per_batch", False):
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_total)
            else:
                scheduler.step()

        # === Checkpoint 저장 (로컬 파일시스템) ===
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Best model 저장
        is_best = val_total < best_val
        if is_best:
            best_val = val_total
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
                "val_logs": val_logs
            }, best_path)
            
            # W&B Artifact 업로드 (조건부)
            if USE_WANDB_ARTIFACTS:
                try:
                    # 이전 best artifact가 있으면 삭제 (선택적)
                    if artifact_history['best_uploaded']:
                        # W&B는 자동으로 버전 관리하므로 별도 삭제 불필요
                        pass
                    
                    # 새로운 best artifact 업로드
                    art = wandb.Artifact(
                        name=f"model-best",  # 고정된 이름 사용
                        type="model",
                        description=f"Best model at epoch {epoch} with val_loss={val_total:.4f}",
                        metadata={
                            "epoch": epoch,
                            "val_loss": val_total,
                            "train_loss": train_total,
                            "best_val": best_val
                        }
                    )
                    art.add_file(best_path)
                    wandb.run.log_artifact(art)
                    artifact_history['best_uploaded'] = True
                    artifact_history['total_artifacts'] += 1
                    print(f"  📤 W&B Artifact uploaded: best model (epoch {epoch})")
                except Exception as e:
                    print(f"  ⚠️ Failed to upload W&B artifact: {e}")

        # Periodic checkpoint 저장
        if (epoch + 1) % ckpt_interval == 0:
            ep_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val
            }, ep_path)
            
            # Periodic artifact 업로드 (매우 제한적으로)
            if USE_WANDB_ARTIFACTS and not ARTIFACT_SAVE_BEST_ONLY:
                if (epoch + 1) % ARTIFACT_SAVE_INTERVAL == 0:
                    try:
                        art = wandb.Artifact(
                            name=f"model-checkpoint",
                            type="model",
                            description=f"Checkpoint at epoch {epoch+1}",
                            metadata={
                                "epoch": epoch + 1,
                                "val_loss": val_total,
                                "train_loss": train_total
                            }
                        )
                        art.add_file(ep_path)
                        wandb.run.log_artifact(art, aliases=[f"epoch-{epoch+1}"])
                        artifact_history['last_periodic_epoch'] = epoch + 1
                        artifact_history['total_artifacts'] += 1
                        print(f"  📤 W&B Artifact uploaded: checkpoint (epoch {epoch+1})")
                    except Exception as e:
                        print(f"  ⚠️ Failed to upload periodic artifact: {e}")

        # Print epoch summary
        print(f"Epoch {epoch:3d} | Train: total={train_total:.4f} mse={train_logs['mse_loss']:.4f} "
              f"cont={train_logs.get('continuity_loss', 0):.2e} mom={train_logs.get('momentum_loss', 0):.2e} "
              f"bc={train_logs.get('bc_loss', 0):.2e} | "
              f"Val: total={val_total:.4f} bc={val_logs.get('bc_loss', 0):.2e}"
              f" {'[BEST]' if is_best else ''}")

    # === Training 완료 후 최종 artifact ===
    if USE_WANDB_ARTIFACTS:
        # 최종 모델 저장
        final_path = os.path.join(ckpt_dir, "final.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": (scheduler.state_dict() if scheduler is not None else None),
            "scaler": (scaler.state_dict() if scaler is not None else None),
            "epoch": EPOCHS - 1,
            "global_step": global_step,
            "best_val": best_val
        }, final_path)
        
        try:
            art = wandb.Artifact(
                name=f"model-final",
                type="model",
                description=f"Final model after {EPOCHS} epochs",
                metadata={
                    "total_epochs": EPOCHS,
                    "best_val": best_val,
                    "total_artifacts": artifact_history['total_artifacts']
                }
            )
            art.add_file(final_path)
            art.add_file(os.path.join(ckpt_dir, "best.pt"), name="best.pt")  # best도 포함
            wandb.run.log_artifact(art, aliases=["latest", "final"])
            print(f"\n📤 Final W&B Artifact uploaded with {artifact_history['total_artifacts']+1} total artifacts")
        except Exception as e:
            print(f"⚠️ Failed to upload final artifact: {e}")

    # 종료
    wandb.finish()
    
    return {
        'lr_history': [],
        'train_total_loss': [],
        'train_continuity_loss': [],
        'train_bc_loss': [],
        'val_total_loss': [],
        'val_continuity_loss': [],
        'val_bc_loss': [],
        'artifacts_uploaded': artifact_history['total_artifacts']
    }

# 훈련 실행 함수 (Physics Loss 지원)
def run_training_experiment(config_updates=None, physics_config_updates=None):
    """
    설정을 업데이트하고 physics loss와 함께 훈련을 실행합니다.
    
    Args:
        config_updates: 모델 설정 변경 딕셔너리 (예: {'lr_scheduler': 'cosine'})
        physics_config_updates: Physics loss 설정 변경 딕셔너리
    """
    
    # 설정 업데이트
    if config_updates:
        for key, value in config_updates.items():
            setattr(scfg, key, value)
        print(f"🔧 Model configuration updated: {config_updates}")
    


    # 새로운 optimizer와 scheduler 생성
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=scfg.lr, 
                                 weight_decay=scfg.weight_decay, 
                                 betas=scfg.betas, 
                                 eps=scfg.eps)
    
    scheduler = create_lr_scheduler(optimizer, scfg)
    
    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler() if scfg.amp and torch.cuda.is_available() else None
    
    # 훈련 실행
    history = train_with_scheduler(
        model, optimizer, scheduler, train_loader, val_loader, 
        scfg, device, scaler, physics_loss_fn=loss_fn
    )
    
    return history


# 스케줄러 및 Physics Loss 테스트
print(f"Current scheduler: {scfg.lr_scheduler}")

if lr_scheduler is None:
    print("⚠️  No scheduler active - training with constant LR")
else:
    print(f"✅ Scheduler ready: {type(lr_scheduler).__name__}")

print(f"\n🧪 Running test training with physics loss...")

# 짧은 훈련으로 테스트 (physics loss 포함)
test_history = run_training_experiment()

print(f"\n📊 Test Results with Physics Loss:")
if test_history['lr_history']:
    print(f"   LR progression: {' -> '.join([f'{lr:.2e}' for lr in test_history['lr_history']])}")
    print(f"   Train losses (total): {' -> '.join([f'{loss:.4f}' for loss in test_history['train_total_loss']])}")
    print(f"   Train losses (continuity): {' -> '.join([f'{loss:.2e}' for loss in test_history['train_continuity_loss']])}")
    print(f"   Val losses (total): {' -> '.join([f'{loss:.4f}' for loss in test_history['val_total_loss']])}")
    print(f"   Val losses (continuity): {' -> '.join([f'{loss:.2e}' for loss in test_history['val_continuity_loss']])}")
else:
    print("   Training was interrupted before any results could be recorded.")
