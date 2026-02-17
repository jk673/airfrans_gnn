"""Data loading, normalization, and dataset classes for AirfRANS GNN training."""

from __future__ import annotations

import math
import os
import glob
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import BaseTransform

from .config import SmokeCfg
from .preprocessing import prepare_airfrans_graph_for_physics, build_bc_masks_airfrans
from .utils import prep_graph, validate_edges, _prep_graph_for_norm


PREBUILT_EDGES_DIR = Path(__file__).resolve().parents[1] / "prebuilt_edges_v2"


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

class StandardScaler:
    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, t: torch.Tensor):
        self.mean = t.mean(dim=0)
        self.std = t.std(dim=0).clamp_min(1e-8)
        return self

    def transform(self, t: torch.Tensor):
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fitted before calling transform.")
        return (t - self.mean) / self.std

    def inverse(self, t: torch.Tensor):
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fitted before calling inverse.")
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

        # Attach norm params so physics loss can denormalize properly
        dm.y_norm_params = {
            'mean': self.y_scaler.mean.clone(),
            'scale': self.y_scaler.std.clone(),
        }
        dm.x_norm_params = {
            'mean': self.x_scaler.mean.clone(),
            'scale': self.x_scaler.std.clone(),
        }

        # Ensure edge_attr_dxdy is present (needed for physics loss)
        if hasattr(d, 'edge_attr_dxdy'):
            dm.edge_attr_dxdy = d.edge_attr_dxdy
        elif hasattr(d, 'edge_attr'):
            if d.edge_attr.shape[1] >= 2:
                dm.edge_attr_dxdy = d.edge_attr[:, -2:]
            dm.edge_attr = d.edge_attr

        # Build BC masks from RAW (unnormalized) features
        d_raw = d.clone()
        if hasattr(dm, 'edge_index'):
            d_raw.edge_index = dm.edge_index
        if hasattr(dm, 'edge_attr_dxdy'):
            d_raw.edge_attr_dxdy = dm.edge_attr_dxdy
        elif hasattr(dm, 'edge_attr'):
            d_raw.edge_attr = dm.edge_attr
        d_raw = build_bc_masks_airfrans(d_raw)

        for attr in ['is_wall', 'is_inlet', 'is_outlet', 'is_farfield', 'inlet_u', 'wall_normal']:
            if hasattr(d_raw, attr):
                setattr(dm, attr, getattr(d_raw, attr))

        return dm


# ---------------------------------------------------------------------------
# PreparePhysics transform
# ---------------------------------------------------------------------------

class PreparePhysics(BaseTransform):
    def forward(self, data):
        return prepare_airfrans_graph_for_physics(data, verbose=False)


# ---------------------------------------------------------------------------
# Edge feature enrichment (5D → 10D at load time)
# ---------------------------------------------------------------------------

def enrich_edge_features(data: Data) -> Data:
    """Append 5 additional edge features to existing edge_attr (5D → 10D).

    New features: log_dist, edge_angle, relative_sdf, min_sdf, has_boundary_node.
    Called AFTER _prep_graph_for_norm. x[:, 2] = sdf in both 5D and 7D x.
    """
    ea = data.edge_attr   # [E, 5]: [dist, dir_x, dir_y, cos_n, is_surface_pair]
    ei = data.edge_index   # [2, E]
    x = data.x             # [N, 5+]: raw [u_inf, v_inf, sdf, nx, ny]

    if ea is None or ei is None or x is None:
        return data
    if ea.size(1) != 5:
        return data  # already enriched or unexpected schema

    row, col = ei
    dist = ea[:, 0:1]  # [E, 1]

    # 1. log_dist — scale-aware distance
    log_dist = torch.log(dist + 1e-8)

    # 2. edge_angle — edge orientation normalized to [-1, 1]
    dvec = ea[:, 1:3] * dist  # recover [dx, dy] from dir_xy * dist
    edge_angle = (torch.atan2(dvec[:, 1], dvec[:, 0]) / math.pi).unsqueeze(1)

    # 3. relative_sdf — wall distance gradient along edge
    sdf = x[:, 2]
    relative_sdf = ((sdf[col] - sdf[row]) / (dist.squeeze(1) + 1e-8)).unsqueeze(1)

    # 4. min_sdf — boundary proximity
    min_sdf = torch.minimum(sdf[row], sdf[col]).unsqueeze(1)

    # 5. has_boundary_node — at least one endpoint on surface
    from src.utils import get_surface_mask
    surf_mask = get_surface_mask(data)
    has_bnd = (surf_mask[row] | surf_mask[col]).float().unsqueeze(1)

    data.edge_attr = torch.cat([ea, log_dist, edge_angle, relative_sdf, min_sdf, has_bnd], dim=1)
    return data


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
    edge_dim: int = 5


# ---------------------------------------------------------------------------
# collate_pyg
# ---------------------------------------------------------------------------

def collate_pyg(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Extract norm_params before batching (PyG can mis-collate dicts)
    norm_params = {}
    for key in ('y_norm_params', 'x_norm_params'):
        if hasattr(batch[0], key):
            norm_params[key] = getattr(batch[0], key)

    batched = Batch.from_data_list(batch)

    # Reattach norm_params as single shared dicts
    for key, value in norm_params.items():
        setattr(batched, key, value)

    return batched


# ---------------------------------------------------------------------------
# load_and_prepare_data
# ---------------------------------------------------------------------------

def load_and_prepare_data(scfg: SmokeCfg) -> DataBundle:
    """Load AirfRANS data, build edges, normalize, and return DataBundle."""
    prebuilt_root = PREBUILT_EDGES_DIR / scfg.task
    prebuilt_train_dir = prebuilt_root / "train"
    prebuilt_test_dir = prebuilt_root / "test"

    # Load prebuilt edge graphs
    train_edge_files = sorted(glob.glob(os.path.join(str(prebuilt_train_dir), 'graph_*.pt')))
    val_edge_files = sorted(glob.glob(os.path.join(str(prebuilt_test_dir), 'graph_*.pt')))
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

    train_prepped = [enrich_edge_features(_prep_graph_for_norm(g)) for g in train_edges_subset]
    val_prepped = [enrich_edge_features(_prep_graph_for_norm(g)) for g in val_edges_subset] if isinstance(val_edges_subset, list) else []

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

    edge_dim = train_prepped[0].edge_attr.shape[1] if train_prepped and hasattr(train_prepped[0], 'edge_attr') and train_prepped[0].edge_attr is not None else 5

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        train_graphs=train_edges,
        val_graphs=val_edges,
        train_norm=train_norm,
        val_norm=val_norm,
        edge_dim=edge_dim,
    )
