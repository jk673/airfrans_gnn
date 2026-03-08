"""Data loading, normalization, graph utilities, and BC preprocessing for AirfRANS GNN."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch_scatter import scatter_add
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch


PREBUILT_EDGES_DIR = Path(__file__).resolve().parents[1] / "Dataset" / "processed_data" / "prebuilt_edges"


# ---------------------------------------------------------------------------
# Graph utilities (merged from utils.py)
# ---------------------------------------------------------------------------

def _valid_edges(edge_index: torch.Tensor, N: int) -> torch.Tensor:
    """Return boolean mask for edges within valid node range [0, N)."""
    row, col = edge_index
    return (row >= 0) & (row < N) & (col >= 0) & (col < N)


def with_pos2(data):
    """Append 2D position into node features x to make 7D (orig 5 + pos2)."""
    x_orig = data.x
    if hasattr(data, 'pos') and data.pos is not None:
        pos = data.pos
    elif hasattr(data, 'x_norm_params') and data.x_norm_params is not None:
        mean = torch.as_tensor(data.x_norm_params['mean'][:3], dtype=x_orig.dtype, device=x_orig.device)
        scale = torch.as_tensor(data.x_norm_params['scale'][:3], dtype=x_orig.dtype, device=x_orig.device)
        pos = x_orig[:, :3] * scale + mean
    else:
        pos = x_orig[:, :3]
    new = data.clone()
    new.x = torch.cat([x_orig, pos[:, :2]], dim=1)
    new.pos2_appended = True
    return new


def get_surface_mask(d):
    if hasattr(d, 'surf') and isinstance(d.surf, torch.Tensor) and d.surf.dtype == torch.bool:
        return d.surf.view(-1)
    x = d.x
    if x is not None and x.size(1) >= 5:
        return (x[:, 2] < 1e-6) | (x[:, 3:5].abs().sum(dim=1) > 0)
    elif x is not None and x.size(1) >= 3:
        return x[:, 2] < 1e-6
    return torch.zeros(d.x.size(0), dtype=torch.bool, device=d.x.device)


def ensure_edge_features(d, want_dim: int = 5):
    if hasattr(d, 'edge_attr') and d.edge_attr is not None and d.edge_attr.size(1) == want_dim:
        return d
    if not hasattr(d, 'edge_index') or d.edge_index is None or d.edge_index.numel() == 0:
        return d
    row, col = d.edge_index
    pos = d.pos if hasattr(d, 'pos') and d.pos is not None else d.x[:, :3]
    dvec = pos[col, :2] - pos[row, :2]
    dist = dvec.norm(dim=1, keepdim=True)
    dir_xy = dvec / (dist + 1e-8)
    nxy = d.x[:, 3:5] if d.x.size(1) >= 5 else torch.zeros(d.x.size(0), 2, device=d.x.device, dtype=d.x.dtype)
    cos_n = (dir_xy * (nxy[row] + nxy[col]) * 0.5).sum(dim=1, keepdim=True)
    surf = get_surface_mask(d).to(torch.float32)
    surf_pair = torch.stack([(surf[row] > 0.5) & (surf[col] > 0.5),
                             (surf[row] > 0.5) ^ (surf[col] > 0.5)], dim=1).to(dist.dtype)
    d.edge_attr = torch.cat([dist, dir_xy, cos_n, surf_pair], dim=1)
    return d


def prep_graph(g):
    if g.x is None:
        raise ValueError('Graph missing node features x.')
    return ensure_edge_features(g, want_dim=5)


def validate_edges(ds, name='train'):
    bad = 0
    for i, d in enumerate(ds):
        ei, ea, N = getattr(d, 'edge_index', None), getattr(d, 'edge_attr', None), d.x.size(0)
        if ei is None or ea is None:
            bad += 1; print(f'[{name}] {i}: missing edges'); continue
        if ei.dtype != torch.long or ei.size(0) != 2 or ei.size(1) == 0:
            bad += 1; print(f'[{name}] {i}: bad edge_index shape {tuple(ei.shape)}')
        if int(ei.min()) < 0 or int(ei.max()) >= N:
            bad += 1; print(f'[{name}] {i}: edge_index out of range')
        if ea.dim() != 2 or ea.size(0) != ei.size(1):
            bad += 1; print(f'[{name}] {i}: edge_attr shape mismatch')
    print(f'[validate] {name}: total={len(ds)} bad={bad}')


def _prep_graph_for_norm(g):
    d = g.clone()
    try:
        if d.x is not None and d.x.size(1) == 5 and not getattr(d, 'pos2_appended', False):
            d = with_pos2(d)
    except Exception:
        pass
    return ensure_edge_features(d, want_dim=5)


# ---------------------------------------------------------------------------
# BC masks and node area (merged from physics_prep.py)
# ---------------------------------------------------------------------------

def _half_edges(edge_index, edge_attr):
    """Keep only one direction (row < col) for undirected graphs."""
    row, col = edge_index
    mask = row < col
    if mask.any():
        edge_index = edge_index[:, mask]
        if edge_attr is None:
            raise ValueError("_half_edges requires edge_attr")
        edge_attr = edge_attr[mask]
    if edge_attr is None:
        raise ValueError("_half_edges requires non-None edge_attr")
    return edge_index, edge_attr


def _extract_dxdy_length(edge_index, edge_attr, pos, prefer_dxdy=True, eps=1e-12):
    """Return dx, dy, length per edge from edge_attr or pos."""
    if prefer_dxdy and edge_attr.size(1) >= 3:
        col1_med = edge_attr[:, 1].abs().median()
        if col1_med <= 1.5 and torch.all(edge_attr[:, 0] > 0):
            length = edge_attr[:, 0].clamp_min(eps)
            return edge_attr[:, 1] * length, edge_attr[:, 2] * length, length
        return edge_attr[:, 0], edge_attr[:, 1], edge_attr[:, 2].abs().clamp_min(eps)
    if pos is None:
        raise ValueError("Need pos to compute dx,dy")
    row, col = edge_index
    dvec = pos[col, :2] - pos[row, :2]
    length = dvec.norm(dim=1).clamp_min(eps)
    return dvec[:, 0], dvec[:, 1], length


@torch.no_grad()
def estimate_node_area(data):
    """Create data.node_area from triangle faces or perimeter approximation."""
    if not hasattr(data, 'pos') or data.pos is None:
        return data
    device = data.pos.device
    N = getattr(data, 'num_nodes', data.pos.size(0))

    if getattr(data, 'face', None) is not None and data.face.numel() > 0:
        f = data.face.to(torch.long).to(device)
        v0, v1, v2 = data.pos[f[0], :2], data.pos[f[1], :2], data.pos[f[2], :2]
        area_tri = 0.5 * torch.abs((v1 - v0)[:, 0] * (v2 - v0)[:, 1] -
                                    (v1 - v0)[:, 1] * (v2 - v0)[:, 0])
        area_node = torch.zeros(N, device=device)
        for k in range(3):
            area_node.index_add_(0, f[k], area_tri / 3.0)
        data.node_area = area_node.clamp_min(1e-12)
        return data

    if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
        row, col = data.edge_index
        length = (data.pos[col, :2] - data.pos[row, :2]).norm(dim=1)
        per = scatter_add(length, row, dim=0, dim_size=N) + scatter_add(length, col, dim=0, dim_size=N)
        data.node_area = ((per * per) / (4.0 * math.pi)).clamp_min(1e-12)
    else:
        data.node_area = torch.ones(N, device=device)
    return data


def _weighted_gradient(field, edge_index, edge_attr, num_nodes, *, pos=None, eps=1e-12):
    """Compute RBF-weighted gradient for wall normal recovery."""
    device = field.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    edge_attr = edge_attr.to(device=device, dtype=field.dtype)
    if pos is not None:
        pos = pos.to(device=device, dtype=field.dtype)

    N = field.size(0)
    z = torch.zeros(N, device=device, dtype=field.dtype)
    if edge_index.numel() == 0 or edge_attr.numel() == 0:
        return z, z.clone()

    valid = _valid_edges(edge_index, N)
    if not torch.all(valid):
        edge_index, edge_attr = edge_index[:, valid], edge_attr[valid]
        if edge_index.numel() == 0:
            return z, z.clone()

    edge_index, edge_attr = _half_edges(edge_index, edge_attr)
    if edge_index is None or edge_index.numel() == 0:
        return z, z.clone()

    row, col = edge_index
    dx, dy, length = _extract_dxdy_length(edge_index, edge_attr, pos, True, eps)

    h2 = (length.mean() ** 2).clamp_min(eps)
    w = torch.exp(-(length * length) / (h2 + eps))
    df = field[col] - field[row]
    inv_r2 = 1.0 / (length * length + eps)

    gx_e, gy_e = w * df * dx * inv_r2, w * df * dy * inv_r2
    num_x = scatter_add(gx_e, row, dim=0, dim_size=N) + scatter_add(gx_e, col, dim=0, dim_size=N)
    num_y = scatter_add(gy_e, row, dim=0, dim_size=N) + scatter_add(gy_e, col, dim=0, dim_size=N)
    den = (scatter_add(w, row, dim=0, dim_size=N) + scatter_add(w, col, dim=0, dim_size=N)).clamp_min(1.0)
    return num_x / den, num_y / den


@torch.no_grad()
def build_bc_masks_airfrans(data, *, wall_dist_thresh=1e-4):
    """Create BC masks: is_wall, is_inlet, is_outlet, is_farfield, inlet_u, wall_normal."""
    device = data.pos.device
    N = data.num_nodes
    X = data.x
    if X is None:
        raise ValueError("data.x is required")

    pos = data.pos[:, :2]
    xcoord, ycoord = pos[:, 0], pos[:, 1]

    # Freestream velocity
    U_inf = torch.zeros(N, 2, device=device)
    if X.size(1) >= 2:
        U_inf[:, 0], U_inf[:, 1] = X[:, 0], X[:, 1]
    else:
        U_inf[:, 0] = 1.0

    # Wall distance & normals
    wall_dist = X[:, 2].abs() if X.size(1) > 2 else torch.full((N,), 1e9, device=device)
    have_normals = False
    nx_ny = torch.zeros(N, 2, device=device)
    if X.size(1) >= 5:
        nx_ny = torch.stack([X[:, 3], X[:, 4]], dim=-1)
        have_normals = (nx_ny.abs().sum(dim=1) > 0).float().mean().item() > 0.0

    # is_wall
    surf = getattr(data, 'surf', None)
    if surf is not None and isinstance(surf, torch.Tensor) and surf.dtype == torch.bool and surf.numel() == N:
        is_wall = surf.view(-1).to(device)
    else:
        is_wall = (wall_dist <= wall_dist_thresh)
        if have_normals:
            near_thresh = min(wall_dist_thresh * 100.0, float(torch.quantile(wall_dist, 0.05)))
            candidate = is_wall | ((nx_ny.abs().sum(dim=1) > 0) & (wall_dist <= near_thresh))
            if candidate.float().mean().item() <= 0.8:
                is_wall = candidate
    data.is_wall = is_wall.to(torch.bool).to(device)

    # Wall normals
    wall_normal = torch.zeros(N, 2, device=device)
    if have_normals:
        wall_normal = nx_ny.clone()
    else:
        ea = getattr(data, 'edge_attr_dxdy', getattr(data, 'edge_attr', None))
        if ea is not None and hasattr(data, 'edge_index') and data.edge_index is not None:
            gx, gy = _weighted_gradient(wall_dist, data.edge_index, ea.to(device), N, pos=pos)
            g = torch.stack([gx, gy], dim=-1)
            wall_normal = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
    data.wall_normal = wall_normal

    # Inlet / outlet (x-quantile)
    q = 0.02
    is_inlet = (xcoord <= torch.quantile(xcoord, q)) & (~is_wall)
    is_outlet = (xcoord >= torch.quantile(xcoord, 1.0 - q)) & (~is_wall)

    # Farfield (outer box)
    ff = 0.10
    is_outer = ((xcoord <= torch.quantile(xcoord, ff)) | (xcoord >= torch.quantile(xcoord, 1 - ff)) |
                (ycoord <= torch.quantile(ycoord, ff)) | (ycoord >= torch.quantile(ycoord, 1 - ff)))
    is_farfield = is_outer & (~is_wall) & (~is_inlet) & (~is_outlet)

    data.is_inlet = is_inlet.to(torch.bool).to(device)
    data.is_outlet = is_outlet.to(torch.bool).to(device)
    data.is_farfield = is_farfield.to(torch.bool).to(device)

    inlet_u = torch.zeros(N, 2, device=device, dtype=U_inf.dtype)
    inlet_u[is_inlet] = U_inf[is_inlet]
    data.inlet_u = inlet_u
    return data


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
        dm.y = self.y_scaler.transform(d.y) if hasattr(d, 'y') and d.y is not None else d.y
        dm.has_norm = True

        dm.y_norm_params = {'mean': self.y_scaler.mean.clone(), 'scale': self.y_scaler.std.clone()}
        dm.x_norm_params = {'mean': self.x_scaler.mean.clone(), 'scale': self.x_scaler.std.clone()}

        # Preserve edge_attr_dxdy for physics loss
        if hasattr(d, 'edge_attr_dxdy'):
            dm.edge_attr_dxdy = d.edge_attr_dxdy
        elif hasattr(d, 'edge_attr') and d.edge_attr.shape[1] >= 2:
            dm.edge_attr_dxdy = d.edge_attr[:, -2:]
            dm.edge_attr = d.edge_attr

        # Build BC masks and node_area from RAW (unnormalized) features
        d_raw = d.clone()
        if hasattr(dm, 'edge_index'):
            d_raw.edge_index = dm.edge_index
        if hasattr(dm, 'edge_attr_dxdy'):
            d_raw.edge_attr_dxdy = dm.edge_attr_dxdy
        elif hasattr(dm, 'edge_attr'):
            d_raw.edge_attr = dm.edge_attr
        d_raw = build_bc_masks_airfrans(d_raw)
        d_raw = estimate_node_area(d_raw)

        for attr in ['is_wall', 'is_inlet', 'is_outlet', 'is_farfield', 'inlet_u', 'wall_normal', 'node_area']:
            if hasattr(d_raw, attr):
                setattr(dm, attr, getattr(d_raw, attr))

        return dm


# ---------------------------------------------------------------------------
# Edge feature enrichment (5D -> 10D at load time)
# ---------------------------------------------------------------------------

def enrich_edge_features(data: Data) -> Data:
    """Append 5 additional edge features to existing edge_attr (5D -> 10D)."""
    ea, ei, x = data.edge_attr, data.edge_index, data.x
    if ea is None or ei is None or x is None or ea.size(1) != 5:
        return data

    row, col = ei
    dist = ea[:, 0:1]

    log_dist = torch.log(dist + 1e-8)
    dvec = ea[:, 1:3] * dist
    edge_angle = (torch.atan2(dvec[:, 1], dvec[:, 0]) / math.pi).unsqueeze(1)
    sdf = x[:, 2]
    relative_sdf = ((sdf[col] - sdf[row]) / (dist.squeeze(1) + 1e-8)).unsqueeze(1)
    min_sdf = torch.minimum(sdf[row], sdf[col]).unsqueeze(1)
    surf_mask = get_surface_mask(data)
    has_bnd = (surf_mask[row] | surf_mask[col]).float().unsqueeze(1)

    data.edge_attr = torch.cat([ea, log_dist, edge_angle, relative_sdf, min_sdf, has_bnd], dim=1)
    return data


# ---------------------------------------------------------------------------
# DataBundle
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: object
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    train_graphs: list
    val_graphs: list
    train_norm: NormalizedDataset
    val_norm: object
    edge_dim: int = 5


# ---------------------------------------------------------------------------
# collate_pyg
# ---------------------------------------------------------------------------

def collate_pyg(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    norm_params = {}
    for key in ('y_norm_params', 'x_norm_params'):
        if hasattr(batch[0], key):
            norm_params[key] = getattr(batch[0], key)

    batched = Batch.from_data_list(batch)
    for key, value in norm_params.items():
        setattr(batched, key, value)
    return batched
