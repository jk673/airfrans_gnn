"""
Metrics module for AirfRANS GNN evaluation.

Provides functions for computing:
- Per-channel MSE (all, surface, volume)
- Pressure coefficient (Cp) from p/ρ
- Relative L2 error
- Surface masking from physical features
- Force coefficients (lift/drag) comparison
"""

import math
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch_geometric.data import Data


# ============================================================================
# Cp and Relative L2 Error (from 03_eval_cp_relative_l2.ipynb)
# ============================================================================

def surface_mask_from_x_phys(x_phys: torch.Tensor) -> torch.Tensor:
    """
    Extract surface mask from physical node features.

    Surface nodes are identified by:
    - Zero wall distance (x_phys[:, 2] < 1e-6), OR
    - Non-zero wall normals (x_phys[:, 3:5])

    Args:
        x_phys: Physical node features [N, F] where:
            - Column 2 is wall distance
            - Columns 3-4 are wall normals (nx, ny)

    Returns:
        Boolean mask [N] where True indicates surface nodes
    """
    if x_phys.dim() != 2:
        raise ValueError(f"x_phys must be 2D [N,F], got {tuple(x_phys.shape)}")

    if x_phys.size(1) >= 5:
        wall = x_phys[:, 2]
        nxy = x_phys[:, 3:5]
        return (wall < 1e-6) | (nxy.abs().sum(dim=1) > 0)

    if x_phys.size(1) >= 3:
        wall = x_phys[:, 2]
        return (wall < 1e-6)

    return torch.zeros(x_phys.size(0), dtype=torch.bool)


def cp_from_p_over_rho(
    y_phys: torch.Tensor,
    x_phys: torch.Tensor,
    *,
    p_channel: int = 2,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Compute pressure coefficient Cp from p/ρ.

    Cp = (p/ρ) / q_ref, where q_ref = 0.5 * ||U_inf||²

    Uses freestream velocity from normalized x_phys (first two columns).

    Args:
        y_phys: Physical target values [N, C] with p/ρ at column p_channel
        x_phys: Physical node features [N, F] with (u_inf, v_inf) at columns 0-1
        p_channel: Index of p/ρ in y_phys (default: 2)
        eps: Small value to prevent division by zero

    Returns:
        Pressure coefficient [N]
    """
    if y_phys.dim() != 2 or x_phys.dim() != 2:
        raise ValueError("y_phys and x_phys must be [N, C] and [N, F]")
    if x_phys.size(1) < 2:
        raise ValueError("x_phys must have at least 2 columns for (u_inf, v_inf)")

    u_inf = x_phys[0, 0]
    v_inf = x_phys[0, 1]
    q = 0.5 * (u_inf * u_inf + v_inf * v_inf)
    q = q.clamp_min(eps)

    return y_phys[:, p_channel] / q


def relative_l2(pred: torch.Tensor, targ: torch.Tensor, *, eps: float = 1e-12) -> float:
    """
    Compute relative L2 error: ||pred - targ|| / ||targ||.

    Args:
        pred: Predicted values [N] or [N, C]
        targ: Target values (same shape as pred)
        eps: Small value to prevent division by zero

    Returns:
        Relative L2 error as float
    """
    pred = pred.reshape(-1)
    targ = targ.reshape(-1)

    if pred.numel() == 0:
        return float("nan")

    num = torch.linalg.norm(pred - targ)
    den = torch.linalg.norm(targ).clamp_min(eps)

    return float((num / den).item())


# ============================================================================
# Per-Channel MSE (from 01_trainer.ipynb)
# ============================================================================

def mse_per_channel(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> List[float]:
    """
    Compute per-channel mean squared error.

    Computes MSE independently for each output channel (u, v, p/ρ, ν_t).

    Args:
        y_pred: Predicted values [N, C]
        y_true: Ground truth values [N, C]
        mask: Optional boolean mask [N] to select a subset of nodes

    Returns:
        List of MSE values, one per channel
    """
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
        return [float("nan")] * y_true.size(1)

    return [float(err[:, i].mean().item()) for i in range(y_true.size(1))]


# ============================================================================
# Surface Mask from Original (Unnormalized) Features
# ============================================================================

def surface_volume_masks_from_orig(data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract surface and volume masks from original node features.

    Surface: nodes with non-zero wall normals (x[:, 3:5])
    Volume: remaining nodes

    Args:
        data: PyG Data object with x features

    Returns:
        Tuple of (surface_mask, volume_mask), both boolean [N]
    """
    x = data.x
    if x is not None and x.size(1) >= 5:
        nxy = x[:, 3:5]
        surf = (nxy.abs().sum(dim=1) > 0)
    else:
        # Fallback: assume columns 2 is wall distance
        if x is not None and x.size(1) >= 3:
            wall_dist = x[:, 2]
            surf = (wall_dist < 1e-6)
        else:
            surf = torch.zeros(x.size(0), dtype=torch.bool)

    vol = ~surf
    return surf, vol


# ============================================================================
# Force Coefficients (from src/force_coefficients.py + 01_trainer.ipynb)
# ============================================================================

def order_surface(pos_surf: torch.Tensor) -> torch.Tensor:
    """
    Order surface nodes by angle around centroid.

    Args:
        pos_surf: Surface node positions [n_surf, 2]

    Returns:
        Indices that sort surface nodes counterclockwise
    """
    c = pos_surf.mean(dim=0)
    rel = pos_surf - c
    angles = torch.atan2(rel[:, 1], rel[:, 0])
    return torch.argsort(angles)


def panel_lengths(pos_ordered: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute panel lengths and segment vectors for surface panels.

    Args:
        pos_ordered: Ordered surface positions [n_surf, 2]

    Returns:
        Tuple of (ds, seg_vec):
            - ds: Panel lengths [n_surf]
            - seg_vec: Segment vectors [n_surf, 2]
    """
    rolled = torch.roll(pos_ordered, -1, 0)
    seg = rolled - pos_ordered
    ds = seg.norm(dim=1)
    return ds, seg


def compute_force_coefficients(
    data: Data,
    eps: float = 1e-9
) -> Optional[Dict[str, float]]:
    """
    Compute drag (CD) and lift (CL) coefficients from surface pressure.

    Method:
    1. Extract surface nodes (non-zero wall normals)
    2. Order surface nodes around airfoil centroid
    3. Compute pressure force panels: F_p = -p·n_hat·ds
    4. Normalize by dynamic pressure and chord length

    Args:
        data: PyG Data object with:
            - x: Node features [N, ≥5] with u_inf, v_inf, wall_dist, nx, ny
            - y: Target values [N, ≥3] with p/ρ at column 2
            - pos: Node positions [N, 2] (x, y coordinates)
        eps: Small value to prevent division by zero

    Returns:
        Dict with 'CD' and 'CL' keys if computation succeeds, None otherwise
    """
    xvars = torch.tensor(data.x) if not isinstance(data.x, torch.Tensor) else data.x.cpu()
    yvars = torch.tensor(data.y) if not isinstance(data.y, torch.Tensor) else data.y.cpu()

    if data.pos is not None:
        pos = torch.tensor(data.pos) if not isinstance(data.pos, torch.Tensor) else data.pos.cpu()
    else:
        # Fallback: construct pseudo-positions
        N = xvars.size(0)
        pos = torch.zeros(N, 2)
        pos[:, 0] = torch.linspace(0, 1, N)
        pos[:, 1] = 0

    # Surface mask: wall normals are non-zero (columns 3-4)
    normals = xvars[:, 3:5]
    surf_mask = (normals.abs().sum(dim=1) > 0)
    surf_idx = torch.nonzero(surf_mask, as_tuple=False).squeeze(-1).long()

    if surf_idx.numel() < 5:
        return None

    # Order surface nodes
    pos_surf = pos[surf_idx]
    ord_idx = order_surface(pos_surf).long()
    ordered_pos = pos_surf[ord_idx]
    ds, seg_vec = panel_lengths(ordered_pos)

    # Normalize surface normals
    n_nodes = normals[surf_idx[ord_idx]]
    n_norm = n_nodes.norm(dim=1, keepdim=True).clamp_min(eps)
    n_hat = n_nodes / n_norm

    # Freestream velocity
    if hasattr(data, 'x_norm_params'):
        # Data is in normalized space; denormalize first
        mean = data.x_norm_params['mean']
        scale = data.x_norm_params['scale']
        u_inf = xvars[0, 0] * scale[0] + mean[0]
        v_inf = xvars[0, 1] * scale[1] + mean[1]
    else:
        # Data is in physical space
        u_inf = xvars[0, 0]
        v_inf = xvars[0, 1]

    U_inf = float(torch.sqrt(u_inf**2 + v_inf**2)) + eps

    # Pressure force
    temp = torch.index_select(yvars, 0, surf_idx)[:, 2]
    p_over_rho = temp[ord_idx]
    Fp = (-p_over_rho.unsqueeze(1) * n_hat) * ds.unsqueeze(1)
    F = Fp.sum(dim=0)  # Total force [Fx, Fy]

    # Chord length
    chord = float(pos_surf[:, 0].max() - pos_surf[:, 0].min() + eps)

    # Dynamic pressure reference
    q_ref = 0.5 * U_inf**2

    # Force coefficients (sign convention: drag opposes flow, lift perpendicular)
    CD = -F[0] / (q_ref * chord + eps)
    CL = F[1] / (q_ref * chord + eps)

    return {'CD': float(CD), 'CL': float(CL)}


def compare_force_coefficients(
    d: Data,
    y_pred_denorm: torch.Tensor,
    y_scaler
) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """
    Compare force coefficients from predicted vs ground truth pressures.

    Args:
        d: PyG Data object with ground truth data (already denormalized)
        y_pred_denorm: Predicted targets in physical space [N, C]
        y_scaler: Scaler object (for reference; data assumed already denormalized)

    Returns:
        Tuple of (coeffs_gt, coeffs_pred) dicts, or None if computation fails
    """
    # Ground truth coefficients
    coeffs_gt = compute_force_coefficients(d)

    # Predicted coefficients
    d_pred = Data(**{k: v for k, v in d})
    d_pred.y = y_pred_denorm
    coeffs_pred = compute_force_coefficients(d_pred)

    if coeffs_gt is None or coeffs_pred is None:
        return None

    return coeffs_gt, coeffs_pred


# ============================================================================
# Aggregated Metrics (for validation/test sets)
# ============================================================================

def aggregate_mse_stats(
    mse_list: List[List[float]],
    channel_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-channel MSE across multiple graphs.

    Args:
        mse_list: List of MSE lists, each with values for all channels
        channel_names: Optional names for channels (default: ['u', 'v', 'p', 'nu_t'])

    Returns:
        Dict with stats per channel (mean, std, median)
    """
    if channel_names is None:
        channel_names = ['u', 'v', 'p', 'nu_t']

    if not mse_list:
        return {}

    mse_array = np.array(mse_list, dtype=np.float64)
    stats = {}

    for i, name in enumerate(channel_names):
        if i < mse_array.shape[1]:
            vals = mse_array[:, i]
            stats[name] = {
                'mean': float(np.nanmean(vals)),
                'std': float(np.nanstd(vals)),
                'median': float(np.nanmedian(vals)),
                'min': float(np.nanmin(vals)),
                'max': float(np.nanmax(vals)),
            }

    return stats


def aggregate_cp_relative_l2_stats(
    rel_l2_list: List[float],
    sum_err2: float = None,
    sum_gt2: float = None
) -> Dict[str, float]:
    """
    Aggregate Cp relative L2 errors across graphs.

    Args:
        rel_l2_list: List of per-graph relative L2 errors
        sum_err2: Optional cumulative sum of squared errors (for global metric)
        sum_gt2: Optional cumulative sum of squared targets (for global metric)

    Returns:
        Dict with per-graph and global statistics
    """
    rel_l2_array = np.asarray(rel_l2_list, dtype=np.float64)
    finite_mask = np.isfinite(rel_l2_array)

    stats = {
        'per_graph_mean': float(np.nanmean(rel_l2_array)),
        'per_graph_median': float(np.nanmedian(rel_l2_array)),
        'per_graph_std': float(np.nanstd(rel_l2_array)),
        'per_graph_min': float(np.nanmin(rel_l2_array)),
        'per_graph_max': float(np.nanmax(rel_l2_array)),
        'finite_graphs': int(np.sum(finite_mask)),
        'total_graphs': len(rel_l2_array),
    }

    if sum_err2 is not None and sum_gt2 is not None:
        eps = 1e-12
        global_rel_l2 = math.sqrt(sum_err2 / (sum_gt2 + eps))
        stats['global_relative_l2'] = float(global_rel_l2)

    return stats
