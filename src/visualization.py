"""
Visualization utilities for AirfRANS GNN predictions and analysis.

Provides functions for visualizing:
- Prediction vs ground truth comparisons (tricontour plots)
- Surface pressure distributions
- Surface node ordering and panel length calculations
"""

import os
from datetime import datetime
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.path import Path
from torch_geometric.data import Data

from src.utils import get_surface_mask


def _surface_volume_masks_from_orig(d: Data):
    """
    Extract surface and volume masks from node features.

    Surface nodes have non-zero wall normal vectors (features 3:5).
    Volume nodes have zero wall normal vectors.

    Args:
        d: PyG Data object with node features x

    Returns:
        surf: Boolean tensor indicating surface nodes
        vol: Boolean tensor indicating volume nodes
    """
    x = d.x
    if x is not None and x.size(1) >= 5:
        nxy = x[:, 3:5]
        surf = (nxy.abs().sum(dim=1) > 0)
    else:
        # Fallback to earlier heuristic
        surf = get_surface_mask(d)
    vol = ~surf
    return surf, vol


def order_surface(pos_surf: torch.Tensor):
    """
    Order surface nodes by angle around their centroid.

    Used to arrange surface nodes in a consistent order for plotting
    pressure distributions and computing force coefficients.

    Args:
        pos_surf: (N, 2) tensor of surface node positions [x, y]

    Returns:
        indices: (N,) tensor of indices that sort nodes by angle
    """
    c = pos_surf.mean(dim=0)
    rel = pos_surf - c
    angles = torch.atan2(rel[:, 1], rel[:, 0])
    return torch.argsort(angles)


def panel_lengths(pos_ordered: torch.Tensor):
    """
    Compute segment lengths and vectors for ordered surface nodes.

    Computes distances and direction vectors between consecutive
    nodes in an ordered surface discretization.

    Args:
        pos_ordered: (N, 2) tensor of ordered surface node positions

    Returns:
        ds: (N,) tensor of segment lengths (distances)
        seg: (N, 2) tensor of segment direction vectors
    """
    rolled = torch.roll(pos_ordered, -1, 0)
    seg = rolled - pos_ordered
    ds = seg.norm(dim=1)
    return ds, seg


@torch.no_grad()
def plot_pred_vs_gt(
    dm: Data,
    y_pred: torch.Tensor,
    channel: int = 2,
    show_mesh: bool = True,
    mask_airfoil: bool = True,
    titles: tuple[str, str, str] = ('Ground Truth', 'Prediction', 'Abs Error'),
    cmap: str = 'viridis',
    denormalize: bool = True,
    save_dir: str = "figures",
    save: bool = True,
    y_scaler=None,
    x_scaler=None,
):
    """
    Visualize prediction vs ground truth using tricontour plots.

    Creates a 3-panel comparison showing ground truth, prediction, and absolute error
    for a specified output channel (u, v, pressure, or turbulent viscosity).

    Args:
        dm: PyG Data object containing node features, positions, and ground truth
        y_pred: (N, 4) tensor of model predictions [u, v, Cp, nu_t]
        channel: Output channel to visualize (0=u, 1=v, 2=Cp, 3=nu_t). Default: 2 (pressure)
        show_mesh: If True, overlay triangulation mesh on plots. Default: True
        mask_airfoil: If True, mask interior of airfoil. Default: True
        titles: 3-tuple of subplot titles
        cmap: Colormap name for GT and prediction plots. Default: 'viridis'
        denormalize: If True, denormalize GT and prediction to physical units. Default: True
        save_dir: Directory to save figure. Default: "figures"
        save: If True, save figure to disk. Default: True
        y_scaler: StandardScaler for denormalizing output (y). If None, no denormalization.
        x_scaler: StandardScaler for denormalizing input (x). If None, no denormalization.

    Returns:
        None (displays and optionally saves figure)
    """
    # Optionally denormalize both GT and prediction for visualization
    if denormalize and hasattr(dm, 'y') and y_scaler is not None:
        dm_y_cpu = dm.y.detach().cpu()
        y_pred_cpu = y_pred.detach().cpu()
        gt = y_scaler.inverse(dm_y_cpu)
        pr = y_scaler.inverse(y_pred_cpu)
    else:
        gt = dm.y.detach().cpu()
        pr = y_pred.detach().cpu()

    # Prepare coordinates and values
    xy = (
        (dm.pos if hasattr(dm, 'pos') and dm.pos is not None else dm.x)[:, :2]
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    tri = Triangulation(xy[:, 0], xy[:, 1])

    # Denormalize x to physical for q and for robust airfoil polygon detection
    dm_x_cpu = dm.x.detach().cpu()
    if x_scaler is not None:
        x_phys = x_scaler.inverse(dm_x_cpu)
    else:
        x_phys = dm_x_cpu

    vx = float(x_phys[0, 0])
    vy = float(x_phys[0, 1])
    v = math.sqrt(vx * vx + vy * vy)
    q = 0.5 * v * v

    gt_c = gt[:, channel].detach().cpu().float().numpy() / q
    pr_c = pr[:, channel].detach().cpu().float().numpy() / q
    err = np.abs(pr_c - gt_c)
    vmin = -2.0
    vmax = 1.0

    # Optional mask of airfoil interior
    if mask_airfoil:
        try:
            x_np = (
                x_phys.detach().cpu().numpy()
                if isinstance(x_phys, torch.Tensor)
                else np.asarray(x_phys)
            )
            pos_np = (
                (dm.pos if hasattr(dm, 'pos') and dm.pos is not None else dm.x)[:, :2]
                .detach()
                .cpu()
                .numpy()
            )
            surf_mask = None
            try:
                wall = x_np[:, 2] if x_np.shape[1] >= 3 else None
                nxy = x_np[:, 3:5] if x_np.shape[1] >= 5 else None
                if nxy is not None:
                    surf_mask = (np.abs(nxy).sum(axis=1) > 1e-8)
                    if wall is not None:
                        surf_mask = np.logical_or(surf_mask, (wall < 1e-6))
                elif wall is not None:
                    surf_mask = (wall < 1e-6)
            except Exception:
                surf_mask = None

            poly = None
            if surf_mask is not None and np.any(surf_mask):
                pts = pos_np[surf_mask]
                if pts.shape[0] >= 3:
                    c = pts.mean(axis=0)
                    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
                    order = np.argsort(ang)
                    poly = Path(pts[order], closed=True)

            if poly is not None:
                tris = tri.triangles
                centers = np.mean(
                    np.stack([tri.x[tris], tri.y[tris]], axis=-1), axis=1
                )
                inside = poly.contains_points(centers, radius=-1e-6)
                if inside is not None and inside.any():
                    tri.set_mask(inside.astype(bool))
        except Exception:
            pass

    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    a1, a2, a3 = ax
    c1 = a1.tricontourf(tri, gt_c, levels=50, vmin=vmin, vmax=vmax, cmap=cmap)
    c2 = a2.tricontourf(tri, pr_c, levels=50, vmin=vmin, vmax=vmax, cmap=cmap)
    c3 = a3.tricontourf(tri, err, levels=50, cmap='magma')
    if show_mesh:
        for a in (a1, a2, a3):
            a.triplot(tri, color='k', lw=0.25, alpha=0.35)
    for a, t in zip((a1, a2, a3), titles):
        a.set_aspect('equal', 'box')
        a.set_title(t)
        a.set_xlabel('x')
        a.set_ylabel('y')
    m1 = plt.cm.ScalarMappable(cmap=cmap)
    m1.set_clim(vmin, vmax)
    fig.colorbar(m1, ax=[a1, a2], fraction=0.046, pad=0.04, label='Cp')
    fig.colorbar(
        plt.cm.ScalarMappable(cmap='magma'), ax=a3, fraction=0.046, pad=0.04, label='abs error'
    )

    # === Save figure with timestamp ===
    if save:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"pred_vs_gt_channel{channel}_{timestamp}.png")
        plt.savefig(filename, dpi=200)
        print(f"[Saved] {filename}")

    plt.show()


@torch.no_grad()
def plot_surface_pressure(
    d: Data,
    y_pred_denorm: torch.Tensor = None,
    y_true_denorm: torch.Tensor = None,
):
    """
    Visualize surface pressure distribution comparison.

    Plots ground truth and predicted pressure along the airfoil surface,
    ordered by x-coordinate for easy comparison of suction and pressure sides.

    Args:
        d: PyG Data object with node features and positions
        y_pred_denorm: (N, 4) tensor of denormalized predictions. If None, must call external predict function.
        y_true_denorm: (N, 4) tensor of denormalized ground truth. If None, assumes d.y is available.

    Returns:
        None (displays figure)
    """
    # --- Get Surface Data ---
    xvars = d.x.cpu()
    pos = d.pos.cpu()

    normals = xvars[:, 3:5]
    surf_mask = (normals.abs().sum(dim=1) > 0)
    surf_idx = torch.nonzero(surf_mask, as_tuple=False).squeeze(-1).long()

    if surf_idx.numel() < 5:
        print("Not enough surface nodes to plot.")
        return

    pos_surf = pos[surf_idx]

    # Order points by x-coordinate for plotting
    ord_idx = torch.argsort(pos_surf[:, 0])
    ordered_pos = pos_surf[ord_idx]

    # --- Ground Truth Pressure ---
    if y_true_denorm is None:
        y_true_denorm = d.y.cpu()
    p_true = torch.index_select(y_true_denorm, 0, surf_idx)[:, 2]
    p_true_ordered = p_true[ord_idx]

    # --- Predicted Pressure ---
    if y_pred_denorm is None:
        raise ValueError("y_pred_denorm must be provided or prediction must be computed externally")
    p_pred = torch.index_select(y_pred_denorm, 0, surf_idx)[:, 2]
    p_pred_ordered = p_pred[ord_idx]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ordered_pos[:, 0], p_true_ordered, 'o-', label='Ground Truth Pressure', markersize=4)
    ax.plot(ordered_pos[:, 0], p_pred_ordered, 'x-', label='Predicted Pressure', markersize=4)

    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Pressure / Density (p/rho)')
    ax.set_title('Surface Pressure Distribution')
    ax.legend()
    ax.grid(True)
    plt.gca().invert_yaxis()  # In aerodynamics, it's common to plot negative pressure upwards
    plt.show()
