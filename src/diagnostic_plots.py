"""Diagnostic plots for training analysis (e.g., inlet BC velocity)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.training_common import DataBundle
from src.prediction import predict_one_local

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INLET_BC_PLOT_DIR = PROJECT_ROOT / "experiments" / "loss_behavior"


@torch.no_grad()
def plot_inlet_bc_velocity(model, data_bundle: DataBundle, device, scfg):
    """Plot inlet velocity-u: ground truth vs prediction along y at x=x_min.

    Saves to experiments/loss_behavior/inlet_bc_velocity_u.png (overwritten each run).
    """
    from src.airfrans_utils import build_bc_masks_airfrans

    val_graphs = data_bundle.val_graphs
    train_graphs = data_bundle.train_graphs
    d_raw = val_graphs[0] if len(val_graphs) > 0 else (train_graphs[0] if len(train_graphs) > 0 else None)
    if d_raw is None:
        print("[inlet BC plot] No graph available, skipping.")
        return

    x_scaler = data_bundle.x_scaler
    y_scaler = data_bundle.y_scaler

    # Run prediction (returns normalized data + normalized prediction)
    dm_norm, y_pred_norm = predict_one_local(d_raw, model, x_scaler, y_scaler, device, scfg.amp)

    # Denormalize y
    y_gt_denorm = y_scaler.inverse(dm_norm.y.cpu())
    y_pred_denorm = y_scaler.inverse(y_pred_norm.cpu())

    # Get raw (unnormalized) x features for position and inlet mask
    x_raw = d_raw.x.cpu()
    pos = d_raw.pos.cpu() if d_raw.pos is not None else x_raw[:, :2]

    # Build inlet mask from raw features
    d_for_mask = d_raw.clone()
    if hasattr(d_for_mask, 'edge_index') and d_for_mask.edge_index is not None:
        if hasattr(d_raw, 'edge_attr_dxdy'):
            d_for_mask.edge_attr_dxdy = d_raw.edge_attr_dxdy
        elif hasattr(d_raw, 'edge_attr'):
            d_for_mask.edge_attr = d_raw.edge_attr
    d_for_mask = build_bc_masks_airfrans(d_for_mask)
    inlet_mask = d_for_mask.is_inlet.bool().cpu()

    if not inlet_mask.any():
        print("[inlet BC plot] No inlet nodes found, skipping.")
        return

    # Extract inlet data
    y_inlet = pos[inlet_mask, 1].numpy()
    u_gt = y_gt_denorm[inlet_mask, 0].numpy()   # velocity-u ground truth
    u_pred = y_pred_denorm[inlet_mask, 0].numpy()  # velocity-u prediction

    # Also extract inlet_u target if available
    inlet_u_target = getattr(d_for_mask, 'inlet_u', None)
    u_target = None
    if inlet_u_target is not None:
        u_target = inlet_u_target[inlet_mask, 0].cpu().numpy()

    # Sort by y for clean plot
    sort_idx = np.argsort(y_inlet)
    y_inlet = y_inlet[sort_idx]
    u_gt = u_gt[sort_idx]
    u_pred = u_pred[sort_idx]
    if u_target is not None:
        u_target = u_target[sort_idx]

    # Compute stats
    mae = float(np.mean(np.abs(u_pred - u_gt)))
    rmse = float(np.sqrt(np.mean((u_pred - u_gt) ** 2)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(y_inlet, u_gt, 'o-', label='Ground Truth', markersize=3, linewidth=1.2, alpha=0.8)
    ax.plot(y_inlet, u_pred, 'x-', label='Prediction', markersize=3, linewidth=1.2, alpha=0.8)
    if u_target is not None:
        ax.plot(y_inlet, u_target, '--', label='BC Target (inlet_u)', linewidth=1.0, alpha=0.6, color='gray')
    ax.set_xlabel('y-coordinate')
    ax.set_ylabel('Velocity u')
    ax.set_title(f'Inlet BC: Velocity-u vs y  (MAE={mae:.4e}, RMSE={rmse:.4e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    INLET_BC_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = INLET_BC_PLOT_DIR / "inlet_bc_velocity_u.png"
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[inlet BC plot] Saved -> {save_path}")
