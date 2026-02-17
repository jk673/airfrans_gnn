"""Diagnostic plots and boundary-region statistics for AirfRANS GNN evaluation.

Consolidated from diagnostic_plots.py and diagnostic_stats.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt

from src.data import DataBundle
from src.prediction import predict_one_local
from src.preprocessing import build_bc_masks_airfrans

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INLET_BC_PLOT_DIR = PROJECT_ROOT / "experiments" / "loss_behavior"

CHANNEL_NAMES = ("u", "v", "p/rho", "nu_t")
REGION_NAMES = ("wall", "inlet", "outlet", "farfield", "interior", "ALL")


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_inlet_bc_velocity(model, data_bundle: DataBundle, device, scfg):
    """Plot inlet velocity-u: ground truth vs prediction along y at x=x_min.

    Saves to experiments/loss_behavior/inlet_bc_velocity_u.png (overwritten each run).
    """
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


# ---------------------------------------------------------------------------
# Diagnostic statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def print_diagnostic_stats(
    test_edges: list[Data],
    model: torch.nn.Module,
    x_scaler,
    y_scaler,
    device: torch.device,
    *,
    space: str = "physical",
    max_graphs: int | None = None,
    show_worst: int = 3,
) -> dict:
    """Compute and print per-boundary-region error statistics.

    Args:
        test_edges: List of raw (unnormalized) PyG Data graphs with prebuilt edges.
        model: Trained GNN model (will be set to eval mode).
        x_scaler: Fitted StandardScaler for node features.
        y_scaler: Fitted StandardScaler for targets.
        device: Computation device.
        space: 'physical' to denormalize before computing errors, 'normalized' otherwise.
        max_graphs: Limit number of graphs for quick checks.
        show_worst: Number of worst graphs to show per region.

    Returns:
        dict with keys per region containing MAE, rel_L2, percentiles, node counts, etc.
    """
    model.eval()

    graphs = test_edges
    if max_graphs is not None:
        graphs = graphs[:max_graphs]
    n_graphs = len(graphs)
    n_ch = len(CHANNEL_NAMES)

    # Per-region accumulators
    accum = {}
    for region in REGION_NAMES:
        accum[region] = {
            "sum_abs_err": torch.zeros(n_ch, dtype=torch.float64),
            "sum_sq_err": torch.zeros(n_ch, dtype=torch.float64),
            "sum_sq_gt": torch.zeros(n_ch, dtype=torch.float64),
            "sum_nodes": 0,
            "node_counts": [],          # per-graph node count
            "max_err": torch.zeros(n_ch, dtype=torch.float64),
            "all_abs_err": [],          # list of (n_nodes, n_ch) tensors for percentiles
            "per_graph_mae": [],        # (graph_idx, mean_mae_across_channels)
        }

    # CFD-specific
    wall_vel_mag_sum = 0.0
    wall_vel_mag_count = 0
    wall_nut_abs_sum = 0.0
    wall_nut_count = 0

    for gi, d_raw in enumerate(graphs):
        # 1. Predict
        dm_norm, y_pred_norm = predict_one_local(d_raw, model, x_scaler, y_scaler, device)

        # 2. Get targets and predictions in the requested space
        if space == "physical":
            y_pred = y_scaler.inverse(y_pred_norm)
            y_true = y_scaler.inverse(dm_norm.y)
        else:
            y_pred = y_pred_norm
            y_true = dm_norm.y

        abs_err = (y_pred - y_true).abs().double()  # (N, 4)
        sq_err = abs_err.pow(2)
        sq_gt = y_true.double().pow(2)

        # 3. Build BC masks from raw features
        d_mask = Data(**{k: v for k, v in d_raw})
        d_mask = build_bc_masks_airfrans(d_mask)

        is_wall = d_mask.is_wall.bool()
        is_inlet = d_mask.is_inlet.bool()
        is_outlet = d_mask.is_outlet.bool()
        is_farfield = d_mask.is_farfield.bool()
        is_interior = ~(is_wall | is_inlet | is_outlet | is_farfield)
        all_mask = torch.ones(y_pred.size(0), dtype=torch.bool)

        masks = {
            "wall": is_wall,
            "inlet": is_inlet,
            "outlet": is_outlet,
            "farfield": is_farfield,
            "interior": is_interior,
            "ALL": all_mask,
        }

        for region, mask in masks.items():
            n_nodes = int(mask.sum())
            a = accum[region]
            a["node_counts"].append(n_nodes)
            a["sum_nodes"] += n_nodes

            if n_nodes == 0:
                a["per_graph_mae"].append((gi, 0.0))
                continue

            ae = abs_err[mask]   # (n, 4)
            se = sq_err[mask]
            sg = sq_gt[mask]

            a["sum_abs_err"] += ae.sum(dim=0)
            a["sum_sq_err"] += se.sum(dim=0)
            a["sum_sq_gt"] += sg.sum(dim=0)
            a["max_err"] = torch.max(a["max_err"], ae.max(dim=0).values)
            a["all_abs_err"].append(ae.float())
            a["per_graph_mae"].append((gi, float(ae.mean())))

        # CFD diagnostics on wall nodes
        n_wall = int(is_wall.sum())
        if n_wall > 0:
            if space == "physical":
                u_wall = y_pred[is_wall, 0]
                v_wall = y_pred[is_wall, 1]
            else:
                u_wall = y_pred_norm[is_wall, 0]
                v_wall = y_pred_norm[is_wall, 1]
            vel_mag = (u_wall.pow(2) + v_wall.pow(2)).sqrt()
            wall_vel_mag_sum += float(vel_mag.sum())
            wall_vel_mag_count += n_wall

            nut_wall_err = abs_err[is_wall, 3]
            wall_nut_abs_sum += float(nut_wall_err.sum())
            wall_nut_count += n_wall

    # ---- Aggregate statistics ----
    stats = {}
    for region in REGION_NAMES:
        a = accum[region]
        total = a["sum_nodes"]
        nc = torch.tensor(a["node_counts"], dtype=torch.float64)
        rs = {
            "node_count": {
                "mean": float(nc.mean()) if len(nc) > 0 else 0.0,
                "std": float(nc.std()) if len(nc) > 1 else 0.0,
                "min": int(nc.min()) if len(nc) > 0 else 0,
                "max": int(nc.max()) if len(nc) > 0 else 0,
                "total": total,
            },
        }

        if total > 0:
            mae = a["sum_abs_err"] / total
            rel_l2 = (a["sum_sq_err"] / a["sum_sq_gt"].clamp(min=1e-30)).sqrt()
            rs["mae"] = [float(v) for v in mae]
            rs["rel_l2"] = [float(v) for v in rel_l2]
            rs["max_abs_err"] = [float(v) for v in a["max_err"]]

            # Percentiles
            if a["all_abs_err"]:
                cat = torch.cat(a["all_abs_err"], dim=0)  # (total, 4)
                pcts = {}
                for q, label in [(0.50, "p50"), (0.90, "p90"), (0.99, "p99")]:
                    vals = torch.quantile(cat, q, dim=0)
                    pcts[label] = [float(v) for v in vals]
                rs["percentiles"] = pcts
        else:
            rs["mae"] = [float("nan")] * n_ch
            rs["rel_l2"] = [float("nan")] * n_ch
            rs["max_abs_err"] = [float("nan")] * n_ch
            rs["percentiles"] = {
                "p50": [float("nan")] * n_ch,
                "p90": [float("nan")] * n_ch,
                "p99": [float("nan")] * n_ch,
            }

        # Worst graphs
        sorted_graphs = sorted(a["per_graph_mae"], key=lambda x: -x[1])
        rs["worst_graphs"] = sorted_graphs[:show_worst]

        stats[region] = rs

    # CFD diagnostics
    stats["cfd"] = {
        "wall_vel_mag_mae": wall_vel_mag_sum / max(wall_vel_mag_count, 1),
        "wall_nut_mae": wall_nut_abs_sum / max(wall_nut_count, 1),
    }

    # ---- Print report ----
    _print_report(stats, space, n_graphs, show_worst)

    return stats


def _print_report(stats: dict, space: str, n_graphs: int, show_worst: int) -> None:
    """Format and print the diagnostic tables."""
    sep = "=" * 60
    print(sep)
    print(f"DIAGNOSTIC STATISTICS ({space} space, {n_graphs} graphs)")
    print(sep)

    # Node count distribution
    print("\n--- Node Count Distribution ---")
    print(f"{'Region':<11}| {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Total':>9}")
    print("-" * 11 + "+" + ("-" * 10 + "+") * 4 + "-" * 10)
    for region in REGION_NAMES:
        nc = stats[region]["node_count"]
        print(
            f"{region:<11}| {nc['mean']:>8.0f} | {nc['std']:>8.0f} | "
            f"{nc['min']:>8d} | {nc['max']:>8d} | {nc['total']:>9d}"
        )

    # Per-region MAE
    print("\n--- Per-Region MAE ---")
    _print_metric_table(stats, "mae")

    # Per-region relative L2
    print("\n--- Per-Region Relative L2 ---")
    _print_metric_table(stats, "rel_l2")

    # Max absolute error
    print("\n--- Max Absolute Error ---")
    _print_metric_table(stats, "max_abs_err")

    # Percentiles
    print("\n--- Error Percentiles (p50 / p90 / p99) ---")
    for region in REGION_NAMES:
        pcts = stats[region]["percentiles"]
        print(f"  {region}:")
        for ci, ch_name in enumerate(CHANNEL_NAMES):
            p50 = pcts["p50"][ci]
            p90 = pcts["p90"][ci]
            p99 = pcts["p99"][ci]
            print(f"    {ch_name:<6}: {p50:.3e} / {p90:.3e} / {p99:.3e}")

    # CFD diagnostics
    cfd = stats["cfd"]
    print("\n--- CFD Diagnostics ---")
    print(f"  Wall velocity magnitude MAE: {cfd['wall_vel_mag_mae']:.4e}")
    print(f"  Wall nu_t MAE:               {cfd['wall_nut_mae']:.4e}")

    # Worst graphs
    print(f"\n--- Worst {show_worst} Graphs by Region ---")
    for region in REGION_NAMES:
        worst = stats[region]["worst_graphs"]
        parts = [f"Graph {idx} (MAE={mae:.4e})" for idx, mae in worst]
        print(f"  {region:<11}: {', '.join(parts)}")

    print(sep)


def _print_metric_table(stats: dict, key: str) -> None:
    """Print a region x channel table for a given metric key."""
    header = f"{'Region':<11}|"
    for ch in CHANNEL_NAMES:
        header += f" {ch:>10} |"
    print(header)
    print("-" * 11 + ("+" + "-" * 12) * len(CHANNEL_NAMES))
    for region in REGION_NAMES:
        vals = stats[region][key]
        row = f"{region:<11}|"
        for v in vals:
            row += f" {v:>10.3e} |"
        print(row)
