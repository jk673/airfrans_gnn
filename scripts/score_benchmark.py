#!/usr/bin/env python3
"""
scripts/score_benchmark.py — Compute FLOW-GLIDE-style benchmark metrics and compare.

Produces the 6 standard metrics:
  1. Volume Rel. L₂  — relative L2 over all nodes, 4 channels (ux, uy, p, nu_t)
  2. Surface Rel. L₂  — relative L2 over surface nodes, pressure channel only
  3. CD relative error — mean |CD_pred − CD_gt| / |CD_gt| across test samples
  4. CL relative error — mean |CL_pred − CL_gt| / |CL_gt| across test samples
  5. ρ_D (Spearman)    — rank correlation of drag coefficient across test samples
  6. ρ_L (Spearman)    — rank correlation of lift coefficient across test samples

Outputs:
  - Console table comparing your model vs. FLOW-GLIDE baselines
  - JSON file at --output (default: benchmark_results.json)
  - Markdown table at --output-md (default: benchmark_results.md)

Usage:
    python scripts/score_benchmark.py --checkpoint checkpoints/best.pt --task full
    python scripts/score_benchmark.py --checkpoint checkpoints/best.pt --task full --model-name "Ours (v3)"
    python scripts/score_benchmark.py --checkpoint checkpoints/best.pt --task full --output results/benchmark.json

Metric definitions (matching FLOW-GLIDE, Su et al. 2025):
    Volume Rel. L₂ = sqrt(Σ_i Σ_c (pred_ic - gt_ic)²) / sqrt(Σ_i Σ_c gt_ic²)
        where i runs over ALL nodes in ALL test graphs, c over 4 output channels.
    Surface Rel. L₂ = sqrt(Σ_j (p_pred_j - p_gt_j)²) / sqrt(Σ_j p_gt_j²)
        where j runs over SURFACE nodes in ALL test graphs, pressure channel only.
    CD/CL errors and Spearman correlations are computed per-graph then aggregated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats as sp_stats
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from tqdm.auto import tqdm

from src.training_common import SmokeCfg, load_and_prepare_data, collate_pyg, NormalizedDataset
from src.global_context_processor import EnhancedCFDModelWithGlobalContext
from src.utils import _prep_graph_for_norm
from src.metrics import surface_mask_from_x_phys, compute_force_coefficients


# ============================================================================
# Metric computation
# ============================================================================

@torch.no_grad()
def score_test_set(
    eval_loader: DataLoader,
    model: torch.nn.Module,
    x_scaler,
    y_scaler,
    device: torch.device,
    *,
    eps: float = 1e-12,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute all 6 benchmark metrics on the given test loader.

    Returns dict with keys:
        volume_rel_l2, surface_rel_l2,
        cd_relative_error, cl_relative_error,
        rho_d, rho_l
    """
    model.eval()

    # Accumulators for global relative L2 (volume)
    vol_sum_err2 = 0.0  # Σ (pred - gt)²  over all volume nodes, 4 channels
    vol_sum_gt2 = 0.0   # Σ gt²

    # Accumulators for global relative L2 (surface pressure)
    surf_sum_err2 = 0.0
    surf_sum_gt2 = 0.0

    # Per-graph force coefficients
    cd_gt_list: List[float] = []
    cd_pred_list: List[float] = []
    cl_gt_list: List[float] = []
    cl_pred_list: List[float] = []

    for idx, batch in enumerate(tqdm(eval_loader, desc="Scoring", leave=True)):
        if batch is None:
            continue

        b = batch.to(device)

        # Forward pass
        pred_norm = model(b).detach().cpu()
        targ_norm = b.y.detach().cpu()

        # Denormalize to physical space
        pred_phys = y_scaler.inverse(pred_norm)
        targ_phys = y_scaler.inverse(targ_norm)
        x_phys = x_scaler.inverse(b.x.detach().cpu())

        # ------ Surface / Volume masks ------
        surf_mask = surface_mask_from_x_phys(x_phys)
        vol_mask = ~surf_mask

        # ------ Volume Relative L2 (4 channels, all nodes) ------
        # FLOW-GLIDE uses ALL nodes (not volume-only) for "Volume" metric.
        # Their definition: relative L2 over the predicted variables jointly.
        all_err = (pred_phys - targ_phys).float()
        all_gt = targ_phys.float()
        vol_sum_err2 += float((all_err ** 2).sum().item())
        vol_sum_gt2 += float((all_gt ** 2).sum().item())

        # ------ Surface Relative L2 (pressure only, surface nodes) ------
        if surf_mask.any():
            p_pred_surf = pred_phys[surf_mask, 2]  # pressure channel
            p_gt_surf = targ_phys[surf_mask, 2]
            surf_err = (p_pred_surf - p_gt_surf).float()
            surf_sum_err2 += float((surf_err ** 2).sum().item())
            surf_sum_gt2 += float((p_gt_surf ** 2).sum().item())

        # ------ Force coefficients (per-graph) ------
        # Need to handle batched graphs: split by batch index
        if hasattr(b, 'batch') and b.batch is not None:
            batch_ids = b.batch.detach().cpu()
            unique_batches = batch_ids.unique()
        else:
            unique_batches = torch.tensor([0])
            batch_ids = torch.zeros(x_phys.size(0), dtype=torch.long)

        for bid in unique_batches:
            mask_b = (batch_ids == bid)
            n_nodes = mask_b.sum().item()

            # Reconstruct single-graph Data for force coefficient computation
            d_gt = Data(
                x=x_phys[mask_b],
                y=targ_phys[mask_b],
                pos=b.pos[mask_b].detach().cpu() if b.pos is not None else None,
            )
            d_pred = Data(
                x=x_phys[mask_b],
                y=pred_phys[mask_b],
                pos=b.pos[mask_b].detach().cpu() if b.pos is not None else None,
            )

            coeffs_gt = compute_force_coefficients(d_gt, eps=1e-9)
            coeffs_pred = compute_force_coefficients(d_pred, eps=1e-9)

            if coeffs_gt is not None and coeffs_pred is not None:
                cd_gt_list.append(coeffs_gt['CD'])
                cd_pred_list.append(coeffs_pred['CD'])
                cl_gt_list.append(coeffs_gt['CL'])
                cl_pred_list.append(coeffs_pred['CL'])

                if verbose:
                    print(
                        f"  Graph {idx}/{bid}: "
                        f"CD_gt={coeffs_gt['CD']:.4f} CD_pred={coeffs_pred['CD']:.4f}  "
                        f"CL_gt={coeffs_gt['CL']:.4f} CL_pred={coeffs_pred['CL']:.4f}"
                    )

    # ------ Aggregate metrics ------
    results: Dict[str, float] = {}

    # 1. Volume Rel. L₂
    results['volume_rel_l2'] = math.sqrt(vol_sum_err2 / (vol_sum_gt2 + eps))

    # 2. Surface Rel. L₂
    results['surface_rel_l2'] = math.sqrt(surf_sum_err2 / (surf_sum_gt2 + eps))

    # 3-4. CD/CL relative error
    if len(cd_gt_list) > 0:
        cd_gt = np.array(cd_gt_list)
        cd_pred = np.array(cd_pred_list)
        cl_gt = np.array(cl_gt_list)
        cl_pred = np.array(cl_pred_list)

        # Mean relative error (%)
        cd_rel_err = np.mean(np.abs(cd_pred - cd_gt) / (np.abs(cd_gt) + 1e-12))
        cl_rel_err = np.mean(np.abs(cl_pred - cl_gt) / (np.abs(cl_gt) + 1e-12))
        results['cd_relative_error'] = float(cd_rel_err)
        results['cl_relative_error'] = float(cl_rel_err)

        # 5-6. Spearman rank correlations
        if len(cd_gt_list) >= 3:
            rho_d, _ = sp_stats.spearmanr(cd_gt, cd_pred)
            rho_l, _ = sp_stats.spearmanr(cl_gt, cl_pred)
            results['rho_d'] = float(rho_d) if not np.isnan(rho_d) else 0.0
            results['rho_l'] = float(rho_l) if not np.isnan(rho_l) else 0.0
        else:
            results['rho_d'] = float('nan')
            results['rho_l'] = float('nan')

        results['_n_graphs_with_coefficients'] = len(cd_gt_list)
    else:
        results['cd_relative_error'] = float('nan')
        results['cl_relative_error'] = float('nan')
        results['rho_d'] = float('nan')
        results['rho_l'] = float('nan')
        results['_n_graphs_with_coefficients'] = 0

    return results


# ============================================================================
# Comparison table generation
# ============================================================================

def load_benchmark_reference(path: Optional[str] = None) -> Dict:
    """Load FLOW-GLIDE reference table."""
    if path is None:
        path = str(PROJECT_ROOT / "benchmark_reference.json")
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def format_val(v: float, key: str) -> str:
    """Format a metric value for display."""
    if np.isnan(v):
        return "N/A"
    if key in ('cd_relative_error', 'cl_relative_error'):
        return f"{v:.4f}"
    elif key in ('rho_d', 'rho_l'):
        return f"{v:.4f}"
    else:
        return f"{v:.4f}"


def build_comparison_table(
    our_results: Dict[str, float],
    our_name: str,
    ref: Dict,
    top_n: int = 5,
) -> Tuple[str, str]:
    """
    Build Markdown + console comparison tables.

    Returns (markdown_str, console_str).
    """
    metric_keys = [
        'volume_rel_l2', 'surface_rel_l2',
        'cd_relative_error', 'cl_relative_error',
        'rho_d', 'rho_l',
    ]
    headers = [
        'Volume Rel.L₂ ↓', 'Surface Rel.L₂ ↓',
        'CD Rel.Err ↓', 'CL Rel.Err ↓',
        'ρ_D ↑', 'ρ_L ↑',
    ]
    directions = ref.get('_directions', {})

    # Collect all rows: baselines + ours
    baselines = ref.get('baselines', {})

    # Select which baselines to show (top N by volume_rel_l2 + always show ours)
    if baselines:
        sorted_names = sorted(
            baselines.keys(),
            key=lambda n: baselines[n].get('volume_rel_l2', 999),
        )
        show_names = sorted_names[:top_n]
    else:
        show_names = []

    rows = []
    for name in show_names:
        vals = baselines[name]
        rows.append((name, [vals.get(k, float('nan')) for k in metric_keys]))

    # Add our model (always last row, highlighted)
    rows.append((f"**{our_name}**", [our_results.get(k, float('nan')) for k in metric_keys]))

    # Find best values for highlighting
    best_vals = {}
    for i, k in enumerate(metric_keys):
        all_vals = [r[1][i] for r in rows if not np.isnan(r[1][i])]
        if not all_vals:
            best_vals[k] = None
            continue
        d = directions.get(k, 'lower_is_better')
        best_vals[k] = min(all_vals) if d == 'lower_is_better' else max(all_vals)

    # Build Markdown
    md_lines = []
    md_lines.append(f"| Model | {' | '.join(headers)} |")
    md_lines.append(f"|{'---|' * (len(headers) + 1)}")
    for name, vals in rows:
        formatted = []
        for v, k in zip(vals, metric_keys):
            s = format_val(v, k)
            if best_vals.get(k) is not None and not np.isnan(v):
                if abs(v - best_vals[k]) < 1e-8:
                    s = f"**{s}**"
            formatted.append(s)
        md_lines.append(f"| {name} | {' | '.join(formatted)} |")

    md_str = '\n'.join(md_lines)

    # Build console table
    col_w = 18
    name_w = max(20, max(len(r[0].replace('**', '')) for r in rows) + 2)

    sep = '+' + '-' * name_w + '+' + (('-' * col_w + '+') * len(headers))
    hdr_line = f"|{'Model':^{name_w}}|" + '|'.join(f'{h:^{col_w}}' for h in headers) + '|'

    console_lines = [sep, hdr_line, sep]
    for name, vals in rows:
        clean_name = name.replace('**', '')
        cells = [f'{format_val(v, k):^{col_w}}' for v, k in zip(vals, metric_keys)]
        console_lines.append(f"|{clean_name:^{name_w}}|" + '|'.join(cells) + '|')
    console_lines.append(sep)

    console_str = '\n'.join(console_lines)

    return md_str, console_str


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Score a trained model against FLOW-GLIDE benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained checkpoint (.pt)")
    p.add_argument("--task", type=str, default="full",
                   help="AirfRANS task: full, scarce, reynolds, aoa")
    p.add_argument("--model-name", type=str, default="Ours",
                   help="Display name for your model in the table")

    # Model architecture (must match checkpoint)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=14)

    # Output
    p.add_argument("--output", type=str, default="benchmark_results.json",
                   help="Path for JSON results")
    p.add_argument("--output-md", type=str, default="benchmark_results.md",
                   help="Path for Markdown comparison table")
    p.add_argument("--reference", type=str, default=None,
                   help="Path to benchmark_reference.json (auto-detected)")
    p.add_argument("--top-n", type=int, default=5,
                   help="Show top N baselines from reference table")

    # Eval settings
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for inference (1 recommended for per-graph metrics)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Config
    scfg = SmokeCfg()
    scfg.task = args.task
    scfg.hidden = args.hidden
    scfg.layers = args.layers
    scfg.batch_size = args.batch_size

    print(f"\n{'='*60}")
    print(f"BENCHMARK SCORING: {args.model_name}")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Task:       {args.task}")
    print(f"  Hidden:     {args.hidden}")
    print(f"  Layers:     {args.layers}")

    # Load data
    print(f"\nLoading data...")
    data_bundle = load_and_prepare_data(scfg)
    x_scaler = data_bundle.x_scaler
    y_scaler = data_bundle.y_scaler

    # Build test loader
    test_graphs = data_bundle.val_graphs  # 'val_graphs' holds test split for prebuilt_edges
    if not isinstance(test_graphs, list) or len(test_graphs) == 0:
        raise ValueError("No test graphs found. Check prebuilt_edges/<task>/test/ exists.")

    test_prepped = [_prep_graph_for_norm(g) for g in test_graphs]
    test_ds = NormalizedDataset(test_prepped, x_scaler, y_scaler)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_pyg,
    )
    print(f"  Test graphs: {len(test_ds)}")

    # Build model
    print(f"\nBuilding model...")
    model = EnhancedCFDModelWithGlobalContext(
        node_feat_dim=7, edge_feat_dim=5,
        hidden_dim=scfg.hidden, output_dim=4,
        num_mp_layers=scfg.layers,
        dropout_p=0.1, config=scfg,
    )

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "?")
        print(f"  Loaded checkpoint (epoch {epoch})")
    else:
        model.load_state_dict(ckpt)
        print(f"  Loaded raw state dict")

    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Score
    print(f"\nScoring test set...")
    results = score_test_set(
        test_loader, model, x_scaler, y_scaler, device,
        verbose=args.verbose,
    )

    # Add metadata
    results['_model_name'] = args.model_name
    results['_checkpoint'] = args.checkpoint
    results['_task'] = args.task
    results['_n_parameters'] = n_params
    results['_n_test_graphs'] = len(test_ds)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Volume Rel. L₂:    {results['volume_rel_l2']:.6f}")
    print(f"  Surface Rel. L₂:   {results['surface_rel_l2']:.6f}")
    print(f"  CD Relative Error:  {results['cd_relative_error']:.6f}")
    print(f"  CL Relative Error:  {results['cl_relative_error']:.6f}")
    print(f"  ρ_D (Spearman):     {results['rho_d']:.4f}")
    print(f"  ρ_L (Spearman):     {results['rho_l']:.4f}")
    print(f"  Graphs with coeffs: {results['_n_graphs_with_coefficients']}")

    # Save JSON
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved JSON → {args.output}")

    # Load reference and build comparison
    ref = load_benchmark_reference(args.reference)
    if ref:
        md_table, console_table = build_comparison_table(
            results, args.model_name, ref, top_n=args.top_n,
        )

        print(f"\n{'='*60}")
        print(f"COMPARISON (top {args.top_n} baselines + Ours)")
        print(f"{'='*60}")
        print(console_table)

        # Save Markdown
        md_content = f"# Benchmark Results: {args.model_name}\n\n"
        md_content += f"- **Task:** {args.task}\n"
        md_content += f"- **Checkpoint:** `{args.checkpoint}`\n"
        md_content += f"- **Parameters:** {n_params:,}\n"
        md_content += f"- **Test graphs:** {len(test_ds)}\n\n"
        md_content += md_table + "\n"

        with open(args.output_md, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"\n  Saved Markdown → {args.output_md}")
    else:
        print("\n  [No benchmark_reference.json found — skipping comparison table]")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
