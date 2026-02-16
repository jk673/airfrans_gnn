#!/usr/bin/env python3
"""
scripts/eval_cp.py - Standalone Cp (pressure coefficient) evaluation script.

Loads a trained checkpoint and computes Cp relative L2 error on validation or test split.
Uses the same Cp definition as training: Cp = (p/ρ) / q, where q = 0.5 * U_inf^2.

Usage:
    python scripts/eval_cp.py --checkpoint checkpoints/best.pt
    python scripts/eval_cp.py --checkpoint checkpoints/best.pt --split test --surface-only
    python scripts/eval_cp.py --checkpoint checkpoints/best.pt --batch-size 1 --task scarce

Examples:
    # Evaluate on validation set (surface nodes only)
    python scripts/eval_cp.py --checkpoint checkpoints/best.pt --split val --surface-only

    # Evaluate on test set (all nodes)
    python scripts/eval_cp.py --checkpoint checkpoints/best.pt --split test --no-surface-only

    # Custom configuration
    python scripts/eval_cp.py --checkpoint checkpoints/best.pt --hidden 256 --layers 16
"""

from __future__ import annotations

import argparse
import os
import math
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training_common import SmokeCfg, load_and_prepare_data, collate_pyg, NormalizedDataset
from src.global_context_processor import EnhancedCFDModelWithGlobalContext
from src.utils import _prep_graph_for_norm
from src.metrics import (
    cp_from_p_over_rho,
    relative_l2,
    surface_mask_from_x_phys,
    aggregate_cp_relative_l2_stats,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Cp relative L2 error from a trained checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., checkpoints/best.pt)",
    )

    # Evaluation settings
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Split to evaluate: 'val' (training validation) or 'test' (prebuilt_edges/<task>/test)",
    )
    parser.add_argument(
        "--surface-only",
        action="store_true",
        default=False,
        help="Evaluate Cp error on surface nodes only",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (typically 1 for per-graph metrics)",
    )

    # Model configuration (must match training)
    parser.add_argument(
        "--task",
        type=str,
        default="scarce",
        help="Task name (must match checkpoint training)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Hidden dimension (must match checkpoint)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=14,
        help="Number of message-passing layers (must match checkpoint)",
    )

    # Technical settings
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cuda', 'cpu'",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small epsilon to prevent division by zero",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-graph results",
    )

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, model, device):
    """Load model weights from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        print(f"  Format: dict with 'model' key")
        if "epoch" in ckpt:
            print(f"  Epoch: {ckpt['epoch']}")
        if "val_loss" in ckpt:
            print(f"  Val loss: {ckpt['val_loss']:.6g}")
    else:
        state_dict = ckpt
        print(f"  Format: raw state dict")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def evaluate_cp_relative_l2(
    eval_loader: DataLoader,
    model,
    x_scaler,
    y_scaler,
    device,
    *,
    surface_only: bool = True,
    eps: float = 1e-12,
    verbose: bool = False,
) -> dict:
    """
    Evaluate Cp relative L2 error on the provided data loader.

    Args:
        eval_loader: DataLoader with normalized graphs
        model: Trained model in eval mode
        x_scaler: StandardScaler for input features
        y_scaler: StandardScaler for target values
        device: Torch device
        surface_only: If True, evaluate only on surface nodes
        eps: Small value to prevent division by zero
        verbose: If True, print per-graph results

    Returns:
        Dictionary with per-graph and global statistics
    """
    per_graph_errors: List[float] = []
    sum_err2 = 0.0
    sum_gt2 = 0.0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating", leave=True)):
            if batch is None:
                continue

            b = batch.to(device)

            # Model prediction
            pred_norm = model(b).detach().cpu()
            targ_norm = b.y.detach().cpu()

            # Denormalize to physical space
            pred_phys = y_scaler.inverse(pred_norm)
            targ_phys = y_scaler.inverse(targ_norm)
            x_phys = x_scaler.inverse(b.x.detach().cpu())

            # Compute Cp from p/ρ (column 2)
            cp_pred = cp_from_p_over_rho(pred_phys, x_phys, p_channel=2, eps=eps)
            cp_true = cp_from_p_over_rho(targ_phys, x_phys, p_channel=2, eps=eps)

            # Apply surface mask if requested
            if surface_only:
                mask = surface_mask_from_x_phys(x_phys)
                cp_pred = cp_pred[mask]
                cp_true = cp_true[mask]

            # Compute relative L2 for this graph
            rel_l2 = relative_l2(cp_pred, cp_true, eps=eps)
            per_graph_errors.append(rel_l2)

            # Accumulate for global metric
            err = cp_pred - cp_true
            sum_err2 += float((err * err).sum().item())
            sum_gt2 += float((cp_true * cp_true).sum().item())

            if verbose:
                n_nodes = cp_pred.numel()
                print(f"  Graph {idx:3d}: rel_l2={rel_l2:.6g}  nodes={n_nodes}")

    # Aggregate statistics
    stats = aggregate_cp_relative_l2_stats(
        rel_l2_list=per_graph_errors,
        sum_err2=sum_err2,
        sum_gt2=sum_gt2,
    )

    return stats


def main():
    """Main evaluation routine."""
    args = parse_args()

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Build configuration (must match training)
    scfg = SmokeCfg()
    scfg.task = args.task
    scfg.hidden = args.hidden
    scfg.layers = args.layers
    scfg.batch_size = args.batch_size

    print(f"\nConfiguration:")
    print(f"  Task: {scfg.task}")
    print(f"  Hidden dim: {scfg.hidden}")
    print(f"  MP layers: {scfg.layers}")
    print(f"  Split: {args.split}")
    print(f"  Surface only: {args.surface_only}")

    # Load data
    prebuilt_dir = os.path.join("prebuilt_edges", scfg.task)
    if not os.path.isdir(prebuilt_dir):
        raise FileNotFoundError(
            f"Missing prebuilt edges at {prebuilt_dir}. "
            "Run preprocessing first (see CLAUDE.md)."
        )

    print(f"\nLoading data from {prebuilt_dir}...")
    data_bundle = load_and_prepare_data(scfg)
    x_scaler = data_bundle.x_scaler
    y_scaler = data_bundle.y_scaler

    # Select split
    if args.split == "val":
        if not isinstance(data_bundle.val_norm, NormalizedDataset):
            raise ValueError("No validation dataset available")
        eval_ds = data_bundle.val_norm
        print(f"  Validation graphs: {len(eval_ds)}")
    elif args.split == "test":
        test_graphs = data_bundle.val_graphs
        if not isinstance(test_graphs, list) or len(test_graphs) == 0:
            raise ValueError("No test graphs available")
        test_prepped = [_prep_graph_for_norm(g) for g in test_graphs]
        eval_ds = NormalizedDataset(test_prepped, x_scaler, y_scaler)
        print(f"  Test graphs: {len(eval_ds)}")
    else:
        raise ValueError(f"Invalid split: {args.split}")

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pyg,
    )

    # Build model
    print(f"\nBuilding model...")
    node_dim = 7
    edge_dim = 5
    model = EnhancedCFDModelWithGlobalContext(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=scfg.hidden,
        output_dim=4,
        num_mp_layers=scfg.layers,
        dropout_p=0.1,
        config=scfg,
    )

    # Load checkpoint
    model = load_checkpoint(args.checkpoint, model, device)

    # Evaluate
    print(f"\nEvaluating Cp relative L2 error...")
    torch.set_grad_enabled(False)

    stats = evaluate_cp_relative_l2(
        eval_loader=eval_loader,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        surface_only=args.surface_only,
        eps=args.eps,
        verbose=args.verbose,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: Cp Relative L2 Error")
    print("=" * 70)
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Split:         {args.split}")
    print(f"Surface only:  {args.surface_only}")
    print(f"Graphs:        {stats['finite_graphs']} / {stats['total_graphs']}")
    print("-" * 70)
    print(f"Per-graph statistics:")
    print(f"  Mean:        {stats['per_graph_mean']:.6g}")
    print(f"  Median:      {stats['per_graph_median']:.6g}")
    print(f"  Std:         {stats['per_graph_std']:.6g}")
    print(f"  Min:         {stats['per_graph_min']:.6g}")
    print(f"  Max:         {stats['per_graph_max']:.6g}")
    print("-" * 70)
    if "global_relative_l2" in stats:
        print(f"Global relative L2 (aggregated over all nodes):")
        print(f"  {stats['global_relative_l2']:.6g}")
    print("=" * 70)


if __name__ == "__main__":
    main()
