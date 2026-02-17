#!/usr/bin/env python3
"""
scripts/train_multiscale.py — Multi-scale training pipeline for AirfRANS GNN.
Converted from 02_trainer_multi_scale.ipynb with CLI argument support.

This script uses the UltraEnhancedCFDModel which combines:
- Base EnhancedCFDModelWithGlobalContext (global attention tokens)
- Multi-scale graph convolutions (MultiScaleGraphConv)
- Spatial pyramid pooling
- Enhanced turbulence modeling physics loss

Usage:
    python scripts/train_multiscale.py [OPTIONS]
    python scripts/train_multiscale.py --help

Examples:
    # Basic multi-scale training with default settings
    python scripts/train_multiscale.py

    # Train with custom hyperparameters
    python scripts/train_multiscale.py --batch-size 4 --epochs 200 --lr 3e-4 --hidden 256

    # Train with different number of scales
    python scripts/train_multiscale.py --num-scales 4 --num-multiscale-layers 4

    # Disable spatial pyramid pooling
    python scripts/train_multiscale.py --no-spp
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import wandb

from src.config import SmokeCfg, apply_config_dict, DEFAULT_CONFIG_PATH, load_config_file
from src.data import load_and_prepare_data
from src.training import set_seed, get_lr, run_epoch, train_epoch, create_lr_scheduler, init_wandb
from src.turbulent_physics_loss import EnhancedPhysicsLoss
from src.global_context_processor import UltraEnhancedCFDModel
from src.training import train_with_scheduler
from src.prediction import evaluate_model, predict_one_for_viz
from src.visualization import plot_pred_vs_gt


# ---------------------------------------------------------------------------
# Multi-scale specific configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiScaleCfg(SmokeCfg):
    """Extended configuration for multi-scale model variant"""
    # Multi-scale architecture parameters
    num_scales: int = 3                      # Number of scales in MultiScaleGraphConv
    num_multiscale_layers: int = 3           # Number of multi-scale conv layers
    use_spatial_pyramid_pooling: bool = False  # Enable/disable SPP (usually False for node-level)
    spp_pool_sizes: tuple[int, ...] = (1, 2, 4)  # SPP pool sizes
    residual_weight_init: float = 0.1        # Initial residual connection weight

    # Turbulence modeling (EnhancedPhysicsLoss specific)
    turbulence_loss_weight: float = 0.05
    rans_loss_weight: float = 0.05
    smoothness_weight: float = 0.01
    wall_function_weight: float = 0.02
    use_adaptive_weights: bool = False

    # Override default layers for multi-scale (usually fewer base layers)
    layers: int = 7  # Base model layers (multi-scale adds 3 more)


def create_multiscale_config_from_args(args):
    """Create MultiScaleCfg from command line arguments"""
    parser_defaults = vars(parse_args([]))
    cfg = MultiScaleCfg()
    for key, value in parser_defaults.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    apply_config_dict(cfg, load_config_file(getattr(args, 'config', None)))

    arg_values = vars(args)
    for key, value in arg_values.items():
        if key == 'config' or not hasattr(cfg, key):
            continue
        if key in {'limit_train', 'limit_val'} and value is None:
            continue
        if value != parser_defaults.get(key):
            if key == 'lr_scheduler':
                value = None if value in ('', 'none', 'None') else value
            setattr(cfg, key, value)

    return cfg


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train AirfRANS Multi-Scale GNN model with enhanced physics loss',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data configuration
    parser.add_argument('--task', type=str, default='scarce',
                        choices=['full', 'scarce', 'medium'],
                        help='AirfRANS task variant')
    parser.add_argument('--root', type=str, default='Dataset',
                        help='Path to AirfRANS dataset root')
    parser.add_argument('--limit-train', type=int, default=None,
                        help='Limit number of training graphs (None = use all)')
    parser.add_argument('--limit-val', type=int, default=None,
                        help='Limit number of validation graphs (None = use all)')

    # Model architecture
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--layers', type=int, default=7,
                        help='Number of base message passing layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')

    # Multi-scale specific
    parser.add_argument('--num-scales', type=int, default=3,
                        help='Number of scales in multi-scale convolutions')
    parser.add_argument('--num-multiscale-layers', type=int, default=3,
                        help='Number of multi-scale conv layers')
    parser.add_argument('--use-spp', action='store_true', default=False,
                        help='Enable spatial pyramid pooling (usually disabled for node-level)')
    parser.add_argument('--no-spp', dest='use_spp', action='store_false',
                        help='Disable spatial pyramid pooling')

    # Global context & attention
    parser.add_argument('--use-global-tokens', action='store_true', default=True,
                        help='Enable global context tokens')
    parser.add_argument('--no-global-tokens', dest='use_global_tokens', action='store_false',
                        help='Disable global context tokens')
    parser.add_argument('--num-global-tokens', type=int, default=2,
                        help='Number of global tokens')
    parser.add_argument('--attention-heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--attention-layers', type=int, default=2,
                        help='Number of attention layers')
    parser.add_argument('--attention-dropout', type=float, default=0.0,
                        help='Attention dropout rate')
    parser.add_argument('--use-cross-attention', action='store_true', default=True,
                        help='Use cross-attention between nodes and global tokens')
    parser.add_argument('--global-pooling-type', type=str, default='attention',
                        choices=['attention', 'mean', 'max', 'set2set'],
                        help='Global pooling mechanism')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                        help='AdamW weight decay')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use automatic mixed precision (AMP)')

    # Learning rate scheduler
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=['cosine', 'cosine_warm_restarts', 'reduce_on_plateau', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--cosine-T-max', type=int, default=80,
                        help='T_max for cosine annealing')
    parser.add_argument('--cosine-eta-min', type=float, default=1e-6,
                        help='Minimum LR for cosine annealing')

    # Physics loss configuration
    parser.add_argument('--data-loss-weight', type=float, default=1.0,
                        help='Weight for data MSE loss')
    parser.add_argument('--continuity-loss-weight', type=float, default=0.05,
                        help='Initial continuity loss weight')
    parser.add_argument('--continuity-target-weight', type=float, default=0.20,
                        help='Target continuity loss weight after ramp')
    parser.add_argument('--momentum-loss-weight', type=float, default=0.05,
                        help='Initial momentum loss weight')
    parser.add_argument('--momentum-target-weight', type=float, default=0.20,
                        help='Target momentum loss weight after ramp')
    parser.add_argument('--bc-loss-weight', type=float, default=0.1,
                        help='Boundary condition loss weight')
    parser.add_argument('--ramp-start-epoch', type=int, default=40,
                        help='Epoch to start physics loss curriculum ramp')
    parser.add_argument('--ramp-epochs', type=int, default=60,
                        help='Number of epochs for curriculum ramp')
    parser.add_argument('--ramp-mode', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Curriculum ramp schedule mode')

    # Turbulence modeling (multi-scale specific)
    parser.add_argument('--turbulence-loss-weight', type=float, default=0.05,
                        help='Weight for turbulence modeling loss')
    parser.add_argument('--rans-loss-weight', type=float, default=0.05,
                        help='Weight for RANS-specific loss terms')
    parser.add_argument('--smoothness-weight', type=float, default=0.01,
                        help='Weight for smoothness loss')
    parser.add_argument('--wall-function-weight', type=float, default=0.02,
                        help='Weight for wall function loss')
    parser.add_argument('--use-adaptive-weights', action='store_true', default=False,
                        help='Use adaptive weighting for physics loss terms')

    # Physics parameters
    parser.add_argument('--chord-length', type=float, default=1.0,
                        help='Reference chord length')
    parser.add_argument('--nu-molecular', type=float, default=1.5e-5,
                        help='Molecular viscosity')
    parser.add_argument('--dynamic-uref', action='store_true', default=True,
                        help='Dynamically compute U_ref from data')
    parser.add_argument('--dynamic-re', action='store_true', default=True,
                        help='Dynamically compute Reynolds number from data')
    parser.add_argument('--use-huber-physics', action='store_true', default=True,
                        help='Use Huber loss for physics terms')
    parser.add_argument('--huber-delta', type=float, default=0.05,
                        help='Huber loss delta parameter')

    # Checkpointing
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints_multiscale',
                        help='Directory to save checkpoints')
    parser.add_argument('--ckpt-interval', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Weights & Biases
    parser.add_argument('--wandb-project', type=str, default='airfrans-gnn-multiscale',
                        help='W&B project name')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--wandb-mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='W&B logging mode')
    parser.add_argument('--enable-wandb', dest='enable_wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--disable-wandb', dest='enable_wandb', action='store_false',
                        help='Disable W&B logging')
    parser.set_defaults(enable_wandb=True)
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=['multiscale'],
                        help='W&B tags for this run')
    parser.add_argument('--use-wandb-artifacts', action='store_true', default=False,
                        help='Upload model checkpoints as W&B artifacts')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto-detect if not specified)')
    parser.add_argument('--log-every-n-steps', type=int, default=-1,
                        help='Log to W&B every N steps (-1 for epoch-only)')
    parser.add_argument('--no-viz', action='store_true', default=False,
                        help='Skip visualization at the end of training')
    parser.add_argument(
        '--config',
        type=str,
        default=os.environ.get("AIRFRANS_TRAIN_CONFIG", str(DEFAULT_CONFIG_PATH)),
        help=f'Path to JSON config file (defaults to {DEFAULT_CONFIG_PATH})'
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Parse command line arguments
    args = parse_args()
    scfg = create_multiscale_config_from_args(args)

    set_seed(scfg.seed)

    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 80)
    print('AirfRANS Multi-Scale GNN Training')
    print('=' * 80)
    print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {device}')
    print(f'Task: {scfg.task}')
    print(f'Batch size: {scfg.batch_size} | Epochs: {scfg.epochs} | LR: {scfg.lr}')
    print(f'Hidden: {scfg.hidden} | Base layers: {scfg.layers}')
    print(f'Multi-scale: {scfg.num_scales} scales x {scfg.num_multiscale_layers} layers')
    print(f'Spatial pyramid pooling: {scfg.use_spatial_pyramid_pooling}')
    print(f'Global tokens: {scfg.use_global_tokens} (num={scfg.num_global_tokens})')
    print(f'LR scheduler: {scfg.lr_scheduler}')
    print('=' * 80)

    # --- Data ---
    data_bundle = load_and_prepare_data(scfg)

    # --- Model ---
    node_dim = 7
    edge_dim = 5

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = UltraEnhancedCFDModel(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=scfg.hidden,
        output_dim=4,
        num_mp_layers=scfg.layers,
        num_scales=scfg.num_scales,
        dropout_p=scfg.dropout,
        config=scfg,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scfg.lr,
        weight_decay=scfg.weight_decay,
        betas=scfg.betas,
        eps=scfg.eps,
    )
    print(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # --- Enhanced Physics loss ---
    steps_per_epoch = len(data_bundle.train_loader)
    loss_fn = EnhancedPhysicsLoss(
        # Basic physics loss weights with curriculum learning
        data_loss_weight=scfg.data_loss_weight,
        continuity_loss_weight=scfg.continuity_loss_weight,
        continuity_target_weight=scfg.continuity_target_weight,
        momentum_loss_weight=scfg.momentum_loss_weight,
        momentum_target_weight=scfg.momentum_target_weight,
        curriculum_ramp_steps=scfg.ramp_epochs * steps_per_epoch,
        ramp_start_step=scfg.ramp_start_epoch * steps_per_epoch,
        ramp_mode=scfg.ramp_mode,
        bc_loss_weight=scfg.bc_loss_weight,

        # Turbulence modeling extensions
        turbulence_loss_weight=scfg.turbulence_loss_weight,
        rans_loss_weight=scfg.rans_loss_weight,
        smoothness_weight=scfg.smoothness_weight,
        wall_function_weight=scfg.wall_function_weight,

        # Physics parameters
        chord_length=scfg.chord_length,
        dynamic_uref_from_data=scfg.dynamic_uref_from_data,
        dynamic_re_from_data=scfg.dynamic_re_from_data,
        nu_molecular=scfg.nu_molecular,
        uinf_from=scfg.uinf_from,
        use_huber_for_physics=scfg.use_huber_for_physics,
        huber_delta=scfg.huber_delta,
        use_perimeter_norm_for_div=scfg.use_perimeter_norm_for_div,
        div_area_floor_factor=scfg.div_area_floor_factor,
        div_min_degree=scfg.div_min_degree,
        use_adaptive_weights=scfg.use_adaptive_weights,
        debug=scfg.physics_debug,
    )
    print(f"Enhanced physics loss initialized:")
    print(f"  Continuity: {scfg.continuity_loss_weight:.3f} -> {scfg.continuity_target_weight:.3f}")
    print(f"  Momentum: {scfg.momentum_loss_weight:.3f} -> {scfg.momentum_target_weight:.3f}")
    print(f"  Turbulence: {scfg.turbulence_loss_weight:.3f}")
    print(f"  RANS terms: {scfg.rans_loss_weight:.3f}")
    print(f"  Smoothness: {scfg.smoothness_weight:.3f}")
    print(f"  Wall function: {scfg.wall_function_weight:.3f}")

    # --- LR Scheduler ---
    lr_scheduler = create_lr_scheduler(optimizer, scfg)

    # --- W&B ---
    init_wandb(scfg, loss_fn)

    # --- Train ---
    train_with_scheduler(
        model, optimizer, lr_scheduler,
        data_bundle.train_loader, data_bundle.val_loader,
        scfg, device, loss_fn,
    )

    # --- Evaluate ---
    evaluate_model(model, device, data_bundle, scfg)

    # --- Visualize ---
    if not args.no_viz:
        val_edges = data_bundle.val_graphs
        train_edges = data_bundle.train_graphs
        d_vis = val_edges[0] if len(val_edges) > 0 else (train_edges[0] if len(train_edges) > 0 else None)
        if d_vis is not None:
            dm_vis_n, y_pred_vis_n = predict_one_for_viz(
                d_vis, model, device, data_bundle.x_scaler, data_bundle.y_scaler)
            plot_pred_vs_gt(
                dm_vis_n, y_pred_vis_n,
                data_bundle.y_scaler, data_bundle.x_scaler,
                channel=2, show_mesh=False, denormalize=True,
            )

    print('\n' + '=' * 80)
    print('Multi-scale training complete!')
    print('=' * 80)
    print(f'Best checkpoint saved to: {os.path.join(scfg.ckpt_dir, "best.pt")}')


if __name__ == '__main__':
    main()
