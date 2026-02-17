"""Configuration dataclass, config file helpers, and CLI argument parsing for AirfRANS GNN training."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "train_default.json"


def _resolve_config_path(config_path: Optional[str]) -> Optional[Path]:
    if config_path:
        candidate = Path(config_path)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[1] / candidate
        if candidate.exists():
            return candidate
        print(f"[config] Skipping missing config file: {candidate}")
        return None

    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return None


def load_config_file(config_path: Optional[str] = None) -> dict[str, Any]:
    path = _resolve_config_path(config_path)
    if path is None:
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"[config] Config file must be JSON object: {path}")
            return {}
        return data
    except Exception as exc:
        print(f"[config] Failed to load config: {path} ({exc})")
        return {}


def apply_config_dict(cfg_obj: Any, config_updates: dict[str, Any]) -> None:
    for key, value in config_updates.items():
        if not hasattr(cfg_obj, key):
            continue
        current = getattr(cfg_obj, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        if key == "lr_scheduler":
            value = None if value in ("", "none", "None") else value
        setattr(cfg_obj, key, value)


@dataclass
class SmokeCfg:
    seed: int = 42
    task: str = 'scarce'
    root: str = 'Dataset'
    limit_train: int = 180
    limit_val: int = 20

    # training
    batch_size: int = 2
    epochs: int = 100
    hidden: int = 128
    layers: int = 14
    lr: float = 4e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    amp: bool = False
    dropout: float = 0.1
    scheduler_step_per_batch: bool = False

    # lr scheduler: 'cosine', 'cosine_warm_restarts', 'reduce_on_plateau', or None
    lr_scheduler: str = 'cosine'
    cosine_T_max: int = 80
    cosine_eta_min: float = 1e-6
    wr_T_0: int = 10
    wr_T_mult: int = 1
    wr_eta_min: float = 1e-6
    rop_factor: float = 0.5
    rop_patience: int = 5
    rop_min_lr: float = 1e-6

    # Physics-Informed Loss Configuration
    ramp_start_epoch: int = 40
    ramp_epochs: int = 60
    ramp_mode: str = 'linear'

    data_loss_weight: float = 1.0
    continuity_loss_weight: float = 0.05
    continuity_target_weight: float = 0.20
    momentum_loss_weight: float = 0.05
    momentum_target_weight: float = 0.20
    bc_loss_weight: float = 0.1

    chord_length: float = 1.0
    nu_molecular: float = 1.5e-5
    dynamic_uref_from_data: bool = True
    dynamic_re_from_data: bool = True
    uinf_from: str = 'inlet'

    use_huber_for_physics: bool = True
    huber_delta: float = 0.05
    use_perimeter_norm_for_div: bool = True
    div_area_floor_factor: float = 0.25
    div_min_degree: int = 2

    physics_debug: bool = False
    physics_debug_level: int = 1
    physics_debug_every: int = 50

    # Global Context & Attention
    use_global_tokens: bool = True
    num_global_tokens: int = 2
    attention_heads: int = 2
    attention_layers: int = 2
    attention_dropout: float = 0.0
    use_cross_attention: bool = True
    global_pooling_type: str = 'attention'
    positional_encoding: bool = False
    pos_encoding_max_len: int = 50000
    use_residual_attention: bool = True
    attention_normalization: str = 'layer'
    temperature_scaling: bool = False
    attention_bias: bool = False

    # W&B Artifact management
    use_wandb_artifacts: bool = False
    artifact_save_best_only: bool = True
    artifact_save_interval: int = 50

    # Checkpoint management
    ckpt_dir: str = "checkpoints"
    ckpt_interval: int = 5

    # W&B settings
    wandb_project: str = "airfrans-gnn"
    enable_wandb: bool = True
    wandb_mode: str = "online"
    wandb_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[list[str]] = None
    log_every_n_steps: int = -1
    log_epoch_only: bool = True


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train AirfRANS GNN model with physics-informed loss',
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
    parser.add_argument('--layers', type=int, default=14,
                        help='Number of message passing layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')

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
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--ckpt-interval', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Weights & Biases
    parser.add_argument('--wandb-project', type=str, default='airfrans-gnn',
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
    parser.set_defaults(enable_wandb=False)
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=None,
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

    return parser.parse_args(argv)


def create_config_from_args(args):
    """Create SmokeCfg from command line arguments"""
    parser_defaults = vars(parse_args([]))
    cfg = SmokeCfg()
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
