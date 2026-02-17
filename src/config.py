"""Configuration dataclass and config file helpers for AirfRANS GNN training."""

from __future__ import annotations

import json
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
