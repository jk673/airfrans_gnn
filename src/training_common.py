"""Backward-compatibility shim — re-exports from config, data, and training modules.

All code has been split into:
  - src/config.py:   SmokeCfg, load_config_file, apply_config_dict, DEFAULT_CONFIG_PATH
  - src/data.py:     StandardScaler, NormalizedDataset, PreparePhysics, DataBundle, load_and_prepare_data, collate_pyg
  - src/training.py: get_lr, set_seed, compute_loss_with_physics, run_epoch, train_epoch, create_lr_scheduler, init_wandb
"""

# config
from src.config import SmokeCfg, DEFAULT_CONFIG_PATH, load_config_file, apply_config_dict  # noqa: F401

# data
from src.data import (  # noqa: F401
    StandardScaler,
    NormalizedDataset,
    PreparePhysics,
    DataBundle,
    collate_pyg,
    load_and_prepare_data,
)

# training
from src.training import (  # noqa: F401
    get_lr,
    set_seed,
    compute_loss_with_physics,
    run_epoch,
    train_epoch,
    create_lr_scheduler,
    init_wandb,
)
