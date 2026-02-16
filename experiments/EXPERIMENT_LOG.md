# AirfRANS 2D Airfoil - GNN Surrogate

Auto-generated benchmark comparison. Updated: 2026-02-16 12:01:13 UTC

## Benchmark Comparison

| Model | Volume Rel.L₂ ↓ | Surface Rel.L₂ ↓ | CD Rel.Err ↓ | CL Rel.Err ↓ | ρ_D ↑ | ρ_L ↑ |
|---|---|---|---|---|---|---|
| Transolver | 0.0100 | 0.0352 | 0.6316 | 0.1122 | 0.8750 | 0.9946 |
| FLOW-GLIDE | 0.0038 | 0.0063 | 0.5072 | 0.1029 | 0.9286 | 0.9964 |
| **EXP_0003** (scarce-h32-l14) | 0.9944 | 0.9955 | 1.3344 | 1.7454 | 0.0254 | -0.0024 |
| **EXP_0004** (scarce-h32-l14) | 0.9942 | 0.9953 | 1.3367 | 1.8498 | 0.0559 | 0.0072 |

---

## Experiment Details

### EXP_0004 — 2026-02-16 12:01:13 UTC

**Model:** scarce-h32-l14 | **Task:** scarce | **Parameters:** 231,876 | **Duration:** 57s

**Notes:** task=scarce, hidden=32, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.9942 |
| surface_rel_l2 | 0.9953 |
| cd_relative_error | 1.3367 |
| cl_relative_error | 1.8498 |
| rho_d | 0.0559 |
| rho_l | 0.0072 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 26.1700 |
| best_epoch | 4 |
| finalloss | 48.6809 |
| final_val_loss | 26.1700 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 16,
  "epochs": 5,
  "hidden": 32,
  "layers": 14,
  "lr": 0.0004,
  "weight_decay": 0.01,
  "betas": [
    0.9,
    0.95
  ],
  "eps": 1e-08,
  "amp": false,
  "dropout": 0.1,
  "scheduler_step_per_batch": false,
  "lr_scheduler": "cosine",
  "cosine_T_max": 80,
  "cosine_eta_min": 1e-06,
  "wr_T_0": 10,
  "wr_T_mult": 1,
  "wr_eta_min": 1e-06,
  "rop_factor": 0.5,
  "rop_patience": 5,
  "rop_min_lr": 1e-06,
  "ramp_start_epoch": 40,
  "ramp_epochs": 60,
  "ramp_mode": "linear",
  "data_loss_weight": 1.0,
  "continuity_loss_weight": 0.05,
  "continuity_target_weight": 0.2,
  "momentum_loss_weight": 0.05,
  "momentum_target_weight": 0.2,
  "bc_loss_weight": 0.1,
  "chord_length": 1.0,
  "nu_molecular": 1.5e-05,
  "dynamic_uref_from_data": true,
  "dynamic_re_from_data": true,
  "uinf_from": "inlet",
  "use_huber_for_physics": true,
  "huber_delta": 0.05,
  "use_perimeter_norm_for_div": true,
  "div_area_floor_factor": 0.25,
  "div_min_degree": 2,
  "physics_debug": false,
  "physics_debug_level": 1,
  "physics_debug_every": 50,
  "use_global_tokens": true,
  "num_global_tokens": 2,
  "attention_heads": 2,
  "attention_layers": 2,
  "attention_dropout": 0.0,
  "use_cross_attention": true,
  "global_pooling_type": "attention",
  "positional_encoding": false,
  "pos_encoding_max_len": 50000,
  "use_residual_attention": true,
  "attention_normalization": "layer",
  "temperature_scaling": false,
  "attention_bias": false,
  "use_wandb_artifacts": false,
  "artifact_save_best_only": true,
  "artifact_save_interval": 50,
  "ckpt_dir": "checkpoints",
  "ckpt_interval": 5,
  "wandb_project": "airfrans-gnn",
  "enable_wandb": false,
  "wandb_mode": "online",
  "wandb_name": null,
  "wandb_run_name": null,
  "wandb_tags": null,
  "log_every_n_steps": -1,
  "log_epoch_only": true
}
```

</details>

---

### EXP_0003 — 2026-02-16 11:54:02 UTC

**Model:** scarce-h32-l14 | **Task:** scarce | **Parameters:** 231,876 | **Duration:** 23s

**Notes:** task=scarce, hidden=32, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.9944 |
| surface_rel_l2 | 0.9955 |
| cd_relative_error | 1.3344 |
| cl_relative_error | 1.7454 |
| rho_d | 0.0254 |
| rho_l | -0.0024 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 58.9141 |
| best_epoch | 0 |
| finalloss | 176.6693 |
| final_val_loss | 58.9141 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 2,
  "limit_val": 1,
  "batch_size": 1,
  "epochs": 1,
  "hidden": 32,
  "layers": 14,
  "lr": 0.0004,
  "weight_decay": 0.01,
  "betas": [
    0.9,
    0.95
  ],
  "eps": 1e-08,
  "amp": false,
  "dropout": 0.1,
  "scheduler_step_per_batch": false,
  "lr_scheduler": "cosine",
  "cosine_T_max": 80,
  "cosine_eta_min": 1e-06,
  "wr_T_0": 10,
  "wr_T_mult": 1,
  "wr_eta_min": 1e-06,
  "rop_factor": 0.5,
  "rop_patience": 5,
  "rop_min_lr": 1e-06,
  "ramp_start_epoch": 40,
  "ramp_epochs": 60,
  "ramp_mode": "linear",
  "data_loss_weight": 1.0,
  "continuity_loss_weight": 0.05,
  "continuity_target_weight": 0.2,
  "momentum_loss_weight": 0.05,
  "momentum_target_weight": 0.2,
  "bc_loss_weight": 0.1,
  "chord_length": 1.0,
  "nu_molecular": 1.5e-05,
  "dynamic_uref_from_data": true,
  "dynamic_re_from_data": true,
  "uinf_from": "inlet",
  "use_huber_for_physics": true,
  "huber_delta": 0.05,
  "use_perimeter_norm_for_div": true,
  "div_area_floor_factor": 0.25,
  "div_min_degree": 2,
  "physics_debug": false,
  "physics_debug_level": 1,
  "physics_debug_every": 50,
  "use_global_tokens": true,
  "num_global_tokens": 2,
  "attention_heads": 2,
  "attention_layers": 2,
  "attention_dropout": 0.0,
  "use_cross_attention": true,
  "global_pooling_type": "attention",
  "positional_encoding": false,
  "pos_encoding_max_len": 50000,
  "use_residual_attention": true,
  "attention_normalization": "layer",
  "temperature_scaling": false,
  "attention_bias": false,
  "use_wandb_artifacts": false,
  "artifact_save_best_only": true,
  "artifact_save_interval": 50,
  "ckpt_dir": "checkpoints",
  "ckpt_interval": 5,
  "wandb_project": "airfrans-gnn",
  "enable_wandb": false,
  "wandb_mode": "online",
  "wandb_name": null,
  "wandb_run_name": null,
  "wandb_tags": null,
  "log_every_n_steps": -1,
  "log_epoch_only": true
}
```

</details>

---

### EXP_0002 — 2026-02-16 11:53:19 UTC

**Model:** scarce-h32-l14 | **Task:** scarce | **Parameters:** 231,876 | **Duration:** 23s

**Notes:** task=scarce, hidden=32, layers=14, lr=0.0004, scheduler=cosine

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 58.9141 |
| best_epoch | 0 |
| finalloss | 176.6693 |
| final_val_loss | 58.9141 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 2,
  "limit_val": 1,
  "batch_size": 1,
  "epochs": 1,
  "hidden": 32,
  "layers": 14,
  "lr": 0.0004,
  "weight_decay": 0.01,
  "betas": [
    0.9,
    0.95
  ],
  "eps": 1e-08,
  "amp": false,
  "dropout": 0.1,
  "scheduler_step_per_batch": false,
  "lr_scheduler": "cosine",
  "cosine_T_max": 80,
  "cosine_eta_min": 1e-06,
  "wr_T_0": 10,
  "wr_T_mult": 1,
  "wr_eta_min": 1e-06,
  "rop_factor": 0.5,
  "rop_patience": 5,
  "rop_min_lr": 1e-06,
  "ramp_start_epoch": 40,
  "ramp_epochs": 60,
  "ramp_mode": "linear",
  "data_loss_weight": 1.0,
  "continuity_loss_weight": 0.05,
  "continuity_target_weight": 0.2,
  "momentum_loss_weight": 0.05,
  "momentum_target_weight": 0.2,
  "bc_loss_weight": 0.1,
  "chord_length": 1.0,
  "nu_molecular": 1.5e-05,
  "dynamic_uref_from_data": true,
  "dynamic_re_from_data": true,
  "uinf_from": "inlet",
  "use_huber_for_physics": true,
  "huber_delta": 0.05,
  "use_perimeter_norm_for_div": true,
  "div_area_floor_factor": 0.25,
  "div_min_degree": 2,
  "physics_debug": false,
  "physics_debug_level": 1,
  "physics_debug_every": 50,
  "use_global_tokens": true,
  "num_global_tokens": 2,
  "attention_heads": 2,
  "attention_layers": 2,
  "attention_dropout": 0.0,
  "use_cross_attention": true,
  "global_pooling_type": "attention",
  "positional_encoding": false,
  "pos_encoding_max_len": 50000,
  "use_residual_attention": true,
  "attention_normalization": "layer",
  "temperature_scaling": false,
  "attention_bias": false,
  "use_wandb_artifacts": false,
  "artifact_save_best_only": true,
  "artifact_save_interval": 50,
  "ckpt_dir": "checkpoints",
  "ckpt_interval": 5,
  "wandb_project": "airfrans-gnn",
  "enable_wandb": false,
  "wandb_mode": "online",
  "wandb_name": null,
  "wandb_run_name": null,
  "wandb_tags": null,
  "log_every_n_steps": -1,
  "log_epoch_only": true
}
```

</details>

---

### EXP_0001 — 2026-02-16 11:51:58 UTC

**Model:** scarce-h32-l14 | **Task:** scarce | **Parameters:** 231,876 | **Duration:** 52s

**Notes:** task=scarce, hidden=32, layers=14, lr=0.0004, scheduler=cosine

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 26.3555 |
| best_epoch | 4 |
| finalloss | 9.1791 |
| final_val_loss | 26.3555 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 8,
  "epochs": 5,
  "hidden": 32,
  "layers": 14,
  "lr": 0.0004,
  "weight_decay": 0.01,
  "betas": [
    0.9,
    0.95
  ],
  "eps": 1e-08,
  "amp": false,
  "dropout": 0.1,
  "scheduler_step_per_batch": false,
  "lr_scheduler": "cosine",
  "cosine_T_max": 80,
  "cosine_eta_min": 1e-06,
  "wr_T_0": 10,
  "wr_T_mult": 1,
  "wr_eta_min": 1e-06,
  "rop_factor": 0.5,
  "rop_patience": 5,
  "rop_min_lr": 1e-06,
  "ramp_start_epoch": 40,
  "ramp_epochs": 60,
  "ramp_mode": "linear",
  "data_loss_weight": 1.0,
  "continuity_loss_weight": 0.05,
  "continuity_target_weight": 0.2,
  "momentum_loss_weight": 0.05,
  "momentum_target_weight": 0.2,
  "bc_loss_weight": 0.1,
  "chord_length": 1.0,
  "nu_molecular": 1.5e-05,
  "dynamic_uref_from_data": true,
  "dynamic_re_from_data": true,
  "uinf_from": "inlet",
  "use_huber_for_physics": true,
  "huber_delta": 0.05,
  "use_perimeter_norm_for_div": true,
  "div_area_floor_factor": 0.25,
  "div_min_degree": 2,
  "physics_debug": false,
  "physics_debug_level": 1,
  "physics_debug_every": 50,
  "use_global_tokens": true,
  "num_global_tokens": 2,
  "attention_heads": 2,
  "attention_layers": 2,
  "attention_dropout": 0.0,
  "use_cross_attention": true,
  "global_pooling_type": "attention",
  "positional_encoding": false,
  "pos_encoding_max_len": 50000,
  "use_residual_attention": true,
  "attention_normalization": "layer",
  "temperature_scaling": false,
  "attention_bias": false,
  "use_wandb_artifacts": false,
  "artifact_save_best_only": true,
  "artifact_save_interval": 50,
  "ckpt_dir": "checkpoints",
  "ckpt_interval": 5,
  "wandb_project": "airfrans-gnn",
  "enable_wandb": false,
  "wandb_mode": "online",
  "wandb_name": null,
  "wandb_run_name": null,
  "wandb_tags": null,
  "log_every_n_steps": -1,
  "log_epoch_only": true
}
```

</details>

---
