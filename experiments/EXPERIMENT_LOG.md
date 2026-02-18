# AirfRANS 2D Airfoil - GNN Surrogate

Auto-generated benchmark comparison. Updated: 2026-02-18 04:13:23 UTC

## Benchmark Comparison

| Model | Volume Rel.L₂ ↓ | Surface Rel.L₂ ↓ | CD Rel.Err ↓ | CL Rel.Err ↓ | ρ_D ↑ | ρ_L ↑ |
|---|---|---|---|---|---|---|
| Transolver | 0.0100 | 0.0352 | 0.6316 | 0.1122 | 0.8750 | 0.9946 |
| FLOW-GLIDE | 0.0038 | 0.0063 | 0.5072 | 0.1029 | 0.9286 | 0.9964 |
| **EXP_0001** (scarce-h32-l14) | 0.9943 | 0.9954 | 1.2413 | 1.7829 | 0.1187 | -0.0182 |
| **EXP_0002** (scarce-h64-l14) | 0.5575 | 0.5878 | 2.7262 | 0.6075 | 0.7839 | 0.7625 |
| **EXP_0003** (scarce-h64-l14) | 0.3041 | 0.3067 | 2.0603 | 0.4624 | 0.8764 | 0.7659 |
| **EXP_0004** (scarce-h32-l14) | 0.7797 | 0.8275 | 1.5829 | 0.8680 | 0.7861 | 0.4303 |

---

## Experiment Details

### EXP_0004 — 2026-02-18 04:13:23 UTC

**Model:** scarce-h32-l14 | **Task:** scarce | **Parameters:** 232,036 | **Duration:** 8m 20s

**Notes:** task=scarce, hidden=32, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.7797 |
| surface_rel_l2 | 0.8275 |
| cd_relative_error | 1.5829 |
| cl_relative_error | 0.8680 |
| rho_d | 0.7861 |
| rho_l | 0.4303 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.2313 |
| best_epoch | 84 |
| finalloss | 0.4742 |
| final_val_loss | 0.2328 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 9,
  "epochs": 100,
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
  "cosine_T_max": 100,
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
  "cont_ramp_start_epoch": 20,
  "cont_ramp_epochs": 10,
  "mom_ramp_start_epoch": 20,
  "mom_ramp_epochs": 10,
  "bc_ramp_start_epoch": 20,
  "bc_ramp_epochs": 10,
  "data_loss_weight": 1.0,
  "continuity_loss_weight": 0.0,
  "continuity_target_weight": 0.0,
  "momentum_loss_weight": 0.0,
  "momentum_target_weight": 0.01,
  "bc_loss_weight": 0.0,
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

### EXP_0003 — 2026-02-18 04:03:05 UTC

**Model:** scarce-h64-l14 | **Task:** scarce | **Parameters:** 912,580 | **Duration:** 36m 57s

**Notes:** task=scarce, hidden=64, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.3041 |
| surface_rel_l2 | 0.3067 |
| cd_relative_error | 2.0603 |
| cl_relative_error | 0.4624 |
| rho_d | 0.8764 |
| rho_l | 0.7659 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.0539 |
| best_epoch | 198 |
| finalloss | 0.1913 |
| final_val_loss | 0.0656 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 2,
  "epochs": 200,
  "hidden": 64,
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
  "cosine_T_max": 100,
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
  "cont_ramp_start_epoch": 20,
  "cont_ramp_epochs": 10,
  "mom_ramp_start_epoch": 20,
  "mom_ramp_epochs": 10,
  "bc_ramp_start_epoch": 20,
  "bc_ramp_epochs": 10,
  "data_loss_weight": 1.0,
  "continuity_loss_weight": 0.0,
  "continuity_target_weight": 0.0,
  "momentum_loss_weight": 0.0,
  "momentum_target_weight": 0.01,
  "bc_loss_weight": 0.0,
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

### EXP_0002 — 2026-02-18 03:21:45 UTC

**Model:** scarce-h64-l14 | **Task:** scarce | **Parameters:** 912,580 | **Duration:** 4m 8s

**Notes:** task=scarce, hidden=64, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.5575 |
| surface_rel_l2 | 0.5878 |
| cd_relative_error | 2.7262 |
| cl_relative_error | 0.6075 |
| rho_d | 0.7839 |
| rho_l | 0.7625 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.7526 |
| best_epoch | 11 |
| finalloss | 0.2505 |
| final_val_loss | 0.8417 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 2,
  "epochs": 20,
  "hidden": 64,
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
  "cont_ramp_start_epoch": 20,
  "cont_ramp_epochs": 10,
  "mom_ramp_start_epoch": 20,
  "mom_ramp_epochs": 10,
  "bc_ramp_start_epoch": 20,
  "bc_ramp_epochs": 10,
  "data_loss_weight": 1.0,
  "continuity_loss_weight": 0.0,
  "continuity_target_weight": 0.0,
  "momentum_loss_weight": 0.0,
  "momentum_target_weight": 0.01,
  "bc_loss_weight": 0.0,
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

### EXP_0001 — 2026-02-17 07:46:02 UTC

**Model:** scarce-h32-l14 | **Task:** scarce | **Parameters:** 231,876 | **Duration:** 40s

**Notes:** task=scarce, hidden=32, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.9943 |
| surface_rel_l2 | 0.9954 |
| cd_relative_error | 1.2413 |
| cl_relative_error | 1.7829 |
| rho_d | 0.1187 |
| rho_l | -0.0182 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 34.8547 |
| best_epoch | 4 |
| finalloss | 72.0202 |
| final_val_loss | 34.8547 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 24,
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
