# AirfRANS 2D Airfoil - GNN Surrogate

Auto-generated benchmark comparison. Updated: 2026-02-17 12:33:06 UTC

## Benchmark Comparison

| Model | Volume Rel.L₂ ↓ | Surface Rel.L₂ ↓ | CD Rel.Err ↓ | CL Rel.Err ↓ | ρ_D ↑ | ρ_L ↑ |
|---|---|---|---|---|---|---|
| Transolver | 0.0100 | 0.0352 | 0.6316 | 0.1122 | 0.8750 | 0.9946 |
| FLOW-GLIDE | 0.0038 | 0.0063 | 0.5072 | 0.1029 | 0.9286 | 0.9964 |
| **EXP_0001** (scarce-h32-l14) | 0.9943 | 0.9954 | 1.2413 | 1.7829 | 0.1187 | -0.0182 |
| **EXP_0002** (scarce-h32-l14) | 0.9944 | 0.9955 | 1.3177 | 2.0607 | 0.1463 | -0.0846 |
| **EXP_0003** (scarce-h32-l14) | 0.9951 | 0.9962 | 1.4482 | 2.0044 | 0.0233 | -0.0035 |
| **EXP_0004** (scarce-h32-l8) | 0.9931 | 0.9920 | 1.1468 | 1.0217 | 0.0232 | -0.4048 |
| **EXP_0005** (scarce-h32-l14) | 0.7854 | 0.8202 | 5.8294 | 0.6235 | 0.6881 | 0.6959 |
| **EXP_0006** (scarce-h128-l14) | 0.6470 | 0.6790 | 4.8006 | 0.9296 | 0.7913 | 0.7000 |
| **EXP_0008** (scarce-h128-l14) | 0.5904 | 0.6200 | 3.3403 | 1.2471 | 0.8383 | 0.7535 |

---

## Experiment Details

### EXP_0008 — 2026-02-17 12:33:06 UTC

**Model:** scarce-h128-l14 | **Task:** scarce | **Parameters:** 3,619,204 | **Duration:** 7m 37s

**Notes:** task=scarce, hidden=128, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.5904 |
| surface_rel_l2 | 0.6200 |
| cd_relative_error | 3.3403 |
| cl_relative_error | 1.2471 |
| rho_d | 0.8383 |
| rho_l | 0.7535 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 1.0023 |
| best_epoch | 11 |
| finalloss | 0.2822 |
| final_val_loss | 1.3919 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 10,
  "epochs": 20,
  "hidden": 128,
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

### EXP_0007 — 2026-02-17 12:21:56 UTC

**Model:** scarce-h128-l14 | **Task:** scarce | **Parameters:** 3,619,204 | **Duration:** 7m 40s

**Notes:** task=scarce, hidden=128, layers=14, lr=0.0004, scheduler=cosine

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.1081 |
| best_epoch | 13 |
| finalloss | 0.2711 |
| final_val_loss | 0.1362 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 10,
  "epochs": 20,
  "hidden": 128,
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
  "continuity_loss_weight": 0.0,
  "continuity_target_weight": 0.0,
  "momentum_loss_weight": 0.0,
  "momentum_target_weight": 0.0,
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

### EXP_0006 — 2026-02-17 11:43:42 UTC

**Model:** scarce-h128-l14 | **Task:** scarce | **Parameters:** 3,618,564 | **Duration:** 4m 27s

**Notes:** task=scarce, hidden=128, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.6470 |
| surface_rel_l2 | 0.6790 |
| cd_relative_error | 4.8006 |
| cl_relative_error | 0.9296 |
| rho_d | 0.7913 |
| rho_l | 0.7000 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.1216 |
| best_epoch | 18 |
| finalloss | 0.2977 |
| final_val_loss | 0.1270 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 10,
  "epochs": 20,
  "hidden": 128,
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
  "continuity_loss_weight": 0.0,
  "continuity_target_weight": 0.0,
  "momentum_loss_weight": 0.0,
  "momentum_target_weight": 0.0,
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

### EXP_0005 — 2026-02-17 11:38:03 UTC

**Model:** scarce-h32-l14 | **Task:** scarce | **Parameters:** 231,876 | **Duration:** 2m 17s

**Notes:** task=scarce, hidden=32, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.7854 |
| surface_rel_l2 | 0.8202 |
| cd_relative_error | 5.8294 |
| cl_relative_error | 0.6235 |
| rho_d | 0.6881 |
| rho_l | 0.6959 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.1609 |
| best_epoch | 19 |
| finalloss | 0.4335 |
| final_val_loss | 0.1609 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 10,
  "epochs": 20,
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
  "continuity_loss_weight": 0.0,
  "continuity_target_weight": 0.0,
  "momentum_loss_weight": 0.0,
  "momentum_target_weight": 0.0,
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

### EXP_0004 — 2026-02-17 11:31:22 UTC

**Model:** scarce-h32-l8 | **Task:** scarce | **Parameters:** 143,172 | **Duration:** 1m 55s

**Notes:** task=scarce, hidden=32, layers=8, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.9931 |
| surface_rel_l2 | 0.9920 |
| cd_relative_error | 1.1468 |
| cl_relative_error | 1.0217 |
| rho_d | 0.0232 |
| rho_l | -0.4048 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.6936 |
| best_epoch | 19 |
| finalloss | 1.3420 |
| final_val_loss | 0.6936 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 10,
  "epochs": 20,
  "hidden": 32,
  "layers": 8,
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

### EXP_0003 — 2026-02-17 09:15:28 UTC

**Model:** scarce-h32-l14 | **Task:** scarce | **Parameters:** 231,876 | **Duration:** 5m 28s

**Notes:** task=scarce, hidden=32, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.9951 |
| surface_rel_l2 | 0.9962 |
| cd_relative_error | 1.4482 |
| cl_relative_error | 2.0044 |
| rho_d | 0.0233 |
| rho_l | -0.0035 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 32.6333 |
| best_epoch | 49 |
| finalloss | 7.4486 |
| final_val_loss | 32.6333 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 20,
  "epochs": 50,
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

### EXP_0002 — 2026-02-17 08:40:10 UTC

**Model:** scarce-h32-l14 | **Task:** scarce | **Parameters:** 231,876 | **Duration:** 1m 8s

**Notes:** task=scarce, hidden=32, layers=14, lr=0.0004, scheduler=cosine

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.9944 |
| surface_rel_l2 | 0.9955 |
| cd_relative_error | 1.3177 |
| cl_relative_error | 2.0607 |
| rho_d | 0.1463 |
| rho_l | -0.0846 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 34.4000 |
| best_epoch | 9 |
| finalloss | 72.2494 |
| final_val_loss | 34.4000 |
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
  "epochs": 10,
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
  "ckpt_dir": "checkpoints_ddp_test",
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
