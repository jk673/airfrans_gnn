# AirfRANS 2D Airfoil - GNN Surrogate

Auto-generated benchmark comparison. Updated: 2026-02-16 03:34:39 UTC

## Benchmark Comparison

| Model | Volume Rel.L₂ ↓ | Surface Rel.L₂ ↓ | CD Rel.Err ↓ | CL Rel.Err ↓ | ρ_D ↑ | ρ_L ↑ |
|---|---|---|---|---|---|---|
| Transolver | 0.0100 | 0.0352 | 0.6316 | 0.1122 | 0.8750 | 0.9946 |
| FLOW-GLIDE | 0.0038 | 0.0063 | 0.5072 | 0.1029 | 0.9286 | 0.9964 |
| **EXP_0001** (Test A) | 0.9945 | 0.9956 | 1.2860 | 1.6101 | 0.0271 | -0.0067 |
| **EXP_0002** (Test B) | 0.9936 | 0.9947 | 1.2872 | 1.4626 | 0.0289 | -0.0136 |

---

## Experiment Details

### EXP_0002 — 2026-02-16 03:34:39 UTC

**Model:** Test B | **Task:** scarce | **Parameters:** 562,820 | **Duration:** 15m 16s

**Notes:** hidden=64, layers=8, lr=1e-3 (higher lr+capacity)

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.9936 |
| surface_rel_l2 | 0.9947 |
| cd_relative_error | 1.2872 |
| cl_relative_error | 1.4626 |
| rho_d | 0.0289 |
| rho_l | -0.0136 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| finalloss | 41.8082 |
| final_val_loss | 35.2524 |
| best_val_loss | 35.2524 |
| best_epoch | 99 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 10,
  "epochs": 100,
  "hidden": 64,
  "layers": 8,
  "lr": 0.001,
  "weight_decay": 0.01,
  "betas": [
    0.9,
    0.95
  ],
  "eps": 1e-08,
  "amp": false,
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
  "wandb_mode": "disabled",
  "wandb_name": null,
  "wandb_tags": null,
  "log_every_n_steps": -1,
  "log_epoch_only": true
}
```

</details>

---

### EXP_0001 — 2026-02-16 03:18:59 UTC

**Model:** Test A | **Task:** scarce | **Parameters:** 143,172 | **Duration:** 12m 24s

**Notes:** hidden=32, layers=8, baseline config

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.9945 |
| surface_rel_l2 | 0.9956 |
| cd_relative_error | 1.2860 |
| cl_relative_error | 1.6101 |
| rho_d | 0.0271 |
| rho_l | -0.0067 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| finalloss | 42.1238 |
| final_val_loss | 35.4139 |
| best_val_loss | 35.4139 |
| best_epoch | 99 |

<details><summary>Config</summary>

```json
{
  "seed": 42,
  "task": "scarce",
  "root": "Dataset",
  "limit_train": 180,
  "limit_val": 20,
  "batch_size": 10,
  "epochs": 100,
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
  "wandb_mode": "disabled",
  "wandb_name": null,
  "wandb_tags": null,
  "log_every_n_steps": -1,
  "log_epoch_only": true
}
```

</details>

---
