# AirfRANS 2D Airfoil - GNN Surrogate

Auto-generated benchmark comparison. Updated: 2026-03-02 23:18:41 UTC

## Benchmark Comparison

| Model | Volume Rel.L₂ ↓ | Surface Rel.L₂ ↓ | CD Rel.Err ↓ | CL Rel.Err ↓ | ρ_D ↑ | ρ_L ↑ |
|---|---|---|---|---|---|---|
| Transolver | 0.0100 | 0.0352 | 0.6316 | 0.1122 | 0.8750 | 0.9946 |
| FLOW-GLIDE | 0.0038 | 0.0063 | 0.5072 | 0.1029 | 0.9286 | 0.9964 |
| **EXP_0001** (scarce-h64-l8) | 0.6405 | 0.7011 | 1.7329 | 0.7744 | 0.8977 | 0.9444 |

---

## Experiment Details

### EXP_0001 — 2026-03-02 23:18:41 UTC

**Model:** scarce-h64-l8 | **Task:** scarce | **Parameters:** 471,940 | **Duration:** 2m 22s

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.6405 |
| surface_rel_l2 | 0.7011 |
| cd_relative_error | 1.7329 |
| cl_relative_error | 0.7744 |
| rho_d | 0.8977 |
| rho_l | 0.9444 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.0758 |
| best_epoch | 17 |
| finalloss | 0.1963 |
| final_val_loss | 0.1138 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 64,
  "layers": 8,
  "scheduler": {
    "type": "StepLR",
    "params": {
      "scheduler_T_0": 10,
      "scheduler_T_max": 80,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.66,
      "scheduler_milestones": "30,60,90",
      "scheduler_min_lr": 1e-06,
      "scheduler_patience": 10,
      "scheduler_step_mode": "epoch",
      "scheduler_step_size": 10,
      "scheduler_warmup_end_factor": 1.0,
      "scheduler_warmup_start_factor": 0.1,
      "scheduler_warmup_steps": 0
    },
    "step_mode": "epoch",
    "warmup_steps": 0,
    "first_lr": 0.0009667421268223949,
    "final_lr": 0.00042111287044383524
  }
}
```

</details>

---
