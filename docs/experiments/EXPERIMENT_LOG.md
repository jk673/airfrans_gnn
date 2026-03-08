# AirfRANS 2D Airfoil - GNN Surrogate

Auto-generated benchmark comparison. Updated: 2026-03-08 14:13:11 UTC

## Benchmark Comparison

| Model | Volume Rel.L₂ ↓ | Surface Rel.L₂ ↓ | CD Rel.Err ↓ | CL Rel.Err ↓ | ρ_D ↑ | ρ_L ↑ |
|---|---|---|---|---|---|---|
| Transolver | 0.0100 | 0.0352 | 0.6316 | 0.1122 | 0.8750 | 0.9946 |
| FLOW-GLIDE | 0.0038 | 0.0063 | 0.5072 | 0.1029 | 0.9286 | 0.9964 |
| **EXP_0001** (scarce-h48-l10) | 0.1464 | 0.1608 | 1.0097 | 0.3249 | 0.9549 | 0.9880 |

---

## Experiment Details

### EXP_0001 — 2026-03-08 14:13:11 UTC

**Model:** scarce-h48-l10 | **Task:** scarce | **Parameters:** 384,580 | **Duration:** 35m 46s

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.1464 |
| surface_rel_l2 | 0.1608 |
| cd_relative_error | 1.0097 |
| cl_relative_error | 0.3249 |
| rho_d | 0.9549 |
| rho_l | 0.9880 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.0033 |
| best_epoch | 179 |
| finalloss | 0.0040 |
| final_val_loss | 0.0037 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 48,
  "layers": 10,
  "scheduler": {
    "type": "ReduceLROnPlateau",
    "params": {
      "scheduler_T_0": 10,
      "scheduler_T_max": 190,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.76,
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
    "first_lr": 0.0010560685583058836,
    "final_lr": 1e-06
  }
}
```

</details>

---
