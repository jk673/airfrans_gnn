# AirfRANS 2D Airfoil - GNN Surrogate

Auto-generated benchmark comparison. Updated: 2026-03-02 11:19:28 UTC

## Benchmark Comparison

| Model | Volume Rel.L₂ ↓ | Surface Rel.L₂ ↓ | CD Rel.Err ↓ | CL Rel.Err ↓ | ρ_D ↑ | ρ_L ↑ |
|---|---|---|---|---|---|---|
| Transolver | 0.0100 | 0.0352 | 0.6316 | 0.1122 | 0.8750 | 0.9946 |
| FLOW-GLIDE | 0.0038 | 0.0063 | 0.5072 | 0.1029 | 0.9286 | 0.9964 |
| **EXP_0001** (scarce-h16-l14) | 0.6696 | 0.7074 | 4.3956 | 2.7865 | 0.7880 | 0.7564 |
| **EXP_0002** (scarce-h16-l14) | 0.5669 | 0.5996 | 5.3528 | 2.4234 | 0.8632 | 0.9414 |
| **EXP_0003** (scarce-h16-l14) | 0.7512 | 0.7799 | 2.7251 | 2.6412 | 0.6271 | 0.6361 |
| **EXP_0004** (scarce-h8-l4) | 0.9936 | 0.9654 | 3.4215 | 1.2168 | 0.0647 | -0.5504 |
| **EXP_0005** (scarce-h16-l14) | 0.6430 | 0.6778 | 4.0872 | 1.5974 | 0.8647 | 0.8887 |
| **EXP_0006** (scarce-h16-l14) | 0.4100 | 0.4340 | 5.3623 | 0.9358 | 0.8256 | 0.9083 |
| **EXP_0007** (scarce-h16-l14) | 0.3266 | 0.3351 | 3.4650 | 0.3331 | 0.8632 | 0.9714 |
| **EXP_0008** (scarce-h16-l14) | 0.3930 | 0.4127 | 2.7429 | 0.7370 | 0.9068 | 0.8887 |
| **EXP_0009** (scarce-h16-l14) | 0.3826 | 0.3986 | 1.6633 | 0.6192 | 0.8902 | 0.9293 |
| **EXP_0010** (scarce-h16-l14) | 0.4002 | 0.4218 | 1.5452 | 0.7788 | 0.9068 | 0.9128 |
| **EXP_0011** (scarce-h128-l14) | 0.2948 | 0.3075 | 1.5456 | 0.5274 | 0.9624 | 0.9383 |
| **EXP_0012** (scarce-h128-l14) | 0.6616 | 0.7141 | 2.5309 | 0.9676 | 0.9188 | 0.9308 |
| **EXP_0013** (scarce-h128-l14) | 0.7136 | 0.7198 | 1.8224 | 2.2324 | 0.8406 | 0.9098 |
| **EXP_0014** (scarce-h128-l14) | 0.9978 | 0.9653 | 2.4793 | 2.0110 | 0.1654 | 0.6451 |

---

## Experiment Details

### EXP_0014 — 2026-03-02 11:19:28 UTC

**Model:** scarce-h128-l14 | **Task:** scarce | **Parameters:** 3,256,580 | **Duration:** 20s

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.9978 |
| surface_rel_l2 | 0.9653 |
| cd_relative_error | 2.4793 |
| cl_relative_error | 2.0110 |
| rho_d | 0.1654 |
| rho_l | 0.6451 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.3685 |
| best_epoch | 0 |
| finalloss | 0.7896 |
| final_val_loss | 0.3685 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 128,
  "layers": 14,
  "scheduler": {
    "type": "ReduceLROnPlateau",
    "params": {
      "scheduler_T_0": 10,
      "scheduler_T_max": 100,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.1,
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
    "first_lr": 0.001,
    "final_lr": 1e-06
  }
}
```

</details>

---

### EXP_0013 — 2026-03-02 11:18:17 UTC

**Model:** scarce-h128-l14 | **Task:** scarce | **Parameters:** 3,256,580 | **Duration:** 33s

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.7136 |
| surface_rel_l2 | 0.7198 |
| cd_relative_error | 1.8224 |
| cl_relative_error | 2.2324 |
| rho_d | 0.8406 |
| rho_l | 0.9098 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.2553 |
| best_epoch | 0 |
| finalloss | N/A |
| final_val_loss | 0.2553 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 128,
  "layers": 14,
  "scheduler": {
    "type": "ReduceLROnPlateau",
    "params": {
      "scheduler_T_0": 10,
      "scheduler_T_max": 100,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.1,
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
    "first_lr": 0.001,
    "final_lr": 1e-06
  }
}
```

</details>

---

### EXP_0012 — 2026-03-02 11:17:33 UTC

**Model:** scarce-h128-l14 | **Task:** scarce | **Parameters:** 3,256,580 | **Duration:** 1m 55s

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.6616 |
| surface_rel_l2 | 0.7141 |
| cd_relative_error | 2.5309 |
| cl_relative_error | 0.9676 |
| rho_d | 0.9188 |
| rho_l | 0.9308 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.1255 |
| best_epoch | 6 |
| finalloss | 0.3033 |
| final_val_loss | 0.1255 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 128,
  "layers": 14,
  "scheduler": {
    "type": "ReduceLROnPlateau",
    "params": {
      "scheduler_T_0": 10,
      "scheduler_T_max": 100,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.1,
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
    "first_lr": 0.001,
    "final_lr": 1e-06
  }
}
```

</details>

---

### EXP_0011 — 2026-03-02 10:52:24 UTC

**Model:** scarce-h128-l14 | **Task:** scarce | **Parameters:** 3,256,580 | **Duration:** 18m 47s

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.2948 |
| surface_rel_l2 | 0.3075 |
| cd_relative_error | 1.5456 |
| cl_relative_error | 0.5274 |
| rho_d | 0.9624 |
| rho_l | 0.9383 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.0471 |
| best_epoch | 49 |
| finalloss | 0.1808 |
| final_val_loss | 0.0603 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 128,
  "layers": 14,
  "scheduler": {
    "type": "CosineAnnealingLR",
    "params": {
      "scheduler_T_0": 10,
      "scheduler_T_max": 100,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0.0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.1,
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
    "first_lr": 0.0009997532801828658,
    "final_lr": 0.0
  }
}
```

</details>

---

### EXP_0010 — 2026-02-19 11:50:22 UTC

**Model:** scarce-h16-l14 | **Task:** scarce | **Parameters:** 54,052 | **Duration:** 12m 36s

**Notes:** momentum + bc loss + continuity loss

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.4002 |
| surface_rel_l2 | 0.4218 |
| cd_relative_error | 1.5452 |
| cl_relative_error | 0.7788 |
| rho_d | 0.9068 |
| rho_l | 0.9128 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.1685 |
| best_epoch | 253 |
| finalloss | 0.3136 |
| final_val_loss | 0.1705 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 16,
  "layers": 14,
  "scheduler": {
    "type": "CosineAnnealingWarmRestarts",
    "params": {
      "scheduler_T_0": 100,
      "scheduler_T_max": 100,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0.0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.1,
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
    "first_lr": 0.0009997532801828658,
    "final_lr": 0.001
  }
}
```

</details>

---

### EXP_0009 — 2026-02-19 11:34:45 UTC

**Model:** scarce-h16-l14 | **Task:** scarce | **Parameters:** 54,052 | **Duration:** 8m 23s

**Notes:** momentum + bc loss + continuity loss

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.3826 |
| surface_rel_l2 | 0.3986 |
| cd_relative_error | 1.6633 |
| cl_relative_error | 0.6192 |
| rho_d | 0.8902 |
| rho_l | 0.9293 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.1804 |
| best_epoch | 157 |
| finalloss | 0.3444 |
| final_val_loss | 0.1867 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 16,
  "layers": 14,
  "scheduler": {
    "type": "CosineAnnealingWarmRestarts",
    "params": {
      "scheduler_T_0": 100,
      "scheduler_T_max": 100,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0.0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.1,
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
    "first_lr": 0.0009997532801828658,
    "final_lr": 0.001
  }
}
```

</details>

---

### EXP_0008 — 2026-02-19 11:22:38 UTC

**Model:** scarce-h16-l14 | **Task:** scarce | **Parameters:** 54,052 | **Duration:** 8m 27s

**Notes:** momentum + bc loss

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.3930 |
| surface_rel_l2 | 0.4127 |
| cd_relative_error | 2.7429 |
| cl_relative_error | 0.7370 |
| rho_d | 0.9068 |
| rho_l | 0.8887 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.1321 |
| best_epoch | 157 |
| finalloss | 0.2910 |
| final_val_loss | 0.1377 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 16,
  "layers": 14,
  "scheduler": {
    "type": "CosineAnnealingWarmRestarts",
    "params": {
      "scheduler_T_0": 100,
      "scheduler_T_max": 100,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0.0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.1,
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
    "first_lr": 0.0009997532801828658,
    "final_lr": 0.001
  }
}
```

</details>

---

### EXP_0007 — 2026-02-19 11:13:44 UTC

**Model:** scarce-h16-l14 | **Task:** scarce | **Parameters:** 54,052 | **Duration:** 8m 19s

**Notes:** data loss only

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.3266 |
| surface_rel_l2 | 0.3351 |
| cd_relative_error | 3.4650 |
| cl_relative_error | 0.3331 |
| rho_d | 0.8632 |
| rho_l | 0.9714 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.0708 |
| best_epoch | 162 |
| finalloss | 0.1876 |
| final_val_loss | 0.0833 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 16,
  "layers": 14,
  "scheduler": {
    "type": "CosineAnnealingWarmRestarts",
    "params": {
      "scheduler_T_0": 100,
      "scheduler_T_max": 100,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0.0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.1,
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
    "first_lr": 0.0009997532801828658,
    "final_lr": 0.001
  }
}
```

</details>

---

### EXP_0006 — 2026-02-19 11:04:52 UTC

**Model:** scarce-h16-l14 | **Task:** scarce | **Parameters:** 54,052 | **Duration:** 8m 14s

**Notes:** momentum only 

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.4100 |
| surface_rel_l2 | 0.4340 |
| cd_relative_error | 5.3623 |
| cl_relative_error | 0.9358 |
| rho_d | 0.8256 |
| rho_l | 0.9083 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.1396 |
| best_epoch | 170 |
| finalloss | 0.3761 |
| final_val_loss | 0.1464 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 16,
  "layers": 14,
  "scheduler": {
    "type": "CosineAnnealingWarmRestarts",
    "params": {
      "scheduler_T_0": 100,
      "scheduler_T_max": 100,
      "scheduler_T_mult": 1,
      "scheduler_eta_min": 0.0,
      "scheduler_factor": 0.5,
      "scheduler_gamma": 0.1,
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
    "first_lr": 0.0009997532801828658,
    "final_lr": 0.001
  }
}
```

</details>

---

### EXP_0005 — 2026-02-19 00:30:40 UTC

**Model:** scarce-h16-l14 | **Task:** scarce | **Parameters:** 54,052 | **Duration:** 51s

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.6430 |
| surface_rel_l2 | 0.6778 |
| cd_relative_error | 4.0872 |
| cl_relative_error | 1.5974 |
| rho_d | 0.8647 |
| rho_l | 0.8887 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.3894 |
| best_epoch | 18 |
| finalloss | 0.4302 |
| final_val_loss | 0.4194 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 16,
  "layers": 14
}
```

</details>

---

### EXP_0004 — 2026-02-19 00:01:50 UTC

**Model:** scarce-h8-l4 | **Task:** scarce | **Parameters:** 4,340 | **Duration:** 5s

**Notes:** physics loss test

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.9936 |
| surface_rel_l2 | 0.9654 |
| cd_relative_error | 3.4215 |
| cl_relative_error | 1.2168 |
| rho_d | 0.0647 |
| rho_l | -0.5504 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.7374 |
| best_epoch | 2 |
| finalloss | 1.1612 |
| final_val_loss | 0.7374 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 8,
  "layers": 4
}
```

</details>

---

### EXP_0003 — 2026-02-18 23:54:30 UTC

**Model:** scarce-h16-l14 | **Task:** scarce | **Parameters:** 54,052 | **Duration:** 25s

**Notes:** Momentum loss on

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.7512 |
| surface_rel_l2 | 0.7799 |
| cd_relative_error | 2.7251 |
| cl_relative_error | 2.6412 |
| rho_d | 0.6271 |
| rho_l | 0.6361 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.3244 |
| best_epoch | 9 |
| finalloss | 0.5154 |
| final_val_loss | 0.3244 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 16,
  "layers": 14
}
```

</details>

---

### EXP_0002 — 2026-02-18 23:53:50 UTC

**Model:** scarce-h16-l14 | **Task:** scarce | **Parameters:** 54,052 | **Duration:** 1m 31s

**Notes:** Momentum loss on

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.5669 |
| surface_rel_l2 | 0.5996 |
| cd_relative_error | 5.3528 |
| cl_relative_error | 2.4234 |
| rho_d | 0.8632 |
| rho_l | 0.9414 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.1679 |
| best_epoch | 33 |
| finalloss | 0.3390 |
| final_val_loss | 0.1794 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 16,
  "layers": 14
}
```

</details>

---

### EXP_0001 — 2026-02-18 23:46:36 UTC

**Model:** scarce-h16-l14 | **Task:** scarce | **Parameters:** 54,052 | **Duration:** 49s

**Notes:** dd

| Benchmark Metric | Value |
|--------|-------|
| volume_rel_l2 | 0.6696 |
| surface_rel_l2 | 0.7074 |
| cd_relative_error | 4.3956 |
| cl_relative_error | 2.7865 |
| rho_d | 0.7880 |
| rho_l | 0.7564 |

| Training Metric | Value |
|--------|-------|
| status | completed |
| best_val_loss | 0.2348 |
| best_epoch | 18 |
| finalloss | 0.4207 |
| final_val_loss | 0.2349 |
| artifacts_uploaded | 0 |

<details><summary>Config</summary>

```json
{
  "task": "scarce",
  "hidden": 16,
  "layers": 14
}
```

</details>

---
