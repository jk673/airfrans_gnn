# AirfRANS 2D Airfoil - GNN Surrogate

Auto-generated benchmark comparison. Updated: 2026-02-19 00:30:40 UTC

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

---

## Experiment Details

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
