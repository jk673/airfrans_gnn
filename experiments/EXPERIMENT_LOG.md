# AirfRANS 2D Airfoil - GNN Surrogate

Auto-generated benchmark comparison. Updated: 2026-02-18 23:46:36 UTC

## Benchmark Comparison

| Model | Volume Rel.L₂ ↓ | Surface Rel.L₂ ↓ | CD Rel.Err ↓ | CL Rel.Err ↓ | ρ_D ↑ | ρ_L ↑ |
|---|---|---|---|---|---|---|
| Transolver | 0.0100 | 0.0352 | 0.6316 | 0.1122 | 0.8750 | 0.9946 |
| FLOW-GLIDE | 0.0038 | 0.0063 | 0.5072 | 0.1029 | 0.9286 | 0.9964 |
| **EXP_0001** (scarce-h16-l14) | 0.6696 | 0.7074 | 4.3956 | 2.7865 | 0.7880 | 0.7564 |

---

## Experiment Details

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
