# Physics Loss Audit Report — 2026-03-08

## Executive Summary

5-member research team conducted a comprehensive audit of the physics loss implementation
(`src/physics_loss.py`) covering literature, open-source benchmarks, physics correctness,
ML optimization, and software engineering.

**Critical finding**: The active physics loss class (`NavierStokesPhysicsLoss`) contains
fundamental errors that render the continuity loss physically meaningless and cause
training instability via scale mismatch. Ironically, the unused class
(`UnifiedNavierStokesPhysicsLoss`) has the correct divergence formula but is never called.

---

## Issue Inventory (Priority Order)

### P0 — CRITICAL (Must Fix)

#### P0-1. Divergence Normal Direction — WRONG
- **File**: `src/physics_loss.py:168-169`
- **Problem**: Uses edge tangent `(dx/r, dy/r)` instead of perpendicular face normal `(dy/r, -dx/r)`
- **Impact**: Continuity loss computes directional flux alignment, NOT divergence ∇·u
- **Evidence**: The unused `UnifiedNavierStokesPhysicsLoss:1472` has the correct formula `flux = u*dy - v*dx`
- **Confirmed by**: T3 (physics), T4 (code), T2 (Gen-FVGN uses perpendicular normals)
- **Fix**: 2 lines — `nx = dy/length; ny = -dx/length`

#### P0-2. Loss Scale Mismatch — 2-4 Orders of Magnitude
- **Problem**: Momentum residual O(10,000) vs data loss O(0.01-1.0)
- **Impact**: When curriculum ramp activates, momentum overwhelms data fitting → catastrophic forgetting
- **Root cause**: Momentum residuals not non-dimensionalized by reference acceleration U²/L
- **Confirmed by**: T5 (ML), T4 (code — `UnifiedNavierStokesPhysicsLoss` DOES normalize by `ref_accel`)
- **Fix options**:
  1. Non-dimensionalize residuals: `res /= (U_ref² / L_ref)`
  2. Uncertainty weighting: 3 learnable `nn.Parameter` log-variances (Kendall et al. 2018)
  3. ReLoBRaLo adaptive balancing (gradient-free, Bischof & Kraus 2025)

### P1 — HIGH (Should Fix)

#### P1-1. Two-Pass Laplacian — Excessive Numerical Diffusion
- **File**: `src/physics_loss.py:267-289`
- **Problem**: `grad(grad(f))` via 3 nested `weighted_gradient` calls = 18 scatter_add + double-smoothing
- **Impact**: Viscous term underestimated → physics loss too lenient on non-smooth fields
- **Literature**: PhyMPGN (ICLR 2025) encodes direct Laplace-Beltrami operator
- **Fix**: Single-pass direct Laplacian `sum w*(f_j-f_i)/r² / sum w` — saves 12 scatter_add per call

#### P1-2. Single Global RBF Bandwidth
- **File**: `src/physics_loss.py:247`
- **Problem**: `h² = (length.mean())²` — one bandwidth for wall (h~0.001) AND far-field (h~0.1)
- **Impact**: Near-wall gradients over-smoothed, far-field gradients under-supported
- **Cross-graph contamination in batched mode**
- **Fix**: Per-node bandwidth via scatter, or switch to `weight_mode="inv_r2"`

#### P1-3. Duplicate `weighted_gradient(uv)` Call
- **File**: `src/physics_loss.py:692-697`
- **Problem**: Same computation called twice, discarding opposite components each time
- **Fix**: `duvdx, duvdy = weighted_gradient(uv, ...)` — trivial 3-line fix, saves 6 scatter_add

### P2 — MEDIUM (Should Address)

#### P2-1. Missing ∇u^T in Viscous Stress
- **File**: `src/physics_loss.py:712-713`
- **Problem**: RANS requires `∇·[ν_eff(∇u + ∇u^T)]`, code only has `∇·[ν_eff ∇u]`
- **Missing**: `dνt/dx·du/dx + dνt/dy·dv/dx` (x-mom), `dνt/dx·du/dy + dνt/dy·dv/dy` (y-mom)
- **Note**: `dudx, dvdx, dudy, dvdy` already computed — zero additional gradient calls
- **Fix**:
  ```python
  visc_u = (mol_coeff+nu_t)*lap_u + dnutdx*(2*dudx) + dnutdy*(dudy+dvdx)
  visc_v = (mol_coeff+nu_t)*lap_v + dnutdx*(dvdx+dudy) + dnutdy*(2*dvdy)
  ```

#### P2-2. Redundant Edge Processing (10x per forward)
- **Problem**: `_valid_edges` + `_half_edges` + `_extract_dxdy_length` (with median sort)
  repeated identically ~11 times per forward pass
- **Impact**: 11 unnecessary O(E log E) median sorts on ~200K edges
- **Fix**: Precompute filtered edges + dx/dy/length once at `_momentum_loss` entry

#### P2-3. Huber Delta Miscalibration
- **File**: `src/physics_loss.py:369` (default 0.01), `src/pipeline.py:223` (default 0.05)
- **Problem**: Residuals O(100) with delta=0.05 → everything in linear regime → constant gradients
- **Fix**: After scale normalization, set delta=0.5-1.0

#### P2-4. Simplified Least-Squares Gradient (Scalar Divisor)
- **Problem**: Uses `sum(w*df*d/r²) / sum(w)` instead of full 2x2 matrix solve
- **Impact**: Degrades on anisotropic meshes (stretched boundary layers)
- **Literature**: Gen-FVGN and PhysicsNeMo both solve proper `A*grad=B` systems
- **Fix**: Medium effort — implement 2x2 per-node system via scatter

### P3 — LOW (Nice to Have)

| Issue | Description |
|-------|-------------|
| P3-1 | Precompute RBF weights during data loading (geometry-only, not model-dependent) |
| P3-2 | Cache schema detection result per graph |
| P3-3 | Detach `Uref_local` and `mol_coeff` defensively |
| P3-4 | Batch `weighted_gradient` for multiple fields simultaneously (4→1 calls) |
| P3-5 | Compute physics loss every N steps to amortize cost |
| P3-6 | Dual area estimation `P²/4π` overestimates — use triangle-based when available |

### P4 — CLEANUP

| Issue | Description |
|-------|-------------|
| P4-1 | `UnifiedNavierStokesPhysicsLoss` is dead code with incompatible `forward()` signature |
| P4-2 | Reconcile the two classes: port correct formulas (divergence, direct Laplacian) to active class |

---

## Performance Impact Summary

### Current Forward Pass Cost
- **88 scatter_add operations** per training step
- **~3-4x slowdown** vs pure MSE training
- **~70MB intermediate edge tensors** in autograd graph

### After Proposed Fixes
| Fix | scatter_add Saved | Other Savings |
|-----|-------------------|---------------|
| P1-1 (direct Laplacian) | -24 (12 per Laplacian × 2) | Memory: fewer intermediates |
| P1-3 (merge uv gradient) | -6 | — |
| P2-2 (precompute edges) | — | 11 median sorts eliminated |
| P3-4 (batch gradients) | -18 (4→1 per field group) | — |
| **Total** | **~48 fewer** (88→40) | Significant memory + compute savings |

---

## Literature-Backed Alternatives

### Loss Weighting (Replace Manual Curriculum)
| Method | Paper | Pros | Cons |
|--------|-------|------|------|
| **ReLoBRaLo** | Bischof & Kraus, CMAME 2025 | Gradient-free, efficient | Requires EMA tuning |
| **Uncertainty Weighting** | Kendall et al., CVPR 2018 | 3 learnable params, simple | May need warmup |
| **GradNorm** | Chen et al., 2018 | Theoretically grounded | Expensive (per-task grad norms) |

### Continuity Enforcement (Alternative to Soft Loss)
- **Stream function parameterization**: u=∂ψ/∂y, v=-∂ψ/∂x → divergence-free by construction
- Reference: Neural Conservation Laws (NeurIPS 2022)
- Eliminates continuity loss entirely

### Gradient Computation (Upgrade Path)
- **Full WLSQ with 2x2 solve**: PhysicsNeMo pattern with `torch.linalg.lstsq(A+λI, B)`
- **Gen-FVGN pattern**: Higher-order Taylor expansion with ghost-point boundaries

---

## Key GitHub Repositories for Reference

| Repo | Relevance | Key Pattern |
|------|-----------|-------------|
| [Gen-FVGN-steady](https://github.com/Litianyu141/Gen-FVGN-steady) | FVM-based GNN, WLSQ gradient, ghost boundaries | Conservative flux + `torch.linalg.solve` |
| [NVIDIA/physicsnemo-sym](https://github.com/NVIDIA/physicsnemo-sym) | Regularized LSQ gradient, symbolic PDE | `lstsq(A+λI, B)` with λ=1e-6 |
| [rbischof/relative_balancing](https://github.com/rbischof/relative_balancing) | ReLoBRaLo loss balancing | Adaptive weights without gradient computation |

---

## Recommended Action Plan

### Phase 1: Critical Fixes (Immediate)
1. Fix divergence normal: `nx=dy/r, ny=-dx/r` (P0-1)
2. Non-dimensionalize physics residuals by `U_ref²/L_ref` (P0-2)
3. Merge duplicate `weighted_gradient(uv)` (P1-3)

### Phase 2: Accuracy Improvements
4. Replace two-pass Laplacian with direct single-pass (P1-1)
5. Add viscous transpose term (P2-1, zero-cost since gradients already computed)
6. Switch to `inv_r2` weighting or per-node bandwidth (P1-2)

### Phase 3: Training Stability
7. Implement ReLoBRaLo or uncertainty weighting (P0-2 permanent fix)
8. Fix Huber delta after scale normalization (P2-3)
9. Precompute edge filtering + dx/dy/length per forward (P2-2)

### Phase 4: Performance Optimization
10. Batch `weighted_gradient` for multiple fields (P3-4)
11. Precompute RBF weights at data load time (P3-1)
12. Clean up dead `UnifiedNavierStokesPhysicsLoss` class (P4)

---

## Team Credits
- **T1 (Researcher)**: Literature survey — 30+ papers, arXiv + MDPI + Elsevier
- **T2 (Repo Scouter)**: GitHub analysis — 20+ repos, Gen-FVGN/PhysicsNeMo deep dives
- **T3 (Scientist)**: Physics correctness — divergence normal bug, Laplacian diffusion, ∇u^T omission
- **T4 (Software Engineer)**: Code analysis — 88 scatter_add count, redundant calls, schema detection
- **T5 (ML Engineer)**: ML optimization — scale mismatch quantification, uncertainty weighting, RBF bandwidth
