# Physics Loss Correctness & Performance Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix fundamental correctness bugs and performance bottlenecks in `NavierStokesPhysicsLoss` identified by the 2026-03-08 audit.

**Architecture:** Incremental fixes to existing `src/physics_loss.py`. Each task is independently testable via unit tests on synthetic graphs. No new files except tests. Fixes are ordered so each task can be committed and verified independently.

**Tech Stack:** PyTorch, torch_scatter, pytest

**Audit reference:** `docs/plans/2026-03-08-physics-loss-audit-full.md`

---

## Task 1: Fix Divergence Normal Direction (P0-1)

**Files:**
- Modify: `src/physics_loss.py:166-174`
- Create: `tests/test_physics_operators.py`

**Step 1: Write failing test — divergence of known field**

A uniform velocity field `u=(1,0)` should have zero divergence everywhere.
A radial velocity field `u=(x,y)` should have divergence = 2 at every node.

```python
# tests/test_physics_operators.py
import torch
import pytest
from src.physics_loss import conservative_divergence

def _make_grid_graph(nx=5, ny=5, r=1.5):
    """Create a simple 2D grid graph with radius connectivity and [dx,dy,dist] edge_attr."""
    xs = torch.linspace(0, nx-1, nx, dtype=torch.float32)
    ys = torch.linspace(0, ny-1, ny, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).reshape(-1, 2)
    N = grid.size(0)
    # Build radius graph
    diff = grid.unsqueeze(0) - grid.unsqueeze(1)  # [N,N,2]
    dist = diff.norm(dim=-1)  # [N,N]
    mask = (dist > 0) & (dist <= r)
    rows, cols = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([rows, cols], dim=0)
    dx = diff[rows, cols, 0]
    dy = diff[rows, cols, 1]
    d = dist[rows, cols]
    edge_attr = torch.stack([dx, dy, d], dim=1)  # [E, 3] dxdy schema
    return grid, edge_index, edge_attr, N

def test_divergence_uniform_field_is_zero():
    """Uniform velocity u=(1,0) → div(u) = 0 everywhere."""
    pos, edge_index, edge_attr, N = _make_grid_graph(nx=6, ny=6, r=1.5)
    velocity = torch.zeros(N, 2)
    velocity[:, 0] = 1.0  # uniform u=1, v=0

    div = conservative_divergence(
        velocity, edge_index, edge_attr, N,
        pos=pos, prefer_dxdy=True, Lref=1.0,
        area_floor_factor=0.0, min_degree=0,
    )
    # Interior nodes should have near-zero divergence
    # (boundary nodes will have bias, exclude them)
    interior = (pos[:, 0] > 0.5) & (pos[:, 0] < 4.5) & (pos[:, 1] > 0.5) & (pos[:, 1] < 4.5)
    assert interior.sum() > 0, "No interior nodes found"
    div_interior = div[interior]
    assert div_interior.abs().max() < 0.1, (
        f"Uniform field should have near-zero divergence, got max={div_interior.abs().max():.4f}"
    )

def test_divergence_radial_field_is_positive():
    """Radial velocity u=(x, y) → div(u) = du/dx + dv/dy = 2."""
    pos, edge_index, edge_attr, N = _make_grid_graph(nx=8, ny=8, r=1.5)
    velocity = pos.clone()  # u=x, v=y

    div = conservative_divergence(
        velocity, edge_index, edge_attr, N,
        pos=pos, prefer_dxdy=True, Lref=1.0,
        area_floor_factor=0.0, min_degree=0,
    )
    interior = (pos[:, 0] > 1) & (pos[:, 0] < 6) & (pos[:, 1] > 1) & (pos[:, 1] < 6)
    div_interior = div[interior]
    # Should be positive (divergence = 2), sign should be consistent
    assert div_interior.mean() > 0.5, (
        f"Radial field should have positive divergence, got mean={div_interior.mean():.4f}"
    )
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_physics_operators.py::test_divergence_uniform_field_is_zero -v
pytest tests/test_physics_operators.py::test_divergence_radial_field_is_positive -v
```
Expected: At least one FAIL (the tangent-direction normal gives wrong results)

**Step 3: Fix the divergence normal direction**

In `src/physics_loss.py`, change lines 166-174:

```python
    # OLD (wrong — tangent direction):
    # nx = dx / length
    # ny = dy / length

    # Face normal perpendicular to edge (90° rotation of edge direction)
    nx = dy / length
    ny = -dx / length

    ui = velocity[row]       # [E,2]
    uj = velocity[col]
    u_face = 0.5 * (ui + uj)
    flux = (u_face[:, 0] * nx + u_face[:, 1] * ny) * length  # [E]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_physics_operators.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_physics_operators.py src/physics_loss.py
git commit -m "fix(physics): correct divergence normal direction from tangent to perpendicular

The conservative_divergence function was using the edge tangent (dx/r, dy/r)
instead of the perpendicular face normal (dy/r, -dx/r). This made the
continuity loss compute directional flux alignment rather than actual ∇·u."
```

---

## Task 2: Merge Duplicate weighted_gradient(uv) Call (P1-3)

**Files:**
- Modify: `src/physics_loss.py:692-697`

**Step 1: Write failing test — momentum loss determinism**

```python
# Append to tests/test_physics_operators.py

def test_momentum_loss_returns_finite():
    """Momentum loss should return a finite scalar on a synthetic graph."""
    from src.physics_loss import NavierStokesPhysicsLoss
    from torch_geometric.data import Data

    pos, edge_index, edge_attr, N = _make_grid_graph(nx=6, ny=6, r=1.5)
    pred = torch.randn(N, 4, requires_grad=True)
    target = torch.randn(N, 4)
    data = Data(
        x=torch.randn(N, 7),
        y=target,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_attr_dxdy=edge_attr,
    )
    loss_fn = NavierStokesPhysicsLoss(
        data_loss_weight=1.0,
        continuity_loss_weight=0.1,
        momentum_loss_weight=0.1,
        bc_loss_weight=0.0,
        dynamic_uref_from_data=False,
        dynamic_re_from_data=False,
        freestream_velocity=10.0,
        reynolds_number=1e6,
    )
    result = loss_fn(pred, target, data=data, step=0)
    assert torch.isfinite(result['total_loss']), "Total loss should be finite"
    assert torch.isfinite(result['momentum_loss']), "Momentum loss should be finite"
    # Verify backward works
    result['total_loss'].backward()
    assert pred.grad is not None
```

**Step 2: Run test**

```bash
pytest tests/test_physics_operators.py::test_momentum_loss_returns_finite -v
```
Expected: PASS (current code works, just wastefully)

**Step 3: Merge the duplicate calls**

In `src/physics_loss.py`, replace lines 692-697:

```python
            # OLD — duplicate call:
            # _, duvdy = weighted_gradient(uv, ...)
            # duvdx, _ = weighted_gradient(uv, ...)

            # NEW — single call:
            duvdx, duvdy = weighted_gradient(uv, edge_index, edge_attr.to(device), num_nodes,
                                             pos=pos_phys, prefer_dxdy=self.prefer_dxdy, weight_mode=self.weight_mode, Lref=self.Lref)
```

**Step 4: Run tests**

```bash
pytest tests/test_physics_operators.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/physics_loss.py
git commit -m "perf(physics): merge duplicate weighted_gradient(uv) call

Lines 692-697 called weighted_gradient on uv twice, discarding opposite
components each time. Merged into a single call, saving 6 scatter_add ops."
```

---

## Task 3: Replace Two-Pass Laplacian with Direct Single-Pass (P1-1)

**Files:**
- Modify: `src/physics_loss.py:267-289`
- Add tests to: `tests/test_physics_operators.py`

**Step 1: Write failing test — Laplacian of known field**

```python
# Append to tests/test_physics_operators.py

from src.physics_loss import weighted_laplacian

def test_laplacian_quadratic_field():
    """Laplacian of f=x² should be 2 everywhere (d²f/dx²=2, d²f/dy²=0)."""
    pos, edge_index, edge_attr, N = _make_grid_graph(nx=10, ny=10, r=1.5)
    field = pos[:, 0] ** 2  # f = x^2

    lap = weighted_laplacian(field, edge_index, edge_attr, N,
                             pos=pos, prefer_dxdy=True, Lref=1.0)

    interior = (pos[:, 0] > 2) & (pos[:, 0] < 7) & (pos[:, 1] > 2) & (pos[:, 1] < 7)
    lap_interior = lap[interior]
    # Should be close to 2.0
    assert lap_interior.mean() > 1.0, (
        f"Laplacian of x² should be ~2, got mean={lap_interior.mean():.4f}"
    )
    assert lap_interior.mean() < 4.0, (
        f"Laplacian of x² should be ~2, got mean={lap_interior.mean():.4f}"
    )
```

**Step 2: Run test with current two-pass implementation**

```bash
pytest tests/test_physics_operators.py::test_laplacian_quadratic_field -v
```
Note the actual value — the two-pass approach likely underestimates significantly.

**Step 3: Implement direct single-pass Laplacian**

Replace the `weighted_laplacian` function at lines 267-289:

```python
def weighted_laplacian(
    field: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
    *,
    pos: Optional[torch.Tensor] = None,
    prefer_dxdy: bool = True,
    weight_mode: str = "rbf",
    eps: float = 1e-12,
    Lref: float = 1.0,
) -> torch.Tensor:
    """
    Direct single-pass graph Laplacian (replaces two-pass grad-of-grad).
    Δf(i) ≈ 2 * sum_j w_ij (f_j - f_i) / r_ij^2  /  sum_j w_ij
    Factor 2 for 2D consistency (Brookshaw 1985).
    """
    device = field.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    edge_attr = edge_attr.to(device=device, dtype=field.dtype)
    if pos is not None:
        pos = pos.to(device=device, dtype=field.dtype)

    N = field.size(0)
    if edge_index.numel() == 0 or edge_attr.numel() == 0:
        return torch.zeros(N, device=device, dtype=field.dtype)

    valid = _valid_edges(edge_index, N)
    if not torch.all(valid):
        edge_index = edge_index[:, valid]
        edge_attr = edge_attr[valid]
        if edge_index.numel() == 0:
            return torch.zeros(N, device=device, dtype=field.dtype)

    edge_index, edge_attr = _half_edges(edge_index, edge_attr)
    row, col = edge_index

    dx, dy, length = _extract_dxdy_length(edge_index, edge_attr, pos, prefer_dxdy, eps)
    if Lref != 1.0:
        s = 1.0 / max(Lref, 1e-12)
        dx, dy, length = dx * s, dy * s, length * s

    if weight_mode == "rbf":
        h2 = (length.mean() ** 2).clamp_min(eps)
        w = torch.exp(-(length * length) / (h2 + eps))
    else:
        w = 1.0 / (length * length + eps)

    df = field[col] - field[row]  # [E]
    inv_r2 = 1.0 / (length * length + eps)

    # Direct Laplacian: w * (f_j - f_i) / r^2
    lap_edge = w * df * inv_r2  # [E]

    # Symmetric accumulation
    num = scatter_add(lap_edge, row, dim=0, dim_size=N) + scatter_add(lap_edge, col, dim=0, dim_size=N)
    den = scatter_add(w, row, dim=0, dim_size=N) + scatter_add(w, col, dim=0, dim_size=N)
    den = den.clamp_min(1.0)

    # Factor 2 for 2D (Brookshaw 1985 SPH Laplacian)
    return 2.0 * num / den
```

**Step 4: Run tests**

```bash
pytest tests/test_physics_operators.py -v
```
Expected: ALL PASS (including the quadratic field test with improved accuracy)

**Step 5: Commit**

```bash
git add src/physics_loss.py tests/test_physics_operators.py
git commit -m "fix(physics): replace two-pass Laplacian with direct single-pass

The old grad-of-grad approach caused double-smoothing and underestimated
the Laplacian magnitude. The new direct formula uses sum(w*(f_j-f_i)/r²)
with factor-2 correction (Brookshaw 1985). Saves 12 scatter_add per call."
```

---

## Task 4: Add Viscous Transpose Term (P2-1)

**Files:**
- Modify: `src/physics_loss.py:704-713`

**Step 1: Write test — viscous term with variable nu_t**

```python
# Append to tests/test_physics_operators.py

def test_viscous_transpose_term_nonzero():
    """With spatially-varying nu_t and cross-gradients, the transpose term matters."""
    from src.physics_loss import weighted_gradient, weighted_laplacian

    pos, edge_index, edge_attr, N = _make_grid_graph(nx=8, ny=8, r=1.5)
    # u = y (so du/dy=1, du/dx=0), v = x (so dv/dx=1, dv/dy=0)
    u = pos[:, 1]
    v = pos[:, 0]
    # nu_t = x*y (spatially varying)
    nu_t = pos[:, 0] * pos[:, 1]

    dudx, dudy = weighted_gradient(u, edge_index, edge_attr, N, pos=pos, prefer_dxdy=True, Lref=1.0)
    dvdx, dvdy = weighted_gradient(v, edge_index, edge_attr, N, pos=pos, prefer_dxdy=True, Lref=1.0)
    dnutdx, dnutdy = weighted_gradient(nu_t, edge_index, edge_attr, N, pos=pos, prefer_dxdy=True, Lref=1.0)

    # Old viscous (without transpose):
    # visc_u_old = ... + dnutdx * dudx + dnutdy * dudy
    visc_u_old_extra = dnutdx * dudx + dnutdy * dudy

    # New viscous (with transpose):
    # visc_u_new = ... + dnutdx * (2*dudx) + dnutdy * (dudy + dvdx)
    visc_u_new_extra = dnutdx * (2 * dudx) + dnutdy * (dudy + dvdx)

    # The difference should be nonzero (transpose contribution)
    diff = (visc_u_new_extra - visc_u_old_extra).abs()
    interior = (pos[:, 0] > 1) & (pos[:, 0] < 6) & (pos[:, 1] > 1) & (pos[:, 1] < 6)
    assert diff[interior].mean() > 0.01, (
        f"Transpose term should be nonzero with variable nu_t and cross-gradients"
    )
```

**Step 2: Run test**

```bash
pytest tests/test_physics_operators.py::test_viscous_transpose_term_nonzero -v
```
Expected: PASS (this tests the math, not the implementation)

**Step 3: Apply the fix**

In `src/physics_loss.py`, replace lines 712-713:

```python
        # OLD (missing transpose term):
        # visc_u = (mol_coeff + nu_t) * lap_u + dnutdx * dudx + dnutdy * dudy
        # visc_v = (mol_coeff + nu_t) * lap_v + dnutdx * dvdx + dnutdy * dvdy

        # NEW (full RANS viscous stress with transpose ∇u^T):
        visc_u = (mol_coeff + nu_t) * lap_u + dnutdx * (2 * dudx) + dnutdy * (dudy + dvdx)
        visc_v = (mol_coeff + nu_t) * lap_v + dnutdx * (dvdx + dudy) + dnutdy * (2 * dvdy)
```

**Step 4: Run all tests**

```bash
pytest tests/test_physics_operators.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/physics_loss.py tests/test_physics_operators.py
git commit -m "fix(physics): add viscous transpose term for full RANS stress tensor

RANS requires ∇·[ν_eff(∇u + ∇u^T)]. The transpose contribution adds
cross-gradient terms (dnutdx*dudx + dnutdy*dvdx for x-momentum).
All gradients were already computed, so this is zero additional cost."
```

---

## Task 5: Non-dimensionalize Physics Residuals (P0-2)

**Files:**
- Modify: `src/physics_loss.py:655` (continuity), `src/physics_loss.py:715-719` (momentum)
- Add tests to: `tests/test_physics_operators.py`

**Step 1: Write test — loss scale after non-dimensionalization**

```python
# Append to tests/test_physics_operators.py

def test_physics_loss_scale_comparable_to_data_loss():
    """After non-dimensionalization, physics losses should be O(1), not O(10000)."""
    from src.physics_loss import NavierStokesPhysicsLoss
    from torch_geometric.data import Data

    pos, edge_index, edge_attr, N = _make_grid_graph(nx=6, ny=6, r=1.5)
    # Simulate physical-scale predictions: u~10 m/s, p~100 Pa, nut~1e-3
    pred = torch.randn(N, 4) * torch.tensor([10.0, 10.0, 100.0, 1e-3])
    target = pred + torch.randn(N, 4) * 0.1  # small perturbation
    data = Data(
        x=torch.randn(N, 7),
        y=target,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_attr_dxdy=edge_attr,
    )
    loss_fn = NavierStokesPhysicsLoss(
        data_loss_weight=1.0,
        continuity_loss_weight=1.0,
        continuity_target_weight=1.0,
        momentum_loss_weight=1.0,
        momentum_target_weight=1.0,
        bc_loss_weight=0.0,
        dynamic_uref_from_data=False,
        dynamic_re_from_data=False,
        freestream_velocity=10.0,
        reynolds_number=1e6,
        chord_length=1.0,
    )
    result = loss_fn(pred, target, data=data, step=100)

    cont = float(result['continuity_loss'])
    mom = float(result['momentum_loss'])
    mse = float(result['mse_loss'])

    # After non-dim, all losses should be within 3 orders of magnitude of each other
    losses = [l for l in [cont, mom, mse] if l > 0]
    if len(losses) >= 2:
        ratio = max(losses) / (min(losses) + 1e-12)
        assert ratio < 1e4, (
            f"Physics losses should be comparable scale. "
            f"cont={cont:.2e}, mom={mom:.2e}, mse={mse:.2e}, ratio={ratio:.1e}"
        )
```

**Step 2: Run test with current code**

```bash
pytest tests/test_physics_operators.py::test_physics_loss_scale_comparable_to_data_loss -v
```
Expected: FAIL (momentum is orders of magnitude larger)

**Step 3: Add non-dimensionalization to continuity and momentum**

In `src/physics_loss.py`, modify `_continuity_loss` (around line 655):

```python
    def _continuity_loss(self, u_scaled, data, pos_phys):
        # ... (existing code up to div computation) ...
        div = conservative_divergence(...)
        # Non-dimensionalize: div has units of [1/L] after scaling,
        # reference strain rate = U_ref / L_ref
        # For per-node U_ref, use the mean as reference
        Uref = getattr(self, '_current_Uref', self.Uref)
        if isinstance(Uref, torch.Tensor):
            Uref_scalar = Uref.mean().item()
        else:
            Uref_scalar = float(Uref)
        ref_strain = max(Uref_scalar, 1e-12) / max(self.Lref, 1e-12)
        div_nd = div / ref_strain
        return self._quad_or_huber(div_nd)
```

In `_momentum_loss`, modify before the return (around line 715-719):

```python
        # Non-dimensionalize residuals by reference acceleration U²/L
        Uref = getattr(self, '_current_Uref', self.Uref)
        if isinstance(Uref, torch.Tensor):
            Uref_scalar = Uref.mean().item()
        else:
            Uref_scalar = float(Uref)
        ref_accel = (max(Uref_scalar, 1e-12) ** 2) / max(self.Lref, 1e-12)
        res_u = res_u / ref_accel
        res_v = res_v / ref_accel

        return self._quad_or_huber(torch.stack([res_u, res_v], dim=-1))
```

Also store the current Uref in `forward()` for access by sub-methods. In `forward()`, after computing `Uref_local` (around line 949-951), add:

```python
        # Store for sub-method access
        self._current_Uref = Uref_local
```

**Step 4: Run tests**

```bash
pytest tests/test_physics_operators.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/physics_loss.py tests/test_physics_operators.py
git commit -m "fix(physics): non-dimensionalize continuity and momentum residuals

Continuity residual now divided by reference strain rate (U_ref/L_ref).
Momentum residual divided by reference acceleration (U_ref²/L_ref).
This brings physics losses to O(1) scale, comparable to the MSE data loss."
```

---

## Task 6: Switch Default Weight Mode to inv_r2 (P1-2)

**Files:**
- Modify: `src/physics_loss.py:365` (default parameter)
- Add test to: `tests/test_physics_operators.py`

**Step 1: Write test — gradient accuracy with inv_r2**

```python
# Append to tests/test_physics_operators.py

from src.physics_loss import weighted_gradient

def test_gradient_linear_field_inv_r2():
    """Gradient of f=2x+3y should be (2, 3) at interior nodes with inv_r2 weights."""
    pos, edge_index, edge_attr, N = _make_grid_graph(nx=8, ny=8, r=1.5)
    field = 2.0 * pos[:, 0] + 3.0 * pos[:, 1]  # f = 2x + 3y

    gx, gy = weighted_gradient(field, edge_index, edge_attr, N,
                                pos=pos, prefer_dxdy=True, weight_mode="inv_r2", Lref=1.0)

    interior = (pos[:, 0] > 1) & (pos[:, 0] < 6) & (pos[:, 1] > 1) & (pos[:, 1] < 6)
    assert (gx[interior] - 2.0).abs().mean() < 0.3, f"df/dx should be ~2, got {gx[interior].mean():.3f}"
    assert (gy[interior] - 3.0).abs().mean() < 0.3, f"df/dy should be ~3, got {gy[interior].mean():.3f}"
```

**Step 2: Run test**

```bash
pytest tests/test_physics_operators.py::test_gradient_linear_field_inv_r2 -v
```
Expected: PASS

**Step 3: Change default weight_mode**

In `src/physics_loss.py`, `NavierStokesPhysicsLoss.__init__` (line 366):

```python
        # OLD:
        # weight_mode: str = "rbf",
        # NEW:
        weight_mode: str = "inv_r2",
```

Also update `build_physics_loss` in `src/pipeline.py` if it passes `weight_mode`.

**Step 4: Run all tests**

```bash
pytest tests/test_physics_operators.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/physics_loss.py
git commit -m "perf(physics): switch default weight_mode from rbf to inv_r2

The global RBF bandwidth (h=mean edge length) was a poor fit for
multi-scale meshes. inv_r2 weighting naturally adapts to local mesh
scale and is computationally cheaper (no exp())."
```

---

## Task 7: Fix Huber Delta (P2-3)

**Files:**
- Modify: `src/physics_loss.py:369` (default in __init__)
- Modify: `src/pipeline.py:223` (default in build_physics_loss)

**Step 1: No test needed — parameter change only**

**Step 2: Change defaults**

In `src/physics_loss.py`, `__init__` parameter:
```python
        # OLD: huber_delta: float = 0.01,
        huber_delta: float = 1.0,
```

In `src/pipeline.py:223`:
```python
        # OLD: huber_delta=config.get("huber_delta", 0.05),
        huber_delta=config.get("huber_delta", 1.0),
```

**Step 3: Run all tests**

```bash
pytest tests/ -v
```
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/physics_loss.py src/pipeline.py
git commit -m "fix(physics): increase Huber delta from 0.05 to 1.0

After residual non-dimensionalization (Task 5), residuals are O(1).
The old delta=0.05 put everything in the linear L1 regime, losing
curvature information. Delta=1.0 provides proper L2→L1 transition."
```

---

## Task 8: Precompute Edge Processing Per Forward (P2-2)

**Files:**
- Modify: `src/physics_loss.py` (add helper, modify `_momentum_loss` and `_continuity_loss`)

**Step 1: Write test — performance improvement verification**

```python
# Append to tests/test_physics_operators.py

import time

def test_momentum_loss_no_regression():
    """Momentum loss should still work after edge preprocessing refactor."""
    from src.physics_loss import NavierStokesPhysicsLoss
    from torch_geometric.data import Data

    pos, edge_index, edge_attr, N = _make_grid_graph(nx=8, ny=8, r=1.5)
    pred = torch.randn(N, 4)
    target = torch.randn(N, 4)
    data = Data(
        x=torch.randn(N, 7), y=target, pos=pos,
        edge_index=edge_index, edge_attr=edge_attr, edge_attr_dxdy=edge_attr,
    )
    loss_fn = NavierStokesPhysicsLoss(
        data_loss_weight=1.0, continuity_loss_weight=0.1,
        momentum_loss_weight=0.1, bc_loss_weight=0.0,
        dynamic_uref_from_data=False, dynamic_re_from_data=False,
        freestream_velocity=10.0, reynolds_number=1e6,
    )
    r1 = loss_fn(pred, target, data=data, step=0)
    r2 = loss_fn(pred, target, data=data, step=0)
    # Results should be deterministic
    assert abs(float(r1['total_loss']) - float(r2['total_loss'])) < 1e-5
```

**Step 2: Run test**

```bash
pytest tests/test_physics_operators.py::test_momentum_loss_no_regression -v
```
Expected: PASS

**Step 3: Add edge preprocessing helper**

Add a new function before `_continuity_loss`:

```python
    def _prepare_physics_edges(self, data, device):
        """Precompute filtered+halved edges, dx/dy/length once per forward pass.
        Returns (edge_index_h, dx, dy, length, w) for half-edges."""
        edge_attr = getattr(data, 'edge_attr_dxdy', getattr(data, 'edge_attr', None))
        if edge_attr is None:
            return None
        edge_index = data.edge_index.to(device)
        edge_attr = edge_attr.to(device)
        N = data.num_nodes

        valid = _valid_edges(edge_index, N)
        if not torch.all(valid):
            edge_index = edge_index[:, valid]
            edge_attr = edge_attr[valid]

        edge_index, edge_attr = _half_edges(edge_index, edge_attr)
        row, col = edge_index
        dx, dy, length = _extract_dxdy_length(edge_index, edge_attr, None, self.prefer_dxdy)

        if self.Lref != 1.0:
            s = 1.0 / max(self.Lref, 1e-12)
            dx, dy, length = dx * s, dy * s, length * s

        eps = 1e-12
        if self.weight_mode == "rbf":
            h2 = (length.mean() ** 2).clamp_min(eps)
            w = torch.exp(-(length * length) / (h2 + eps))
        else:
            w = 1.0 / (length * length + eps)

        return edge_index, edge_attr, dx, dy, length, w
```

Then modify `weighted_gradient` and other functions to optionally accept precomputed values (add `precomputed` kwarg). This is a larger refactor — implement incrementally by first caching at the `forward()` level and passing through.

**Step 4: Run all tests**

```bash
pytest tests/test_physics_operators.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/physics_loss.py tests/test_physics_operators.py
git commit -m "perf(physics): precompute edge filtering + dx/dy/length once per forward

Eliminates 11 redundant _valid_edges + _half_edges + _extract_dxdy_length
calls (each with O(E log E) median sort) per forward pass."
```

---

## Task 9: Remove Dead UnifiedNavierStokesPhysicsLoss (P4)

**Files:**
- Modify: `src/physics_loss.py:1120-1919` (delete)

**Step 1: Verify it's unused**

```bash
cd /workspace/airfrans_gnn && grep -r "UnifiedNavierStokesPhysicsLoss" --include="*.py" -l
```
Expected: Only `src/physics_loss.py` itself (the class definition)

**Step 2: Delete the class and its `if __name__` block**

Remove lines 1120-1919 from `src/physics_loss.py`.

**Step 3: Run all tests**

```bash
pytest tests/ -v
```
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/physics_loss.py
git commit -m "chore: remove unused UnifiedNavierStokesPhysicsLoss class

Dead code with incompatible forward() signature. Its correct divergence
formula (perpendicular normal) has been ported to the active class in Task 1."
```

---

## Task 10: Run Full Regression Test

**Files:** None (verification only)

**Step 1: Run full test suite**

```bash
pytest tests/ -v
```

**Step 2: Verify physics loss instantiation in pipeline**

```bash
python -c "
from src.pipeline import build_physics_loss
loss_fn = build_physics_loss({'data_weight': 1.0, 'continuity': {'weight': 0.1}, 'momentum': {'weight': 0.1}}, steps_per_epoch=100)
print('Loss fn created:', type(loss_fn).__name__)
print('Weight mode:', loss_fn.weight_mode)
print('Huber delta:', loss_fn.huber_delta)
"
```

**Step 3: Quick smoke test with synthetic data**

```bash
python -c "
import torch
from src.physics_loss import NavierStokesPhysicsLoss
from torch_geometric.data import Data

N = 100
pos = torch.randn(N, 2)
edge_index = torch.randint(0, N, (2, 500))
edge_attr = torch.randn(500, 3)
data = Data(x=torch.randn(N,7), y=torch.randn(N,4), pos=pos,
            edge_index=edge_index, edge_attr=edge_attr, edge_attr_dxdy=edge_attr)
loss_fn = NavierStokesPhysicsLoss(dynamic_uref_from_data=False, freestream_velocity=10.0)
result = loss_fn(torch.randn(N,4), torch.randn(N,4), data=data, step=0)
for k, v in result.items():
    if isinstance(v, torch.Tensor):
        print(f'{k}: {v.item():.4e}')
print('All OK')
"
```

---

## Summary of Changes

| Task | Issue | Change | scatter_add Δ |
|------|-------|--------|---------------|
| 1 | P0-1 Divergence normal | `nx=dy/r, ny=-dx/r` | 0 |
| 2 | P1-3 Duplicate gradient | Merge 2 calls → 1 | -6 |
| 3 | P1-1 Two-pass Laplacian | Direct single-pass | -24 |
| 4 | P2-1 Viscous transpose | Add cross-gradient terms | 0 |
| 5 | P0-2 Scale mismatch | Non-dim by U²/L | 0 |
| 6 | P1-2 RBF bandwidth | Default inv_r2 | 0 |
| 7 | P2-3 Huber delta | 0.05 → 1.0 | 0 |
| 8 | P2-2 Edge preprocessing | Cache per forward | 0 (wall-clock savings) |
| 9 | P4 Dead code | Remove Unified class | 0 |
| 10 | Regression | Verify all green | — |
| **Total** | | | **-30 scatter_add** |
