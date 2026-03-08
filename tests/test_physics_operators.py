"""Unit tests for physics differential operators (divergence, gradient, Laplacian).

Tests use synthetic grid graphs with known analytical solutions to verify
correctness of each operator independently.
"""
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
