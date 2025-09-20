import torch
from torch_geometric.data import Data
from typing import cast

from preprocess_airfrans_edges import Params, build_edges_for_graph
from physics_loss import div_from_edge_flux


def make_positions_grid(nx=10, ny=10, h=0.1):
    xs = torch.arange(nx, dtype=torch.float32)
    ys = torch.arange(ny, dtype=torch.float32)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    pos = torch.stack([X.flatten() * h, Y.flatten() * h, torch.zeros(nx*ny)], dim=1)
    return pos


def test_edge_attr_dxdy_schema_and_divergence_constant_field():
    # Create a simple grid in physical units (h=0.1 m)
    pos = make_positions_grid(8, 8, h=0.1)
    N = pos.size(0)

    # Minimal Data with positions
    data = Data(pos=pos, x=torch.zeros(N, 8))

    # Build edges with small radius to connect 4-neighbors
    p = Params(
        root='.', preset='scarce', task='scarce', include_test=False,
        global_radius=0.11, surface_radius=0.11, max_num_neighbors=8,
        surface_ring=False, output_dir='__unused__', rebuild=True, limit=None,
        workers=0, use_processes=False, aoa_min=None, aoa_max=None, aoa_index=2,
        filter_contains=None, sequential=True, chunk_size=1, mem_highwater=100.0,
        gc_interval=1, max_active_futures=0
    )

    data = build_edges_for_graph(data, p)

    # Ensure auxiliary schema exists
    assert hasattr(data, 'edge_attr_dxdy'), 'edge_attr_dxdy not present'
    assert data.edge_attr_dxdy.size(1) == 3

    # Constant velocity field -> zero divergence
    vel = torch.zeros(N, 2)
    vel[:, 0] = 2.0
    vel[:, 1] = -3.0

    # Compute divergence using dx,dy schema
    assert data.edge_index is not None
    assert data.edge_attr is not None
    assert data.edge_attr_dxdy is not None
    div1 = div_from_edge_flux(
        velocity=vel,
        edge_index=cast(torch.Tensor, data.edge_index),
        edge_attr=cast(torch.Tensor, data.edge_attr_dxdy),
        num_nodes=N,
    )

    # Compute divergence using default schema
    div2 = div_from_edge_flux(
        velocity=vel,
        edge_index=cast(torch.Tensor, data.edge_index),
        edge_attr=cast(torch.Tensor, data.edge_attr),
        num_nodes=N,
    )

    # Mean absolute divergence should be near zero (numerical tolerance)
    assert torch.isfinite(div1).all()
    assert torch.isfinite(div2).all()
    assert div1.abs().mean().item() < 1e-5
    assert div2.abs().mean().item() < 1e-5


def test_edge_attr_dxdy_scaling_effect():
    # Create two nodes with a single edge and check scale response
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    data = Data(pos=pos, x=torch.zeros(2, 8))
    p = Params(
        root='.', preset='scarce', task='scarce', include_test=False,
        global_radius=1.1, surface_radius=1.1, max_num_neighbors=8,
        surface_ring=False, output_dir='__unused__', rebuild=True, limit=None,
        workers=0, use_processes=False, aoa_min=None, aoa_max=None, aoa_index=2,
        filter_contains=None, sequential=True, chunk_size=1, mem_highwater=100.0,
        gc_interval=1, max_active_futures=0
    )
    data = build_edges_for_graph(data, p)
    N = pos.size(0)

    vel = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    # Baseline divergence
    assert data.edge_index is not None and data.edge_attr_dxdy is not None
    div = div_from_edge_flux(vel, cast(torch.Tensor, data.edge_index), cast(torch.Tensor, data.edge_attr_dxdy), N)
    base = div.abs().sum().item()

    # Scale edge_attr_dxdy by factor s in dx,dy and dist -> divergence should scale linearly
    s = 5.0
    edge_scaled = data.edge_attr_dxdy.clone()
    edge_scaled[:, :2] *= s
    edge_scaled[:, 2] *= s

    div_s = div_from_edge_flux(vel, cast(torch.Tensor, data.edge_index), edge_scaled, N)
    scaled = div_s.abs().sum().item()

    # Divergence sums scale by s
    if base > 0:
        ratio = scaled / (base + 1e-12)
        assert abs(ratio - s) < 1e-3
