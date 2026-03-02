# Test Patterns for AirfRANS GNN

## Minimal Graph Fixture

```python
import torch
from torch_geometric.data import Data

@pytest.fixture
def simple_graph():
    N, E = 50, 200
    edge_index = torch.randint(0, N, (2, E))
    return Data(
        x=torch.randn(N, 7),           # 7D node features
        pos=torch.randn(N, 2),         # 2D positions
        edge_index=edge_index,
        edge_attr=torch.randn(E, 3),   # [dx, dy, dist] format
        y=torch.randn(N, 4),           # [u, v, pressure, nu_t]
    )
```

## What to Test per Module Type

### Physics Loss (`src/navier_stokes_physics_loss.py`)
- Uniform flow → zero divergence
- Known pressure gradient → correct momentum residual
- Boundary conditions enforced at wall/inlet/farfield nodes
- Curriculum weight ramping: verify weights at step 0, midpoint, end
- Batch of 2+ graphs → same result as individual graphs

### Edge Preprocessing (`src/preprocess_airfrans_edges.py`)
- Edge symmetry: if (i→j) exists, (j→i) exists
- Degree floor: every node has ≥ min_degree edges
- Length pruning: no edge longer than hard cap
- Schema detection: `[dist, dir_x, dir_y]` vs `[dx, dy, dist]`

### Utils (`src/airfrans_utils.py`)
- `estimate_node_area()`: sum of areas ≈ domain area
- `build_bc_masks_airfrans()`: masks are disjoint, cover boundary nodes
- `weighted_gradient()`: linear field → constant gradient

### Model Components (`src/global_context_processor.py`)
- Output shape matches input batch
- Gradient flows through all parameters
- Works with batch of variable-size graphs

## Edge Attribute Dual Schema

Always test both formats:
```python
@pytest.fixture(params=["dxdy", "dist_dir"])
def edge_attr(request, edge_index):
    E = edge_index.shape[1]
    if request.param == "dxdy":
        dx, dy = torch.randn(E), torch.randn(E)
        dist = (dx**2 + dy**2).sqrt()
        return torch.stack([dx, dy, dist], dim=1)
    else:
        dist = torch.rand(E)
        dir_x, dir_y = torch.randn(E), torch.randn(E)
        return torch.stack([dist, dir_x, dir_y], dim=1)
```
