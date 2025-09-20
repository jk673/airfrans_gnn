import torch

# Minimal stand-in for torch_geometric.data.Data to avoid requiring PyG in tests
class Data:
    def __init__(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        self.x = x
        self.pos = pos
        self.edge_index = edge_index
    # Allow attaching arbitrary attributes (like velocity, pred_velocity)
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

from loss_v3 import ComprehensivePhysicsLoss

def make_grid(nx=5, ny=5, h=1.0):
    xs = torch.arange(nx, dtype=torch.float32)
    ys = torch.arange(ny, dtype=torch.float32)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    pos = torch.stack([X.flatten() * h, Y.flatten() * h], dim=1)
    N = pos.size(0)
    # Build 4-neighborhood edges (undirected)
    idx = lambda i, j: i + j * nx
    edges = []
    for j in range(ny):
        for i in range(nx):
            u = idx(i, j)
            if i + 1 < nx:
                v = idx(i + 1, j)
                edges.append((u, v)); edges.append((v, u))
            if j + 1 < ny:
                v = idx(i, j + 1)
                edges.append((u, v)); edges.append((v, u))
    if len(edges) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Pack into Data with pos as 3D (pad z=0), and store a dummy x to satisfy interface
    pos3 = torch.cat([pos, torch.zeros(N, 1)], dim=1)
    x = torch.zeros(N, 8)  # placeholder features, not used here
    data = Data(x=x, pos=pos3, edge_index=edge_index)
    return data


def test_continuity_zero_divergence_gt_only():
    # Create simple grid graph and constant velocity field
    data = make_grid(6, 6, h=0.1)
    N = data.pos.size(0)
    # Provide ground-truth velocity through data.velocity for continuity sanity check
    u = torch.ones(N) * 2.0
    v = torch.ones(N) * -3.0
    data.velocity = torch.stack([u, v], dim=1)

    loss_fn = ComprehensivePhysicsLoss()
    # No predictions available for velocity; ensure compute_continuity_loss returns gt near 0 and no loss
    res = loss_fn.compute_continuity_loss(data, predictions_physical=None, targets_physical=None)
    # GT div should be ~0
    assert res['gt_div_abs_mean'].item() == 0.0 or res['gt_div_abs_mean'].item() < 1e-6
    # No pred velocity -> continuity loss should be exactly 0
    assert res['loss'].item() == 0.0


def test_continuity_penalizes_divergence_pred():
    data = make_grid(6, 6, h=0.1)
    N = data.pos.size(0)
    # Prediction velocity field: u=x, v=y -> divergence = 2 everywhere
    xy = data.pos[:, :2]
    u = xy[:, 0].clone()
    v = xy[:, 1].clone()
    data.pred_velocity = torch.stack([u, v], dim=1)

    loss_fn = ComprehensivePhysicsLoss()
    res = loss_fn.compute_continuity_loss(data, predictions_physical=torch.zeros(N, 4), targets_physical=None)
    # Pred divergence abs mean should be close to 2 (numerical approx)
    assert res['pred_div_abs_mean'].item() > 1.0
    # Loss should be positive
    assert res['loss'].item() > 0.1
