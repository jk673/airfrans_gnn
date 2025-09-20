import os
import glob
import torch

from physics_loss import div_from_edge_flux


def _find_one_graph(root_dirs):
    for root in root_dirs:
        for split in ("train", "test"):
            pat = os.path.join(root, split, "graph_*.pt")
            files = sorted(glob.glob(pat))
            if files:
                return files[0]
    return None


def test_denormalized_continuity_near_zero_if_edge_dxdy_available():
    # Try common locations
    cand = _find_one_graph([
        os.path.join("downsampled_graphs", "scarce"),
        os.path.join("downsampled_graphs", "full"),
        os.path.join("processed_edges", "scarce"),
        os.path.join("processed_edges", "full"),
    ])
    if cand is None:
        # Skip if no files are found
        return

    d = torch.load(cand, map_location="cpu", weights_only=False)
    # Ensure necessary fields
    if not hasattr(d, 'edge_index') or d.edge_index is None:
        return

    # Prefer physical pos
    pos = getattr(d, 'pos', None)
    x = getattr(d, 'x', None)
    if pos is None and x is not None:
        # fallback to x
        pos = x[:, :3]

    # Prefer dxdy schema
    edge_attr = getattr(d, 'edge_attr_dxdy', None)
    if edge_attr is None:
        edge_attr = d.edge_attr

    N = (x.size(0) if isinstance(x, torch.Tensor) else (pos.size(0) if isinstance(pos, torch.Tensor) else 0))
    if N == 0:
        return
    # Construct a constant velocity field in physical units
    vel = torch.zeros(N, 2)
    vel[:, 0] = 10.0
    vel[:, 1] = -7.0

    div = div_from_edge_flux(
        velocity=vel,
        edge_index=d.edge_index,
        edge_attr=edge_attr,
        num_nodes=N,
        pos=pos
    )

    # Expect near zero (constant field)
    assert torch.isfinite(div).all()
    assert div.abs().mean().item() < 1e-3
