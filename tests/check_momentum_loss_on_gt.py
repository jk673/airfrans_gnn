import argparse
import os
import sys
import torch
from typing import Optional

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from navier_stokes_physics_loss import NavierStokesPhysicsLoss
from preprocess_airfrans_edges import Params, build_edges_for_graph

try:
    from torch_geometric.datasets import AirfRANS
    from torch_geometric.data import Data
except Exception as e:
    print(f"ERROR: PyG is required to run this script: {e}")
    sys.exit(1)


def find_fields(d: 'Data') -> Optional[torch.Tensor]:
    """Attempt to construct [u, v, p, nu_t] tensor from Data if available.

    Returns an [N, K] tensor K>=3 or None if velocity/pressure are not available.
    """
    N = d.pos.size(0)

    # Common attribute names to probe for velocities and pressure
    cand_vel = ['velocity', 'vel', 'U', 'u', 'uvw', 'vel2d']
    cand_p = ['p', 'pressure', 'p_rgh', 'p_over_rho', 'pressure_over_rho']
    cand_nut = ['nut', 'nu_t', 'nu_turb', 'turbulent_viscosity']

    uv = None
    for name in cand_vel:
        if hasattr(d, name):
            val = getattr(d, name)
            if isinstance(val, torch.Tensor) and val.dim() == 2 and val.size(0) == N:
                if val.size(1) >= 2:
                    uv = val[:, :2].to(torch.float32)
                    break
    p = None
    for name in cand_p:
        if hasattr(d, name):
            val = getattr(d, name)
            if isinstance(val, torch.Tensor) and val.dim() == 1 and val.size(0) == N:
                p = val.to(torch.float32)
                break
            if isinstance(val, torch.Tensor) and val.dim() == 2 and val.size(0) == N:
                p = val[:, 0].to(torch.float32)
                break
    nut = torch.zeros(N, dtype=torch.float32)
    for name in cand_nut:
        if hasattr(d, name):
            val = getattr(d, name)
            if isinstance(val, torch.Tensor) and val.size(0) == N:
                nut = (val if val.dim() == 1 else val[:, 0]).to(torch.float32)
                break

    if uv is None or p is None:
        return None

    return torch.stack([uv[:, 0], uv[:, 1], p, nut], dim=1)


def main():
    ap = argparse.ArgumentParser(description="Check momentum loss on ground-truth fields")
    ap.add_argument('--root', type=str, default='Dataset', help='AirfRANS root folder')
    ap.add_argument('--task', type=str, default='scarce', help='AirfRANS task (scarce/full)')
    ap.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    ap.add_argument('--limit', type=int, default=3, help='Number of graphs to check')
    ap.add_argument('--reynolds', type=float, default=1e6)
    ap.add_argument('--global-radius', type=float, default=0.02)
    ap.add_argument('--surface-radius', type=float, default=0.01)
    ap.add_argument('--neighbors', type=int, default=64)
    args = ap.parse_args()

    # Load dataset
    try:
        ds = AirfRANS(root=args.root, task=args.task, train=(args.split == 'train'))
    except Exception as e:
        print(f"ERROR: Failed to load AirfRANS from {args.root}: {e}")
        sys.exit(1)

    count = min(len(ds), max(1, args.limit))
    print(f"Loaded AirfRANS split={args.split} size={len(ds)}; checking first {count} graphs")

    # Physics loss instance
    phys = NavierStokesPhysicsLoss(
        data_loss_weight=0.0,
        continuity_loss_weight=1.0,
        momentum_loss_weight=1.0,
        reynolds_number=args.reynolds,
        use_skew_symmetric_convection=True,
        chord_length=1.0,
        freestream_velocity=1.0,
    )

    # Edge params
    p = Params(
        root=args.root,
        preset='scarce',
        task=args.task,
        include_test=False,
        global_radius=args.global_radius,
        surface_radius=args.surface_radius,
        max_num_neighbors=args.neighbors,
        surface_ring=False,
        output_dir='processed_edges',
        rebuild=False,
        limit=None,
        workers=0,
        use_processes=False,
        aoa_min=None,
        aoa_max=None,
        aoa_index=2,
        filter_contains=None,
        sequential=True,
        chunk_size=16,
        mem_highwater=85.0,
        gc_interval=20,
        max_active_futures=0,
    )

    gt_losses = []
    cont_losses = []

    missing_fields = 0

    for i in range(count):
        d = ds[i]
        # Build edges and dx,dy features if not present
        try:
            if not hasattr(d, 'edge_index') or d.edge_index is None or d.edge_index.numel() == 0 or not hasattr(d, 'edge_attr_dxdy'):
                d = build_edges_for_graph(d, p)
        except Exception as e:
            print(f"[#{i}] WARN: failed to build edges: {e}")
            continue

        gt = find_fields(d)
        if gt is None:
            missing_fields += 1
            print(f"[#{i}] Ground-truth velocity/pressure not found on this Data object; skipping momentum check.")
            continue

        # Scale features like training flow
        # Use d.pos if available (AirfRANS has pos in meters)
        pos = d.pos if hasattr(d, 'pos') else (d.x[:, :3] if hasattr(d, 'x') else None)
        if pos is None:
            print(f"[#{i}] No positions found; skipping.")
            continue

        gt_scaled, _, pos_scaled = phys.apply_dimensional_scaling(gt, gt, pos)

        try:
            mloss = phys.momentum_loss_single_graph(gt_scaled, d)
            closs = phys.continuity_loss_single_graph(gt_scaled, d)
        except Exception as e:
            print(f"[#{i}] ERROR computing losses: {e}")
            continue

        gt_losses.append(mloss.detach().cpu())
        cont_losses.append(closs.detach().cpu())
        print(f"[#{i}] momentum_loss(gt)={mloss.item():.6e}  continuity_loss(gt)={closs.item():.6e}")

    if not gt_losses:
        print("No graphs with ground-truth velocity/pressure found. Likely your current AirfRANS task does not expose velocities (e.g., 'scarce' predicts Cp and WSS only). In that case, momentum loss on model outputs is ill-defined.")
        if missing_fields > 0:
            print(f"Checked {count} graphs; {missing_fields} lacked (u,v,p). Try task='full' or ensure dataset includes velocity + pressure fields.")
        sys.exit(0)

    ml = torch.stack(gt_losses).mean().item()
    cl = torch.stack(cont_losses).mean().item()
    print("\nSummary:")
    print(f"  Mean momentum_loss(gt) over {len(gt_losses)} graphs: {ml:.6e}")
    print(f"  Mean continuity_loss(gt) over {len(cont_losses)} graphs: {cl:.6e}")
    print("Expected: both should be near zero; significant positive values indicate residual/discretization mismatch or scaling issues.")


if __name__ == '__main__':
    main()
