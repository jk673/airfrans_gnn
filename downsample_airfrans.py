#!/usr/bin/env python
"""
Downsample AirfRANS graphs (step 1 of 2).

Saves per-graph .pt files containing node-wise tensors (x, y, pos, optional surf) AFTER downsampling.
No edges are created here.

Defaults match the notebook-equivalent adaptive voxel with surface-preserving behavior.
"""
import os, math, argparse
from typing import Optional

import torch
from torch_geometric.datasets import AirfRANS
from torch_geometric.data import Data
from torch_geometric.nn.pool import voxel_grid
from tqdm import tqdm


def estimate_chord_length(pos: torch.Tensor) -> float:
    if pos.size(1) >= 2:
        c = float(pos[:, 0].max() - pos[:, 0].min())
        if c > 0:
            return c
    mins, _ = pos.min(dim=0)
    maxs, _ = pos.max(dim=0)
    return float((maxs - mins).norm().item())


def get_surface_mask(d: Data) -> torch.Tensor:
    if hasattr(d, 'surf') and isinstance(d.surf, torch.Tensor) and d.surf.dtype == torch.bool:
        return d.surf.view(-1)
    x = d.x
    if x is not None and x.size(1) >= 5:
        wall = x[:, 2]
        nxy = x[:, 3:5]
        return (wall < 1e-6) | (nxy.abs().sum(dim=1) > 0)
    elif x is not None and x.size(1) >= 3:
        wall = x[:, 2]
        return (wall < 1e-6)
    else:
        return torch.zeros(d.x.size(0), dtype=torch.bool, device=d.x.device)


def voxel_preserve_surface(d: Data, voxel_size: float) -> Data:
    N = d.x.size(0)
    surf = get_surface_mask(d)
    si = torch.nonzero(surf, as_tuple=False).view(-1)
    vi = torch.nonzero(~surf, as_tuple=False).view(-1)
    if vi.numel() == 0:
        return d
    p_all = d.pos if (hasattr(d, 'pos') and d.pos is not None) else d.x[:, :2]
    p2 = p_all[:, :2]
    p2_v = p2[vi].cpu()
    cl = voxel_grid(p2_v, size=float(voxel_size), batch=torch.zeros(p2_v.size(0), dtype=torch.long))
    idx = torch.arange(p2_v.size(0))
    pairs = torch.stack([cl, idx], dim=1)[torch.argsort(cl)]
    pick = torch.ones(pairs.size(0), dtype=torch.bool)
    pick[1:] = pairs[1:, 0] != pairs[:-1, 0]
    keep_v = vi[pairs[pick, 1].to(vi.device)]
    keep = torch.unique(torch.cat([si, keep_v], dim=0), sorted=True)
    new = {}
    for k, v in d:
        if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == N:
            new[k] = v[keep]
        else:
            new[k] = v
    newd = Data(**new)
    newd.subsample_keep_idx = keep
    newd.subsample_voxel_size = float(voxel_size)
    return newd


def adapt_voxel(d: Data, tmin: int, tmax: int, frac: float, iters: int) -> Data:
    pos2 = (d.pos if hasattr(d, 'pos') and d.pos is not None else d.x)[:, :2]
    chord = estimate_chord_length(pos2)
    f = max(1e-5, float(frac))
    best = None
    for _ in range(max(1, int(iters))):
        v = chord * f
        sub = voxel_preserve_surface(d, v)
        n = int(sub.x.size(0))
        if tmin <= n <= tmax:
            return sub
        mid = 0.5 * (tmin + tmax)
        if n > 0:
            f = min(1.0, max(1e-5, f * math.sqrt(n / max(1.0, mid))))
        if best is None or abs(n - mid) < abs(best.x.size(0) - mid):
            best = sub
    return best if best is not None else d


def main():
    ap = argparse.ArgumentParser(description='Downsample AirfRANS graphs (no edges).')
    ap.add_argument('--root', type=str, required=True)
    ap.add_argument('--task', type=str, default='scarce', choices=['scarce', 'full'])
    ap.add_argument('--out-dir', type=str, default='downsampled_graphs')
    ap.add_argument('--limit-train', type=int, default=None)
    ap.add_argument('--limit-test', type=int, default=None)
    # Notebook-equivalent defaults
    ap.add_argument('--target-min-nodes', type=int, default=15000)
    ap.add_argument('--target-max-nodes', type=int, default=30000)
    ap.add_argument('--voxel-frac', type=float, default=0.01)
    ap.add_argument('--voxel-iters', type=int, default=5)
    args = ap.parse_args()

    try:
        ds_train = AirfRANS(root=args.root, train=True, task=args.task)
        ds_test = AirfRANS(root=args.root, train=False, task=args.task)
    except TypeError:
        ds_train = AirfRANS(root=args.root, train=True)
        ds_test = AirfRANS(root=args.root, train=False)

    if args.limit_train is not None:
        from torch.utils.data import Subset
        ds_train = Subset(ds_train, list(range(min(args.limit_train, len(ds_train)))))
    if args.limit_test is not None:
        from torch.utils.data import Subset
        ds_test = Subset(ds_test, list(range(min(args.limit_test, len(ds_test)))))

    out_root = os.path.join(args.out_dir, args.task)
    out_train = os.path.join(out_root, 'train')
    out_test = os.path.join(out_root, 'test')
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    def _run_split(ds, out_dir: str) -> int:
        saved = 0
        for i in tqdm(range(len(ds)), desc=f'Downsample -> {out_dir}'):
            d = ds[i]
            d2 = Data(**{k: v for k, v in d})
            d2 = adapt_voxel(d2, args.target_min_nodes, args.target_max_nodes, args.voxel_frac, args.voxel_iters)
            # Save minimal fields (no edges yet)
            keep = {}
            for k, v in d2:
                if k in ('x', 'y', 'pos', 'surf'):
                    keep[k] = v
            # Always include original dataset index to guarantee downstream alignment
            keep['orig_index'] = torch.tensor(int(i), dtype=torch.long)
            torch.save(Data(**keep), os.path.join(out_dir, f'graph_{i:06d}.pt'))
            saved += 1
        return saved

    n_tr = _run_split(ds_train, out_train)
    n_te = _run_split(ds_test, out_test)
    print(f'Saved downsampled: train={n_tr} test={n_te} under {out_root}')


if __name__ == '__main__':
    main()
