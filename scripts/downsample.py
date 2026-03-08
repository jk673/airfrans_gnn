#!/usr/bin/env python
"""
Downsample AirfRANS graphs — V2.

Changes from V1:
  1. Voxel representative selection: gradient-based (highest y-deviation),
     centroid-nearest, or first-hit (configurable).
  2. orig_index: correctly handles torch Subset (uses subset.indices).
"""
import os, math, argparse, inspect

import torch
from torch_geometric.datasets import AirfRANS
from torch_geometric.data import Data
from torch_geometric.nn.pool import voxel_grid
from torch.utils.data import Subset
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
        return (wall < 1e-6) | (nxy.abs().sum(dim=1) > 1e-8)
    elif x is not None and x.size(1) >= 3:
        wall = x[:, 2]
        return (wall < 1e-6)
    else:
        n = d.num_nodes or (x.size(0) if x is not None else 0)
        dev = x.device if x is not None else 'cpu'
        return torch.zeros(n, dtype=torch.bool, device=dev)


def _pick_representative(indices: torch.Tensor, pos: torch.Tensor,
                         y: torch.Tensor | None, policy: str) -> int:
    """Pick one representative node index from a voxel's node indices.

    Args:
        indices: 1D tensor of node indices belonging to this voxel.
        pos: [N, D] full position tensor.
        y: [N, C] full target tensor or None.
        policy: 'gradient', 'centroid', or 'first'.

    Returns:
        Global node index (int) of the chosen representative.
    """
    if indices.numel() == 1 or policy == "first":
        return int(indices[0])

    if policy == "gradient" and y is not None:
        # Proxy for flow gradient importance: deviation from voxel mean.
        # The node whose y differs most from the voxel-mean y captures
        # the steepest local gradient direction.
        y_voxel = y[indices]  # [K, C]
        mean_y = y_voxel.mean(dim=0, keepdim=True)  # [1, C]
        deviation = (y_voxel - mean_y).norm(dim=1)  # [K]
        best_local = int(deviation.argmax())
        return int(indices[best_local])

    # policy == "centroid" or gradient fallback when y is None
    p_voxel = pos[indices]  # [K, D]
    centroid = p_voxel.mean(dim=0, keepdim=True)  # [1, D]
    dists = (p_voxel - centroid).norm(dim=1)
    closest = int(dists.argmin())
    return int(indices[closest])


def voxel_preserve_surface(d: Data, voxel_size: float, policy: str = "gradient") -> Data:
    """Downsample volume nodes via voxel grid while preserving all surface nodes.

    V2 changes:
      - Representative selection is configurable (gradient / centroid / first).
    """
    src = d.x if d.x is not None else d.pos
    N = d.num_nodes if d.num_nodes is not None else (src.size(0) if src is not None else 0)
    surf = get_surface_mask(d)
    si = torch.nonzero(surf, as_tuple=False).view(-1)
    vi = torch.nonzero(~surf, as_tuple=False).view(-1)
    if vi.numel() == 0:
        return d
    p_all = d.pos if (hasattr(d, 'pos') and d.pos is not None) else src
    if p_all is None:
        return d
    p2 = p_all[:, :2]
    p2_v = p2[vi].cpu()
    y_data = d.y if hasattr(d, 'y') and d.y is not None else None

    cl = voxel_grid(p2_v, size=float(voxel_size), batch=torch.zeros(p2_v.size(0), dtype=torch.long))

    # Group volume node indices by voxel cluster
    idx = torch.arange(p2_v.size(0))
    sorted_order = torch.argsort(cl)
    cl_sorted = cl[sorted_order]
    idx_sorted = idx[sorted_order]

    # Find cluster boundaries
    boundaries = torch.where(cl_sorted[1:] != cl_sorted[:-1])[0] + 1
    boundaries = torch.cat([torch.tensor([0]), boundaries, torch.tensor([cl_sorted.size(0)])])

    keep_v_list = []
    for b in range(boundaries.size(0) - 1):
        start, end = int(boundaries[b]), int(boundaries[b + 1])
        local_indices = idx_sorted[start:end]
        global_indices = vi[local_indices]
        rep = _pick_representative(global_indices, p_all, y_data, policy)
        keep_v_list.append(rep)

    keep_v = torch.tensor(keep_v_list, dtype=torch.long, device=vi.device)
    keep = torch.unique(torch.cat([si, keep_v], dim=0), sorted=True)
    new = {}
    for k, v in d.to_dict().items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == N:
            new[k] = v[keep]
        else:
            new[k] = v
    newd = Data(**new)
    newd.subsample_keep_idx = keep
    newd.subsample_voxel_size = float(voxel_size)
    return newd


def adapt_voxel(d: Data, tmin: int, tmax: int, frac: float, iters: int,
                policy: str = "gradient") -> Data:
    src = d.pos if hasattr(d, 'pos') and d.pos is not None else d.x
    if src is None:
        return d
    pos2 = src[:, :2]
    chord = estimate_chord_length(pos2)
    f = max(1e-5, float(frac))
    best = None
    for _ in range(max(1, int(iters))):
        v = chord * f
        sub = voxel_preserve_surface(d, v, policy=policy)
        if sub.x is not None:
            n = int(sub.x.size(0))
        elif hasattr(sub, 'pos') and sub.pos is not None:
            n = int(sub.pos.size(0))
        else:
            n = int(sub.num_nodes or 0)
        if tmin <= n <= tmax:
            return sub
        mid = 0.5 * (tmin + tmax)
        if n > 0:
            f = min(1.0, max(1e-5, f * math.sqrt(n / max(1.0, mid))))

        def _node_count(data: Data) -> int:
            if data.x is not None:
                return int(data.x.size(0))
            if hasattr(data, 'pos') and data.pos is not None:
                return int(data.pos.size(0))
            return int(data.num_nodes or 0)

        if best is None or abs(n - mid) < abs(_node_count(best) - mid):
            best = sub
    return best if best is not None else d


def _get_real_index(ds, loop_idx: int) -> int:
    """Return the true dataset index, handling Subset wrappers.

    V2 fix: when ds is a torch Subset, loop_idx is the position within
    the subset, not the original dataset index.
    """
    if isinstance(ds, Subset):
        return int(ds.indices[loop_idx])
    return loop_idx


def _parse_args() -> 'DownsampleConfigV2':
    from scripts.preprocess_config import DownsampleConfigV2
    defaults = DownsampleConfigV2()
    ap = argparse.ArgumentParser(description='Downsample AirfRANS graphs — V2.')
    ap.add_argument('--root', type=str, default=defaults.root)
    ap.add_argument('--task', type=str, default=defaults.task, choices=['scarce', 'full'])
    ap.add_argument('--out-dir', type=str, default=defaults.out_dir)
    ap.add_argument('--limit-train', type=int, default=defaults.limit_train)
    ap.add_argument('--limit-test', type=int, default=defaults.limit_test)
    ap.add_argument('--target-min-nodes', type=int, default=defaults.target_min_nodes)
    ap.add_argument('--target-max-nodes', type=int, default=defaults.target_max_nodes)
    ap.add_argument('--voxel-frac', type=float, default=defaults.voxel_frac)
    ap.add_argument('--voxel-iters', type=int, default=defaults.voxel_iters)
    ap.add_argument('--voxel-rep', type=str, default=defaults.voxel_rep,
                    choices=['gradient', 'centroid', 'first'],
                    help='Voxel representative selection policy')
    args = ap.parse_args()
    return DownsampleConfigV2(
        root=args.root, task=args.task, out_dir=args.out_dir,
        limit_train=args.limit_train, limit_test=args.limit_test,
        target_min_nodes=args.target_min_nodes, target_max_nodes=args.target_max_nodes,
        voxel_frac=args.voxel_frac, voxel_iters=args.voxel_iters,
        voxel_rep=args.voxel_rep,
    )


def run(cfg: 'DownsampleConfigV2'):
    init_params = inspect.signature(AirfRANS.__init__).parameters
    task_kwargs = {'task': cfg.task} if 'task' in init_params else {}

    ds_train = AirfRANS(root=cfg.root, train=True, **task_kwargs)
    ds_test = AirfRANS(root=cfg.root, train=False, **task_kwargs)

    if cfg.limit_train is not None:
        ds_train = Subset(ds_train, list(range(min(cfg.limit_train, len(ds_train)))))
    if cfg.limit_test is not None:
        ds_test = Subset(ds_test, list(range(min(cfg.limit_test, len(ds_test)))))

    out_root = os.path.join(cfg.out_dir, cfg.task)
    out_train = os.path.join(out_root, 'train')
    out_test = os.path.join(out_root, 'test')
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    def _run_split(ds, out_dir: str) -> int:
        saved = 0
        for i in tqdm(range(len(ds)), desc=f'Downsample -> {out_dir}'):
            d = ds[i]
            d2 = Data(**{k: v for k, v in d})
            d2 = adapt_voxel(d2, cfg.target_min_nodes, cfg.target_max_nodes,
                             cfg.voxel_frac, cfg.voxel_iters, policy=cfg.voxel_rep)
            keep = {}
            for k, v in d2.to_dict().items():
                if k in ('x', 'y', 'pos', 'surf'):
                    keep[k] = v
            # V2 fix: use real dataset index, not loop index
            real_idx = _get_real_index(ds, i)
            keep['orig_index'] = torch.tensor(int(real_idx), dtype=torch.long)
            torch.save(Data(**keep), os.path.join(out_dir, f'graph_{real_idx:06d}.pt'))
            saved += 1
        return saved

    n_tr = _run_split(ds_train, out_train)
    n_te = _run_split(ds_test, out_test)
    print(f'[V2] Saved downsampled: train={n_tr} test={n_te} under {out_root}')
    print(f'[V2] Voxel representative policy: {cfg.voxel_rep}')


def main():
    run(_parse_args())


if __name__ == '__main__':
    main()
