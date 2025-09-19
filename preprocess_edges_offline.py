#!/usr/bin/env python
"""
Offline preprocessor for AirfRANS graphs: builds edges once and saves per-graph .pt files.
Normalization remains in training/notebook scripts.

Outputs layout:
  <out_dir>/train/graph_<idx>.pt
  <out_dir>/test/graph_<idx>.pt

Usage (Linux):
  python preprocess_edges_offline.py \
    --root /workspace/airfrans \
    --task scarce \
    --out-dir prebuilt_edges/scarce \
    --global-radius 0.02 --surface-radius 0.01 --max-num-neighbors 48 --surface-ring
"""
import os
import argparse
from typing import Optional, Tuple
import multiprocessing as mp

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import AirfRANS
from tqdm import tqdm


def _import_preprocess_module(path: str = 'preprocess_airfrans_edges.py'):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required module not found: {path}")
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location('pre_air', path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import preprocess module from {path}")
    pre_air = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = pre_air  # type: ignore[attr-defined]
    spec.loader.exec_module(pre_air)  # type: ignore[union-attr]
    return pre_air


def _init_worker(module_path: str,
                 params_dict: dict,
                 root: str,
                 task: str,
                 is_train: bool,
                 out_split_dir: str):
    """Initializer: import preprocess module, rebuild Params, and open dataset once per process."""
    global G_PRE, G_PARAMS, G_DS, G_OUT_DIR
    G_OUT_DIR = out_split_dir
    # Import preprocess module
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location('pre_air', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import preprocess module from {module_path}")
    pre_air = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = pre_air  # type: ignore[attr-defined]
    spec.loader.exec_module(pre_air)  # type: ignore[union-attr]
    # Rebuild Params
    class EdgeParams(pre_air.Params):
        pass
    G_PARAMS = EdgeParams(**params_dict)
    G_PRE = pre_air
    # Open dataset for this process
    try:
        G_DS = AirfRANS(root=root, train=is_train, task=task)
    except TypeError:
        G_DS = AirfRANS(root=root, train=is_train)


def _process_one(task: Tuple[int, int]) -> int:
    """Process a single graph index. task = (pos_idx, orig_idx). Returns 1 on success, 0 on failure."""
    pos_idx, orig_idx = task
    try:
        d = G_DS[orig_idx]
        d2 = Data(**{k: v for k, v in d})
        d2 = G_PRE.build_edges_for_graph(d2, G_PARAMS)
        if hasattr(d2, 'edge_index') and d2.edge_index is not None and d2.edge_index.dtype != torch.long:
            d2.edge_index = d2.edge_index.long()
        out_path = os.path.join(G_OUT_DIR, f"graph_{pos_idx:06d}.pt")
        torch.save(d2, out_path)
        return 1
    except Exception as e:
        # Best-effort logging without print storms; encode failure by 0
        return 0


def build_edges_for_dataset(ds, out_split_dir: str, params, root: str, task: str, is_train: bool,
                            workers: int = 0, mp_chunksize: int = 4) -> int:
    os.makedirs(out_split_dir, exist_ok=True)
    # Build list of (pos_idx, orig_idx); pos_idx is 0..len-1 for stable filenames
    try:
        from torch.utils.data import Subset
        if isinstance(ds, Subset):
            orig_indices = list(ds.indices)
        else:
            orig_indices = list(range(len(ds)))
    except Exception:
        orig_indices = list(range(len(ds)))
    tasks = [(i, orig_idx) for i, orig_idx in enumerate(orig_indices)]

    if workers is None or workers <= 1:
        saved = 0
        for pos_idx, orig_idx in tqdm(tasks, desc=f"Processing -> {out_split_dir}"):
            try:
                d = ds[orig_idx] if hasattr(ds, '__getitem__') else ds[pos_idx]
                d2 = Data(**{k: v for k, v in d})
                d2 = params._module.build_edges_for_graph(d2, params)  # type: ignore[attr-defined]
                if hasattr(d2, 'edge_index') and d2.edge_index is not None and d2.edge_index.dtype != torch.long:
                    d2.edge_index = d2.edge_index.long()
                torch.save(d2, os.path.join(out_split_dir, f"graph_{pos_idx:06d}.pt"))
                saved += 1
            except Exception:
                continue
        return saved

    # Parallel path
    module_path = os.path.abspath('preprocess_airfrans_edges.py')
    params_dict = {k: getattr(params, k) for k in params.__dict__ if not k.startswith('_')}
    ctx = mp.get_context('spawn')
    saved = 0
    with ctx.Pool(processes=workers,
                  initializer=_init_worker,
                  initargs=(module_path, params_dict, root, task, is_train, out_split_dir)) as pool:
        with tqdm(total=len(tasks), desc=f"Processing (x{workers}) -> {out_split_dir}") as pbar:
            for ok in pool.imap_unordered(_process_one, tasks, chunksize=max(1, int(mp_chunksize))):
                saved += int(ok)
                pbar.update(1)
    return saved


def main():
    ap = argparse.ArgumentParser(description="Offline edge builder for AirfRANS")
    ap.add_argument('--root', type=str, required=True, help='AirfRANS root (parent containing processed/)')
    ap.add_argument('--task', type=str, default='scarce', choices=['scarce','full'], help='Task split to use when available')
    ap.add_argument('--out-dir', type=str, default='prebuilt_edges', help='Output directory root for saved graphs')
    ap.add_argument('--limit-train', type=int, default=None, help='Optional cap on number of training graphs')
    ap.add_argument('--limit-test', type=int, default=None, help='Optional cap on number of test graphs')
    # Edge parameters
    ap.add_argument('--global-radius', type=float, default=0.02)
    ap.add_argument('--surface-radius', type=float, default=0.01)
    ap.add_argument('--max-num-neighbors', type=int, default=48)
    ap.add_argument('--surface-ring', action='store_true', default=True)
    ap.add_argument('--workers', type=int, default=0, help='Number of parallel worker processes (0 or 1 = serial)')
    ap.add_argument('--mp-chunksize', type=int, default=4, help='Multiprocessing task chunk size (advanced)')
    args = ap.parse_args()

    # Import preprocess module and params class
    pre_air = _import_preprocess_module('preprocess_airfrans_edges.py')
    class EdgeParams(pre_air.Params):
        pass

    # Build params instance (module reference stored for convenience)
    params = EdgeParams(
        root=args.root,
        preset='scarce',
        task=args.task,
        include_test=False,
        global_radius=args.global_radius,
        surface_radius=args.surface_radius,
        max_num_neighbors=args.max_num_neighbors,
        surface_ring=args.surface_ring,
        output_dir='__offline__',
        rebuild=True,
        limit=None,
        workers=0,
        use_processes=False,
        aoa_min=None,
        aoa_max=None,
        aoa_index=2,
        filter_contains=None,
        sequential=True,
        chunk_size=1,
        mem_highwater=100.0,
        gc_interval=1,
        max_active_futures=0,
    )
    params._module = pre_air  # attach for use in helper

    # Load datasets
    try:
        ds_train = AirfRANS(root=args.root, train=True, task=args.task)
        ds_test  = AirfRANS(root=args.root, train=False, task=args.task)
    except TypeError:
        ds_train = AirfRANS(root=args.root, train=True)
        ds_test  = AirfRANS(root=args.root, train=False)

    # Apply limits
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

    n_tr = build_edges_for_dataset(ds_train, out_train, params, root=args.root, task=args.task, is_train=True,
                                   workers=args.workers, mp_chunksize=args.mp_chunksize)
    n_te = build_edges_for_dataset(ds_test, out_test, params, root=args.root, task=args.task, is_train=False,
                                   workers=args.workers, mp_chunksize=args.mp_chunksize)

    print(f"Saved: train={n_tr} | test={n_te} under {out_root}")


if __name__ == '__main__':
    main()
