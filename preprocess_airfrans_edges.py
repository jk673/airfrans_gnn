# -*- coding: utf-8 -*-
"""
AirfRANS Edge Preprocessing Utility (Memory-Optimized)
=====================================================
Build & cache radius-graph edges + edge features for the AirfRANS dataset.

Key Features:
  - Presets: scarce, full, coarse, aoa_range
  - Filtering: substring, AoA range
  - Edge construction:
        Global: radius_graph(r=global_radius, max_num_neighbors)
        Surface: radius_graph(r=surface_radius) + optional ring edges
        Edge features: [dist, dir_x, dir_y, cos(n_i,n_j), is_surface_pair]
  - Caching: {split}_{idx}_{posHash}.pt
  - Manifest JSON with stats
  - Parallel (chunked) or fully sequential modes
  - Adaptive memory throttling (optional) using psutil if installed
  - Periodic garbage collection

New Memory-Safety Options:
  --sequential          : Force pure sequential processing (lowest RAM)
  --chunk-size N        : Limit number of graphs processed concurrently per batch
  --mem-highwater P     : If RAM usage % exceeds P, automatically halves workers (psutil needed)
  --gc-interval N       : Run gc.collect() every N processed graphs
  --max-active-futures M: Hard cap on futures in flight (additional safety)
  --workers 0           : Auto => cpu_count(); if --sequential given, workers ignored

Examples:
  Sequential (safe):
    python preprocess_airfrans_edges.py --preset scarce --sequential
  Chunked parallel:
    python preprocess_airfrans_edges.py --preset scarce --workers 6 --chunk-size 24
  Adaptive:
    python preprocess_airfrans_edges.py --preset full --workers 8 --chunk-size 32 --mem-highwater 80
"""
from __future__ import annotations
import os, sys, time, json, argparse, hashlib, re, traceback, gc, math
from dataclasses import asdict, dataclass
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import torch
from torch_geometric.datasets import AirfRANS
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None

SURF_KEY_NAME = 'surf'
EDGE_FEAT_DIM = 5
AOA_REGEX = re.compile(r"(-?\d+\.\d+)")

@dataclass
class Params:
    root: str
    preset: str
    task: str
    include_test: bool
    global_radius: float
    surface_radius: float
    max_num_neighbors: int
    surface_ring: bool
    output_dir: str
    rebuild: bool
    limit: Optional[int]
    workers: int
    use_processes: bool
    aoa_min: Optional[float]
    aoa_max: Optional[float]
    aoa_index: int
    filter_contains: Optional[str]
    sequential: bool
    chunk_size: int
    mem_highwater: float
    gc_interval: int
    max_active_futures: int

def hash_pos(pos: torch.Tensor) -> str:
    with torch.no_grad():
        return hashlib.sha1(pos.detach().cpu().numpy().tobytes()).hexdigest()[:16]

def build_edges_for_graph(data: Data, p: Params) -> Data:
    assert hasattr(data, 'pos') and data.pos is not None, 'Data.pos missing'
    pos2 = data.pos[:, :2].contiguous()
    surface_mask = None
    if hasattr(data, SURF_KEY_NAME):
        surface_mask = getattr(data, SURF_KEY_NAME).bool()
        if surface_mask.ndim > 1:
            surface_mask = surface_mask.view(-1)

    edge_global = radius_graph(pos2, r=p.global_radius, loop=False, max_num_neighbors=p.max_num_neighbors)

    if surface_mask is not None and surface_mask.any():
        surf_idx = torch.nonzero(surface_mask, as_tuple=False).view(-1)
        pos_surf = pos2[surf_idx]
        edge_surf_local = radius_graph(pos_surf, r=p.surface_radius, loop=False, max_num_neighbors=p.max_num_neighbors)
        edge_surf = torch.stack([surf_idx[edge_surf_local[0]], surf_idx[edge_surf_local[1]]], dim=0)
        if p.surface_ring and pos_surf.size(0) > 4:
            center = pos_surf.mean(dim=0)
            rel = pos_surf - center
            ang = torch.atan2(rel[:,1], rel[:,0])
            order = torch.argsort(ang)
            cyc = torch.stack([surf_idx[order], surf_idx[order.roll(-1)]], dim=0)
            edge_surf = torch.cat([edge_surf, cyc], dim=1)
    else:
        edge_surf = torch.empty((2,0), dtype=torch.long)

    edge_all = torch.cat([edge_global, edge_surf], dim=1)
    edge_all = torch.cat([edge_all, edge_all.flip(0)], dim=1)
    edge_all = torch.unique(edge_all, dim=1)

    row, col = edge_all
    dvec = pos2[col] - pos2[row]
    dist = dvec.norm(dim=1, keepdim=True).clamp_min(1e-12)
    dir_xy = dvec / dist

    if hasattr(data, 'x') and data.x is not None and data.x.shape[1] >= 5:
        normals = data.x[:, 3:5]
        n_row = normals[row]
        n_col = normals[col]
        nr_norm = n_row.norm(dim=1, keepdim=True).clamp_min(1e-9)
        nc_norm = n_col.norm(dim=1, keepdim=True).clamp_min(1e-9)
        cos_n = (n_row * n_col).sum(dim=1, keepdim=True) / (nr_norm * nc_norm)
    else:
        cos_n = torch.zeros(dist.size(0), 1)

    if surface_mask is not None:
        surf_pair = (surface_mask[row] & surface_mask[col]).float().unsqueeze(1)
    else:
        surf_pair = torch.zeros(dist.size(0), 1)

    edge_attr = torch.cat([dist, dir_xy, cos_n, surf_pair], dim=1)
    data.edge_index = edge_all
    data.edge_attr = edge_attr
    data.edge_meta = {
        'global_radius': p.global_radius,
        'surface_radius': p.surface_radius,
        'edge_count': edge_all.size(1),
        'surf_edge_count': int(surf_pair.sum().item())
    }
    return data

def parse_aoa_from_name(name: str, index: int) -> Optional[float]:
    parts = name.split('_')
    nums = []
    for tok in parts:
        try:
            nums.append(float(tok))
        except ValueError:
            continue
    if not nums:
        matches = AOA_REGEX.findall(name)
        nums = [float(m) for m in matches]
    if len(nums) > index:
        return nums[index]
    return None

def process_one(idx: int, split: str, output_dir: str, rebuild: bool, params: Params,
                dataset_train: AirfRANS, dataset_test: Optional[AirfRANS]) -> Dict[str, Any]:
    dataset = dataset_train if split == 'train' else dataset_test
    if dataset is None:
        return {'status': 'skip-nosplit'}
    d: Data = dataset[idx]
    if not hasattr(d, 'pos') or d.pos is None:
        return {'status': 'skip-nopos', 'idx': idx}
    h = hash_pos(d.pos)
    cache_file = os.path.join(output_dir, f"{split}_{idx}_{h}.pt")
    if (not rebuild) and os.path.isfile(cache_file):
        try:
            cached = torch.load(cache_file, map_location='cpu')
            edge_count = cached.get('edge_index', torch.empty(2,0)).size(1)
            surf_edge_count = cached.get('edge_meta', {}).get('surf_edge_count', 0)
            return {'status': 'reused', 'idx': idx, 'edge_count': edge_count, 'surf_edge_count': surf_edge_count}
        except Exception:
            pass
    build_edges_for_graph(d, params)
    torch.save({'edge_index': d.edge_index,
                'edge_attr': d.edge_attr,
                'edge_meta': d.edge_meta}, cache_file)
    # Free references quickly
    d.edge_index = None
    d.edge_attr = None
    d.edge_meta = None
    return {'status': 'built', 'idx': idx}

def mem_usage_percent() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return None

def process_indices_parallel(indices: List[int], split: str, params: Params,
                             dataset_train: AirfRANS, dataset_test: Optional[AirfRANS]) -> Dict[str, int]:
    executor_cls = ProcessPoolExecutor if params.use_processes else ThreadPoolExecutor
    built = reused = skipped = 0
    edge_counts = []
    surf_edge_counts = []
    processed = 0

    total = len(indices)
    chunk_size = max(1, params.chunk_size)
    current_workers = params.workers

    with tqdm(total=total, desc=f'Processing {split}') as pbar:
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            batch = indices[start:end]

            # Adaptive memory throttle
            mperc = mem_usage_percent()
            if mperc is not None and mperc > params.mem_highwater and current_workers > 1:
                current_workers = max(1, current_workers // 2)
                print(f"[Throttle] Memory {mperc:.1f}% > {params.mem_highwater}% -> reducing workers to {current_workers}")

            # Launch limited futures
            with executor_cls(max_workers=min(current_workers, len(batch))) as ex:
                futures = []
                for idx in batch:
                    if params.max_active_futures > 0 and len(futures) >= params.max_active_futures:
                        # Wait early if capped
                        for fut in as_completed(futures):
                            r = fut.result()
                            status = r.get('status')
                            if status == 'built':
                                built += 1
                            elif status == 'reused':
                                reused += 1
                            elif status and status.startswith('skip'):
                                skipped += 1
                            processed += 1
                            pbar.update(1)
                        futures.clear()
                    futures.append(ex.submit(process_one, idx, split, params.output_dir,
                                             params.rebuild, params, dataset_train, dataset_test))
                # Collect remaining
                for fut in as_completed(futures):
                    r = fut.result()
                    status = r.get('status')
                    if status == 'built':
                        built += 1
                    elif status == 'reused':
                        reused += 1
                    elif status and status.startswith('skip'):
                        skipped += 1
                    processed += 1
                    pbar.update(1)
                    if processed % params.gc_interval == 0:
                        gc.collect()
            # Extra GC safety per chunk
            gc.collect()

    return {
        'built': built,
        'reused': reused,
        'skipped': skipped,
        'edge_counts': edge_counts,
        'surf_edge_counts': surf_edge_counts
    }

def process_indices_sequential(indices: List[int], split: str, params: Params,
                               dataset_train: AirfRANS, dataset_test: Optional[AirfRANS]) -> Dict[str,int]:
    built = reused = skipped = 0
    with tqdm(total=len(indices), desc=f'Processing {split} (seq)') as pbar:
        for i, idx in enumerate(indices):
            r = process_one(idx, split, params.output_dir, params.rebuild, params, dataset_train, dataset_test)
            status = r.get('status')
            if status == 'built':
                built += 1
            elif status == 'reused':
                reused += 1
            elif status and status.startswith('skip'):
                skipped += 1
            if (i+1) % params.gc_interval == 0:
                gc.collect()
            pbar.update(1)
    gc.collect()
    return {'built': built, 'reused': reused, 'skipped': skipped, 'edge_counts': [], 'surf_edge_counts': []}

def main():
    parser = argparse.ArgumentParser(description='Preprocess AirfRANS edges (memory-optimized).')
    parser.add_argument('--root', type=str, default='Dataset')
    parser.add_argument('--preset', type=str, default='scarce', choices=['scarce','full','coarse','aoa_range'])
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--include-test', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--filter-contains', type=str, default=None)
    parser.add_argument('--global-radius', type=float, default=0.02)
    parser.add_argument('--surface-radius', type=float, default=0.01)
    parser.add_argument('--max-num-neighbors', type=int, default=64)
    parser.add_argument('--surface-ring', action='store_true')
    parser.add_argument('--output-dir', type=str, default='processed_edges')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--processes', action='store_true')
    parser.add_argument('--aoa-min', type=float, default=None)
    parser.add_argument('--aoa-max', type=float, default=None)
    parser.add_argument('--aoa-index', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    # Memory / scheduling options
    parser.add_argument('--sequential', action='store_true', help='Force sequential processing (lowest memory)')
    parser.add_argument('--chunk-size', type=int, default=16, help='Graphs per parallel chunk')
    parser.add_argument('--mem-highwater', type=float, default=85.0, help='Percent RAM usage to trigger throttling')
    parser.add_argument('--gc-interval', type=int, default=20, help='GC every N processed graphs')
    parser.add_argument('--max-active-futures', type=int, default=0, help='Cap futures in flight (0=unlimited within chunk)')
    args = parser.parse_args()

    # Determine task
    task = args.task
    if task is None:
        if args.preset in ('scarce','coarse','aoa_range'):
            task = 'scarce'
        elif args.preset == 'full':
            task = 'full'
        else:
            task = 'scarce'

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Config] preset={args.preset} task={task} root={args.root} output_dir={args.output_dir}")
    print(f"[Mode] sequential={args.sequential} processes={args.processes}")

    # Load datasets
    dataset_train = AirfRANS(root=args.root, task=task, train=True)
    dataset_test = AirfRANS(root=args.root, task=task, train=False) if args.include_test else None

    train_indices = list(range(len(dataset_train)))

    def name_for_idx_train(i: int) -> str:
        d = dataset_train[i]
        if hasattr(d,'name'):
            return str(getattr(d,'name'))
        return f'graph_{i}'

    # Substring filter
    if args.filter_contains:
        fc = args.filter_contains.lower()
        train_indices = [i for i in train_indices if fc in name_for_idx_train(i).lower()]
        print(f"[Filter] substring '{args.filter_contains}' => {len(train_indices)} remain")

    # AoA range filter
    if args.preset == 'aoa_range':
        if args.aoa_min is None or args.aoa_max is None:
            print('[Error] aoa_range requires --aoa-min and --aoa-max'); sys.exit(1)
        kept = []
        miss = 0
        for i in train_indices:
            aoa = parse_aoa_from_name(name_for_idx_train(i), args.aoa_index)
            if aoa is None:
                miss += 1
                continue
            if args.aoa_min <= aoa <= args.aoa_max:
                kept.append(i)
        print(f"[AoA] kept={len(kept)} missed={miss}")
        train_indices = kept

    # Coarse => default limit
    if args.preset == 'coarse' and args.limit is None:
        args.limit = 100

    if args.limit is not None:
        train_indices = train_indices[:args.limit]
        print(f"[Limit] after limit={len(train_indices)}")

    if not train_indices:
        print('[Warn] No train graphs to process.')
        return

    if args.dry_run:
        print(f"[DryRun] would process {len(train_indices)} train graphs")
        return

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)
    if args.sequential:
        workers = 1

    params = Params(
        root=args.root,
        preset=args.preset,
        task=task,
        include_test=args.include_test,
        global_radius=args.global_radius,
        surface_radius=args.surface_radius,
        max_num_neighbors=args.max_num_neighbors,
        surface_ring=args.surface_ring,
        output_dir=args.output_dir,
        rebuild=args.rebuild,
        limit=args.limit,
        workers=workers,
        use_processes=args.processes,
        aoa_min=args.aoa_min,
        aoa_max=args.aoa_max,
        aoa_index=args.aoa_index,
        filter_contains=args.filter_contains,
        sequential=args.sequential,
        chunk_size=args.chunk_size,
        mem_highwater=args.mem_highwater,
        gc_interval=max(1,args.gc_interval),
        max_active_futures=args.max_active_futures
    )

    print(f"[Parallel Config] workers={params.workers} chunk_size={params.chunk_size} mem_highwater={params.mem_highwater}%")

    start = time.time()

    if params.sequential or params.workers == 1:
        stats_train = process_indices_sequential(train_indices, 'train', params, dataset_train, dataset_test)
    else:
        stats_train = process_indices_parallel(train_indices, 'train', params, dataset_train, dataset_test)

    stats_test = {}
    if params.include_test and dataset_test is not None:
        test_indices = list(range(len(dataset_test)))
        if params.sequential or params.workers == 1:
            stats_test = process_indices_sequential(test_indices, 'test', params, dataset_train, dataset_test)
        else:
            stats_test = process_indices_parallel(test_indices, 'test', params, dataset_train, dataset_test)

    elapsed = time.time() - start

    summary = {
        'train': stats_train,
        'test': stats_test,
        'elapsed_sec': elapsed
    }

    manifest = {
        'params': asdict(params),
        'summary': summary,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    manifest_path = os.path.join(params.output_dir, 'edge_preprocess_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n[Summary]")
    print(f"  Train: built={stats_train['built']} reused={stats_train['reused']} skipped={stats_train['skipped']}")
    if stats_test:
        print(f"  Test : built={stats_test['built']} reused={stats_test['reused']} skipped={stats_test['skipped']}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Output dir: {os.path.abspath(params.output_dir)}")
    print(f"  Manifest : {manifest_path}")
    print("Done.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[Interrupted] User aborted.')
    except Exception as e:
        print('[Fatal] Exception:', e)
        traceback.print_exc()
        sys.exit(1)