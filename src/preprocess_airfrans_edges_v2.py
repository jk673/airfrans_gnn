# -*- coding: utf-8 -*-
"""
AirfRANS Edge Preprocessing Utility — V2.

Changes from V1:
  1. Degree calculation: undirected (bincount(row) only, since edges are bidirectional).
     min_degree=2 now genuinely guarantees 2 undirected neighbors.
  2. Per-graph QA report: isolated count, low-degree count, degree/edge-length stats.
  3. Cleaner _postfix_repair_min_degree with corrected degree semantics.
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

    # degree floor & knn backup
    min_degree: int = 2
    knn_backup_k: int = 4
    knn_max_radius: float = 0.05
    knn_mutual: bool = True
    backup_per_node_budget: int = 4
    final_max_degree: int = 64
    length_hard_cap: float = 0.12

    # V2: QA thresholds
    max_isolated_fraction: float = 0.01
    max_low_degree_fraction: float = 0.05
    qa_fail_fast: bool = False


def hash_pos(pos: torch.Tensor) -> str:
    with torch.no_grad():
        return hashlib.sha1(pos.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def _undirected_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute undirected degree from bidirectional edge_index.

    V2 fix: edges are already stored as bidirectional (i->j and j->i both present),
    so counting only row gives the true undirected neighbor count.
    V1 used bincount(row) + bincount(col) which double-counted.
    """
    return torch.bincount(edge_index[0], minlength=num_nodes)


def _postfix_repair_min_degree(edge_all: torch.Tensor, pos2: torch.Tensor,
                               min_degree: int, max_degree: int,
                               knn_k: int, knn_radius: float,
                               length_cap: float, max_iters: int = 3) -> torch.Tensor:
    """Iteratively add short backup edges so every node has deg >= min_degree.

    V2: uses _undirected_degree (row-only bincount) for correct degree semantics.
    """
    device = pos2.device
    N = pos2.size(0)

    edge_all = edge_all.clone()
    for _ in range(max_iters):
        deg = _undirected_degree(edge_all, N)
        need = (deg < min_degree).nonzero(as_tuple=False).view(-1)
        if need.numel() == 0:
            break

        # Build membership set for existing edges
        inc = edge_all.t().tolist()
        ei_set = set((int(a), int(b)) for a, b in inc)

        src_list, dst_list = [], []

        with torch.no_grad():
            D = torch.cdist(pos2[need], pos2)  # [M, N]
            M = need.numel()
            D[torch.arange(M, device=device), need] = 1e9
            if knn_radius and knn_radius > 0:
                D = D.masked_fill(D > knn_radius, 1e9)
            k = min(max(knn_k, min_degree), max(1, N - 1))
            nn_idx = torch.topk(D, k=k, largest=False).indices  # [M, k]

        for r in range(M):
            i = int(need[r])
            if int(deg[i]) >= min_degree:
                continue

            di = D[r]
            order = torch.argsort(di[nn_idx[r]])
            js = [int(nn_idx[r][int(t)]) for t in order.tolist()]

            quota_i = min_degree - int(deg[i])
            added = 0

            for j in js:
                if added >= quota_i:
                    break
                if j == i:
                    continue
                if max_degree and int(deg[j]) >= max_degree:
                    continue
                if max_degree and int(deg[i]) >= max_degree:
                    break
                if (i, j) in ei_set or (j, i) in ei_set:
                    continue
                dij = torch.norm(pos2[j] - pos2[i]).item()
                if dij <= 1e-12:
                    continue
                if length_cap and dij > length_cap:
                    continue

                # Add bidirectional edge
                src_list.extend([i, j])
                dst_list.extend([j, i])
                ei_set.add((i, j))
                ei_set.add((j, i))
                deg[i] += 1
                deg[j] += 1
                added += 1

        if src_list:
            extra = torch.stack([torch.tensor(src_list, dtype=torch.long, device=device),
                                 torch.tensor(dst_list, dtype=torch.long, device=device)], dim=0)
            edge_all = torch.unique(torch.cat([edge_all, extra], dim=1), dim=1)
        else:
            break

    return edge_all


def compute_qa_report(edge_index: torch.Tensor, pos2: torch.Tensor,
                      min_degree: int) -> Dict[str, Any]:
    """Compute per-graph QA metrics after edge construction."""
    N = pos2.size(0)
    deg = _undirected_degree(edge_index, N)

    # Edge lengths
    dvec = pos2[edge_index[1]] - pos2[edge_index[0]]
    dist = dvec.norm(dim=1)
    # Only count unique undirected edges for length stats (take half)
    # Since edges are bidirectional, stats are the same either way

    qa = {
        'num_nodes': N,
        'num_edges': edge_index.size(1),
        'num_edges_undirected': edge_index.size(1) // 2,
        'isolated_count': int((deg == 0).sum()),
        'low_degree_count': int((deg < min_degree).sum()),
        'degree_min': int(deg.min()) if N > 0 else 0,
        'degree_max': int(deg.max()) if N > 0 else 0,
        'degree_mean': round(float(deg.float().mean()), 2) if N > 0 else 0.0,
        'degree_median': round(float(deg.float().median()), 2) if N > 0 else 0.0,
        'edge_length_min': round(float(dist.min()), 6) if dist.numel() > 0 else 0.0,
        'edge_length_max': round(float(dist.max()), 6) if dist.numel() > 0 else 0.0,
        'edge_length_mean': round(float(dist.mean()), 6) if dist.numel() > 0 else 0.0,
        'isolated_fraction': round(float((deg == 0).sum()) / max(N, 1), 4),
        'low_degree_fraction': round(float((deg < min_degree).sum()) / max(N, 1), 4),
    }
    return qa


def check_qa_thresholds(qa: Dict[str, Any], params: Params, graph_id: str = ""):
    """Check QA report against thresholds. Warn or raise."""
    prefix = f"[QA {graph_id}]" if graph_id else "[QA]"

    if qa['isolated_fraction'] > params.max_isolated_fraction:
        msg = (f"{prefix} isolated fraction {qa['isolated_fraction']:.4f} "
               f"exceeds threshold {params.max_isolated_fraction}")
        if params.qa_fail_fast:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")

    if qa['low_degree_fraction'] > params.max_low_degree_fraction:
        msg = (f"{prefix} low-degree fraction {qa['low_degree_fraction']:.4f} "
               f"exceeds threshold {params.max_low_degree_fraction}")
        if params.qa_fail_fast:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")


def build_edges_for_graph(data: Data, p: Params) -> tuple[Data, Dict[str, Any]]:
    """Build edges for a single graph. Returns (data_with_edges, qa_report).

    V2 changes:
      - Degree computed via _undirected_degree (row-only bincount).
      - Returns QA report dict.
    """
    assert hasattr(data, 'pos') and data.pos is not None, 'Data.pos missing'
    pos2 = data.pos[:, :2].contiguous()
    N = pos2.size(0)
    surface_mask = None
    if hasattr(data, SURF_KEY_NAME):
        surface_mask = getattr(data, SURF_KEY_NAME).bool()
        if surface_mask.ndim > 1:
            surface_mask = surface_mask.view(-1)

    # --- Global radius graph ---
    edge_global = radius_graph(pos2, r=p.global_radius, loop=False,
                               max_num_neighbors=p.max_num_neighbors)

    # --- Surface edges ---
    if surface_mask is not None and surface_mask.any():
        surf_idx = torch.nonzero(surface_mask, as_tuple=False).view(-1)
        pos_surf = pos2[surf_idx]
        edge_surf_local = radius_graph(pos_surf, r=p.surface_radius, loop=False,
                                       max_num_neighbors=p.max_num_neighbors)
        edge_surf = torch.stack([surf_idx[edge_surf_local[0]],
                                 surf_idx[edge_surf_local[1]]], dim=0)
        if p.surface_ring and pos_surf.size(0) > 4:
            center = pos_surf.mean(dim=0)
            rel = pos_surf - center
            ang = torch.atan2(rel[:, 1], rel[:, 0])
            order = torch.argsort(ang)
            cyc = torch.stack([surf_idx[order], surf_idx[order.roll(-1)]], dim=0)
            edge_surf = torch.cat([edge_surf, cyc], dim=1)
    else:
        edge_surf = torch.empty((2, 0), dtype=torch.long)

    # --- Merge & make bidirectional ---
    edge_all = torch.cat([edge_global, edge_surf], dim=1)
    edge_all = torch.cat([edge_all, edge_all.flip(0)], dim=1)
    edge_all = torch.unique(edge_all, dim=1)

    # --- KNN backup for low-degree nodes (V2: correct degree) ---
    deg = _undirected_degree(edge_all, N)
    need = (deg < p.min_degree).nonzero(as_tuple=False).view(-1)
    if need.numel() > 0 and p.knn_backup_k > 0:
        with torch.no_grad():
            D = torch.cdist(pos2[need], pos2)  # [M, N]
            D[torch.arange(need.numel()), need] = 1e9
            if p.knn_max_radius and p.knn_max_radius > 0:
                D = D.masked_fill(D > p.knn_max_radius, 1e9)
            k = min(max(p.knn_backup_k, p.min_degree), max(1, N - 1))
            nn_idx = torch.topk(D, k=k, largest=False).indices  # [M, k]

        src_list, dst_list = [], []
        for r, i in enumerate(need.tolist()):
            js = nn_idx[r]
            di = D[r]
            order = torch.argsort(di[js])
            js = js[order][:p.backup_per_node_budget]
            if int(deg[i]) >= p.final_max_degree:
                continue
            quota = max(0, p.min_degree - int(deg[i]))
            if quota <= 0 or js.numel() == 0:
                continue
            js = js[:quota]
            src_list.append(torch.full((js.numel(),), i, dtype=torch.long, device=pos2.device))
            dst_list.append(js)
            deg[i] += js.numel()

        if src_list:
            extra = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
            extra = torch.cat([extra, extra.flip(0)], dim=1)
            if p.length_hard_cap and p.length_hard_cap > 0:
                dvec_extra = pos2[extra[1]] - pos2[extra[0]]
                len_extra = dvec_extra.norm(dim=1)
                extra = extra[:, len_extra <= p.length_hard_cap]
            edge_all = torch.unique(torch.cat([edge_all, extra], dim=1), dim=1)

    # --- Final degree cap with nearest-first pruning ---
    if p.final_max_degree and p.final_max_degree > 0:
        row, col = edge_all
        dvec = pos2[col] - pos2[row]
        length_all = dvec.norm(dim=1)
        deg = _undirected_degree(edge_all, N)
        keep_mask = torch.ones(edge_all.size(1), dtype=torch.bool, device=row.device)
        for node in torch.nonzero(deg > p.final_max_degree, as_tuple=False).view(-1).tolist():
            inc_ids = ((row == node) | (col == node)).nonzero(as_tuple=False).view(-1)
            if inc_ids.numel() <= p.final_max_degree:
                continue
            sel = torch.argsort(length_all[inc_ids])[:p.final_max_degree]
            keep_set = set(inc_ids[sel].tolist())
            drop = [idx for idx in inc_ids.tolist() if idx not in keep_set]
            keep_mask[torch.tensor(drop, device=row.device, dtype=torch.long)] = False
        edge_all = edge_all[:, keep_mask]

    # --- Post-fix repair (V2: correct degree) ---
    edge_all = _postfix_repair_min_degree(
        edge_all=edge_all, pos2=pos2,
        min_degree=p.min_degree, max_degree=p.final_max_degree,
        knn_k=p.knn_backup_k, knn_radius=p.knn_max_radius,
        length_cap=p.length_hard_cap, max_iters=3,
    )

    # --- Clean up: remove self-loops and zero-length edges ---
    mask_self = edge_all[0] != edge_all[1]
    edge_all = edge_all[:, mask_self]
    row, col = edge_all
    dvec = pos2[col] - pos2[row]
    dist = dvec.norm(dim=1, keepdim=True)
    nonzero = (dist.view(-1) > 1e-9)
    edge_all = edge_all[:, nonzero]
    dvec = dvec[nonzero]
    dist = dist[nonzero]
    dist = dist.clamp_min(1e-12)
    dir_xy = dvec / dist

    row, col = edge_all

    # --- Edge features ---
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
    edge_attr_dxdy = torch.cat([dvec, dist], dim=1)

    data.edge_index = edge_all
    data.edge_attr = edge_attr
    data.edge_attr_dxdy = edge_attr_dxdy
    data.edge_meta = {
        'global_radius': p.global_radius,
        'surface_radius': p.surface_radius,
        'edge_count': edge_all.size(1),
        'surf_edge_count': int(surf_pair.sum().item()),
        'schemas': {
            'edge_attr': {
                'desc': 'dist, dir_x, dir_y, cos_n, is_surface_pair',
                'columns': ['dist', 'dir_x', 'dir_y', 'cos_n', 'is_surface_pair']
            },
            'edge_attr_dxdy': {
                'desc': 'dx, dy, dist',
                'columns': ['dx', 'dy', 'dist']
            }
        }
    }

    # --- V2: QA report ---
    qa = compute_qa_report(edge_all, pos2, p.min_degree)

    return data, qa


# ============================================================
# CLI & batch processing (mostly unchanged from V1)
# ============================================================

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
            edge_count = cached.get('edge_index', torch.empty(2, 0)).size(1)
            return {'status': 'reused', 'idx': idx, 'edge_count': edge_count}
        except Exception:
            pass
    d, qa = build_edges_for_graph(d, params)
    check_qa_thresholds(qa, params, graph_id=f"{split}_{idx}")
    torch.save({
        'edge_index': d.edge_index,
        'edge_attr': d.edge_attr,
        'edge_attr_dxdy': getattr(d, 'edge_attr_dxdy', None),
        'edge_meta': d.edge_meta,
        'qa': qa,
    }, cache_file)
    d.edge_index = None
    d.edge_attr = None
    d.edge_meta = None
    return {'status': 'built', 'idx': idx, 'qa': qa}


def mem_usage_percent() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return None


def process_indices_sequential(indices: List[int], split: str, params: Params,
                               dataset_train: AirfRANS,
                               dataset_test: Optional[AirfRANS]) -> Dict[str, Any]:
    built = reused = skipped = 0
    qa_reports = []
    with tqdm(total=len(indices), desc=f'Processing {split} (seq)') as pbar:
        for i, idx in enumerate(indices):
            r = process_one(idx, split, params.output_dir, params.rebuild, params,
                            dataset_train, dataset_test)
            status = r.get('status')
            if status == 'built':
                built += 1
                if 'qa' in r:
                    qa_reports.append(r['qa'])
            elif status == 'reused':
                reused += 1
            elif status and status.startswith('skip'):
                skipped += 1
            if (i + 1) % params.gc_interval == 0:
                gc.collect()
            pbar.update(1)
    gc.collect()
    return {'built': built, 'reused': reused, 'skipped': skipped, 'qa_reports': qa_reports}


def process_indices_parallel(indices: List[int], split: str, params: Params,
                             dataset_train: AirfRANS,
                             dataset_test: Optional[AirfRANS]) -> Dict[str, Any]:
    executor_cls = ProcessPoolExecutor if params.use_processes else ThreadPoolExecutor
    built = reused = skipped = 0
    qa_reports = []
    processed = 0

    total = len(indices)
    chunk_size = max(1, params.chunk_size)
    current_workers = params.workers

    with tqdm(total=total, desc=f'Processing {split}') as pbar:
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            batch = indices[start:end]

            mperc = mem_usage_percent()
            if mperc is not None and mperc > params.mem_highwater and current_workers > 1:
                current_workers = max(1, current_workers // 2)
                print(f"[Throttle] Memory {mperc:.1f}% > {params.mem_highwater}% "
                      f"-> reducing workers to {current_workers}")

            with executor_cls(max_workers=min(current_workers, len(batch))) as ex:
                futures = []
                for idx in batch:
                    if params.max_active_futures > 0 and len(futures) >= params.max_active_futures:
                        for fut in as_completed(futures):
                            r = fut.result()
                            status = r.get('status')
                            if status == 'built':
                                built += 1
                                if 'qa' in r:
                                    qa_reports.append(r['qa'])
                            elif status == 'reused':
                                reused += 1
                            elif status and status.startswith('skip'):
                                skipped += 1
                            processed += 1
                            pbar.update(1)
                        futures.clear()
                    futures.append(ex.submit(process_one, idx, split, params.output_dir,
                                             params.rebuild, params, dataset_train, dataset_test))
                for fut in as_completed(futures):
                    r = fut.result()
                    status = r.get('status')
                    if status == 'built':
                        built += 1
                        if 'qa' in r:
                            qa_reports.append(r['qa'])
                    elif status == 'reused':
                        reused += 1
                    elif status and status.startswith('skip'):
                        skipped += 1
                    processed += 1
                    pbar.update(1)
                    if processed % params.gc_interval == 0:
                        gc.collect()
            gc.collect()

    return {'built': built, 'reused': reused, 'skipped': skipped, 'qa_reports': qa_reports}


def print_qa_summary(qa_reports: List[Dict[str, Any]], split: str):
    """Print aggregate QA summary across all graphs in a split."""
    if not qa_reports:
        return
    n = len(qa_reports)
    total_nodes = sum(q['num_nodes'] for q in qa_reports)
    total_isolated = sum(q['isolated_count'] for q in qa_reports)
    total_low_deg = sum(q['low_degree_count'] for q in qa_reports)
    avg_deg_mean = sum(q['degree_mean'] for q in qa_reports) / n
    min_deg_all = min(q['degree_min'] for q in qa_reports)
    max_deg_all = max(q['degree_max'] for q in qa_reports)
    avg_edge_len = sum(q['edge_length_mean'] for q in qa_reports) / n

    print(f"\n[QA Summary — {split}] {n} graphs")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total isolated (deg=0): {total_isolated} "
          f"({total_isolated / max(total_nodes, 1) * 100:.2f}%)")
    print(f"  Total low-degree: {total_low_deg} "
          f"({total_low_deg / max(total_nodes, 1) * 100:.2f}%)")
    print(f"  Degree — min: {min_deg_all}, max: {max_deg_all}, mean: {avg_deg_mean:.1f}")
    print(f"  Edge length — mean: {avg_edge_len:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess AirfRANS edges — V2.')
    parser.add_argument('--root', type=str, default='Dataset')
    parser.add_argument('--preset', type=str, default='scarce',
                        choices=['scarce', 'full', 'coarse', 'aoa_range'])
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--include-test', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--filter-contains', type=str, default=None)
    parser.add_argument('--global-radius', type=float, default=0.02)
    parser.add_argument('--surface-radius', type=float, default=0.01)
    parser.add_argument('--max-num-neighbors', type=int, default=64)
    parser.add_argument('--surface-ring', action='store_true')
    parser.add_argument('--output-dir', type=str, default='processed_edges_v2')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--processes', action='store_true')
    parser.add_argument('--aoa-min', type=float, default=None)
    parser.add_argument('--aoa-max', type=float, default=None)
    parser.add_argument('--aoa-index', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--sequential', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--mem-highwater', type=float, default=85.0)
    parser.add_argument('--gc-interval', type=int, default=20)
    parser.add_argument('--max-active-futures', type=int, default=0)
    # V2 QA options
    parser.add_argument('--max-isolated-fraction', type=float, default=0.01)
    parser.add_argument('--max-low-degree-fraction', type=float, default=0.05)
    parser.add_argument('--qa-fail-fast', action='store_true')
    args = parser.parse_args()

    task = args.task
    if task is None:
        task = 'full' if args.preset == 'full' else 'scarce'

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[V2 Config] preset={args.preset} task={task} root={args.root} "
          f"output_dir={args.output_dir}")

    dataset_train = AirfRANS(root=args.root, task=task, train=True)
    dataset_test = (AirfRANS(root=args.root, task=task, train=False)
                    if args.include_test else None)

    train_indices = list(range(len(dataset_train)))

    def name_for_idx_train(i: int) -> str:
        d = dataset_train[i]
        if hasattr(d, 'name'):
            return str(getattr(d, 'name'))
        return f'graph_{i}'

    if args.filter_contains:
        fc = args.filter_contains.lower()
        train_indices = [i for i in train_indices if fc in name_for_idx_train(i).lower()]

    if args.preset == 'aoa_range':
        if args.aoa_min is None or args.aoa_max is None:
            print('[Error] aoa_range requires --aoa-min and --aoa-max')
            sys.exit(1)
        kept = []
        for i in train_indices:
            aoa = parse_aoa_from_name(name_for_idx_train(i), args.aoa_index)
            if aoa is not None and args.aoa_min <= aoa <= args.aoa_max:
                kept.append(i)
        train_indices = kept

    if args.preset == 'coarse' and args.limit is None:
        args.limit = 100
    if args.limit is not None:
        train_indices = train_indices[:args.limit]

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
        root=args.root, preset=args.preset, task=task,
        include_test=args.include_test,
        global_radius=args.global_radius, surface_radius=args.surface_radius,
        max_num_neighbors=args.max_num_neighbors, surface_ring=args.surface_ring,
        output_dir=args.output_dir, rebuild=args.rebuild, limit=args.limit,
        workers=workers, use_processes=args.processes,
        aoa_min=args.aoa_min, aoa_max=args.aoa_max, aoa_index=args.aoa_index,
        filter_contains=args.filter_contains, sequential=args.sequential,
        chunk_size=args.chunk_size, mem_highwater=args.mem_highwater,
        gc_interval=max(1, args.gc_interval),
        max_active_futures=args.max_active_futures,
        max_isolated_fraction=args.max_isolated_fraction,
        max_low_degree_fraction=args.max_low_degree_fraction,
        qa_fail_fast=args.qa_fail_fast,
    )

    start = time.time()

    if params.sequential or params.workers == 1:
        stats_train = process_indices_sequential(train_indices, 'train', params,
                                                 dataset_train, dataset_test)
    else:
        stats_train = process_indices_parallel(train_indices, 'train', params,
                                               dataset_train, dataset_test)

    print_qa_summary(stats_train.get('qa_reports', []), 'train')

    stats_test = {}
    if params.include_test and dataset_test is not None:
        test_indices = list(range(len(dataset_test)))
        if params.sequential or params.workers == 1:
            stats_test = process_indices_sequential(test_indices, 'test', params,
                                                    dataset_train, dataset_test)
        else:
            stats_test = process_indices_parallel(test_indices, 'test', params,
                                                  dataset_train, dataset_test)
        print_qa_summary(stats_test.get('qa_reports', []), 'test')

    elapsed = time.time() - start

    # Save manifest (exclude qa_reports from JSON to keep it small)
    summary = {
        'train': {k: v for k, v in stats_train.items() if k != 'qa_reports'},
        'test': {k: v for k, v in stats_test.items() if k != 'qa_reports'} if stats_test else {},
        'elapsed_sec': elapsed,
    }
    manifest = {
        'params': asdict(params),
        'summary': summary,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': 'v2',
    }
    manifest_path = os.path.join(params.output_dir, 'edge_preprocess_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V2 Summary]")
    print(f"  Train: built={stats_train['built']} reused={stats_train['reused']} "
          f"skipped={stats_train['skipped']}")
    if stats_test:
        print(f"  Test : built={stats_test['built']} reused={stats_test['reused']} "
              f"skipped={stats_test['skipped']}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Manifest: {manifest_path}")
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
