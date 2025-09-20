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

    # ---- NEW: degree floor & knn backup ----
    min_degree: int = 2              # ensure each node has at least this many neighbors
    knn_backup_k: int = 4            # how many neighbors to add for nodes below min_degree
    knn_max_radius: float = 0.05     # optional cap (same unit as pos): ignore nn farther than this
    knn_mutual: bool = True          # add backup edge only if mutual neighbors
    backup_per_node_budget: int = 4  # how many backup edges per node at most
    final_max_degree: int = 64       # prune to keep graph tame
    length_hard_cap: float = 0.12    # drop edges longer than this (units of pos)


def hash_pos(pos: torch.Tensor) -> str:
    with torch.no_grad():
        return hashlib.sha1(pos.detach().cpu().numpy().tobytes()).hexdigest()[:16]
    
def _postfix_repair_min_degree(edge_all, pos2, min_degree: int, max_degree: int,
                               knn_k: int, knn_radius: float,
                               length_cap: float, max_iters: int = 3) -> torch.Tensor:
    """
    After pruning, iteratively add short, local backup edges so that
    every node has deg >= min_degree, without exceeding max_degree on any endpoint.
    Only connect nodes with non-zero distance; respect a length cap and a radius.
    """
    device = pos2.device
    N = pos2.size(0)

    def degree(ei):
        row, col = ei
        return (torch.bincount(row, minlength=N) + torch.bincount(col, minlength=N))

    def edge_exists(ei_set, i, j):
        # we’ll check via a set of tuples (i,j) built once per iter
        return (i, j) in ei_set or (j, i) in ei_set

    edge_all = edge_all.clone()
    for _ in range(max_iters):
        row, col = edge_all
        deg = degree(edge_all)

        need = (deg < min_degree).nonzero(as_tuple=False).view(-1)
        if need.numel() == 0:
            break

        # build a fast membership set for existing edges (only for needed subset)
        inc = torch.cat([edge_all[0].view(1,-1), edge_all[1].view(1,-1)], dim=0).t().tolist()
        ei_set = set((int(a), int(b)) for a,b in inc)

        src_list, dst_list = [], []

        with torch.no_grad():
            # local KNN over all nodes (restrict by radius/length cap)
            D = torch.cdist(pos2[need], pos2)  # [M,N]
            M = need.numel()
            arangeM = torch.arange(M, device=device)
            # self mask
            D[arangeM, need] = 1e9
            # radius cap
            if knn_radius and knn_radius > 0:
                D = D.masked_fill(D > knn_radius, 1e9)
            # pick k nearest candidates
            k = min(max(knn_k, min_degree), max(1, N-1))
            nn_idx = torch.topk(D, k=k, largest=False).indices  # [M,k]

        for r in range(M):
            i = int(need[r])
            if int(deg[i]) >= min_degree:
                continue

            js = nn_idx[r].tolist()

            # nearest-first by D
            di = D[r]
            order = torch.argsort(di[nn_idx[r]])
            js = [int(nn_idx[r][int(t)]) for t in order.tolist()]

            quota_i = min_degree - int(deg[i])
            add_to_i = 0

            for j in js:
                if add_to_i >= quota_i:
                    break
                if j == i:
                    continue
                # endpoints must have budget left
                if max_degree and int(deg[j]) >= max_degree:
                    continue
                if max_degree and int(deg[i]) >= max_degree:
                    break
                # skip existing edges
                if edge_exists(ei_set, i, j):
                    continue
                # length cap & non-zero distance
                dij = torch.norm(pos2[j] - pos2[i]).item()
                if dij <= 1e-12:
                    continue
                if length_cap and dij > length_cap:
                    continue

                # add i<->j
                src_list.append(i); dst_list.append(j)
                src_list.append(j); dst_list.append(i)

                ei_set.add((i, j)); ei_set.add((j, i))
                deg[i] += 1; deg[j] += 1
                add_to_i += 1

        if src_list:
            extra = torch.stack([torch.tensor(src_list, dtype=torch.long, device=device),
                                 torch.tensor(dst_list, dtype=torch.long, device=device)], dim=0)
            # merge & unique
            edge_all = torch.unique(torch.cat([edge_all, extra], dim=1), dim=1)
        else:
            # no progress this iteration
            break

    return edge_all


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

    # ---- NEW: ensure min degree via KNN backup (on pos2) ----
    N = pos2.size(0)
    row0, col0 = edge_all
    deg = torch.bincount(row0, minlength=N) + torch.bincount(col0, minlength=N)
    need = (deg < p.min_degree).nonzero(as_tuple=False).view(-1)
    if need.numel() > 0 and p.knn_backup_k > 0:
        # compute distances from need -> all (avoid heavy all-pairs by slicing)
        # (for typical AirfRANS sizes this cdist 사용으로 충분)
        with torch.no_grad():
            D = torch.cdist(pos2[need], pos2)  # [M, N]
            D[torch.arange(need.numel()), need] = 1e9  # remove self
            # optional distance cap
            if p.knn_max_radius and p.knn_max_radius > 0:
                mask_far = D > p.knn_max_radius
                D = D.masked_fill(mask_far, 1e9)
            k = max(p.knn_backup_k, p.min_degree)
            k = min(k, max(1, N-1))
            nn_idx = torch.topk(D, k=k, largest=False).indices  # [M, k]
        src = need.repeat_interleave(k)
        dst = nn_idx.reshape(-1)
        extra = torch.stack([src, dst], dim=0)
        # make undirected
        extra = torch.cat([extra, extra.flip(0)], dim=1)
        edge_all = torch.unique(torch.cat([edge_all, extra], dim=1), dim=1)

        nn_idx = torch.topk(D, k=k, largest=False).indices  # [M, k]

        # ---- mutual KNN (optional) ----
        if p.knn_mutual:
            # compute D_all for symmetry check but avoid full NxN if large:
            # Here we restrict to candidates only (union of need and its NN)
            cand = torch.unique(torch.cat([need, nn_idx.reshape(-1)], dim=0))
            Dc = torch.cdist(pos2[cand], pos2[cand])
            Dc[torch.arange(cand.numel()), torch.arange(cand.numel())] = 1e9
            k2 = min(k, max(1, cand.numel()-1))
            topk_in_cand = torch.topk(Dc, k=k2, largest=False).indices  # [C, k2]
            # map back to global indices
            cand_map = {int(cand[i]): i for i in range(cand.numel())}
            # mutual mask
            mutual_mask = []
            for i_local, i in enumerate(need):
                js = nn_idx[i_local]  # global indices
                keep_for_i = []
                i_c = cand_map[int(i)]
                js_c = [cand_map[int(j)] for j in js.tolist() if int(j) in cand_map]
                neigh_i_c = set(topk_in_cand[i_c].tolist())
                for j, j_c in zip(js.tolist(), js_c):
                    # mutual: j is in i's topk and i is in j's topk within cand set
                    neigh_j_c = set(topk_in_cand[j_c].tolist())
                    if i_c in neigh_j_c:
                        keep_for_i.append(j)
                mutual_mask.append(torch.tensor(keep_for_i, dtype=torch.long, device=pos2.device))
            # stack mutual neighbors, respecting budget
            src_list, dst_list = [], []
            for i, js in zip(need.tolist(), mutual_mask):
                if js.numel() == 0:
                    continue
                # nearest-first & budget
                # sort js by distance D[idx]
                di = D[ (need==i).nonzero(as_tuple=False).view(-1)[0] ]
                order = torch.argsort(di[js])  # nearest first
                js = js[order][:p.backup_per_node_budget]
                src_list.append(torch.full((js.numel(),), i, dtype=torch.long, device=pos2.device))
                dst_list.append(js)
            if src_list:
                src = torch.cat(src_list); dst = torch.cat(dst_list)
            else:
                src = torch.empty(0, dtype=torch.long, device=pos2.device)
                dst = torch.empty(0, dtype=torch.long, device=pos2.device)
        else:
            # no mutual check, but still enforce budget nearest-first
            src = need.repeat_interleave(k)
            dst = nn_idx.reshape(-1)
            # sort within each i by distance and cut to budget
            src2, dst2 = [], []
            for r, i in enumerate(need.tolist()):
                js = dst[r*k:(r+1)*k]
                di = D[r]
                order = torch.argsort(di[js])
                js = js[order][:p.backup_per_node_budget]
                src2.append(torch.full((js.numel(),), i, dtype=torch.long, device=pos2.device))
                dst2.append(js)
            if src2:
                src = torch.cat(src2); dst = torch.cat(dst2)
            else:
                src = torch.empty(0, dtype=torch.long, device=pos2.device)
                dst = torch.empty(0, dtype=torch.long, device=pos2.device)

    extra = torch.stack([src, dst], dim=0)
    extra = torch.cat([extra, extra.flip(0)], dim=1)  # undirected
    # drop edges longer than hard cap (optional)
    if p.length_hard_cap and p.length_hard_cap > 0:
        dvec_extra = pos2[extra[1]] - pos2[extra[0]]
        len_extra = dvec_extra.norm(dim=1)
        mask = len_extra <= p.length_hard_cap
        extra = extra[:, mask]
    edge_all = torch.unique(torch.cat([edge_all, extra], dim=1), dim=1)

    # ---- OPTIONAL: final degree cap with nearest-first pruning ----
    if p.final_max_degree and p.final_max_degree > 0:
        row, col = edge_all
        # compute lengths for sorting
        dvec = pos2[col] - pos2[row]
        length_all = dvec.norm(dim=1)
        # for each node, keep up to max_degree shortest incident edges
        # build adjacency list indices
        deg = torch.bincount(row, minlength=N) + torch.bincount(col, minlength=N)
        keep_mask = torch.ones(edge_all.size(1), dtype=torch.bool, device=row.device)
        # prune pass: for high-degree nodes, drop longest incident edges
        # (simple greedy per-node)
        for node in torch.nonzero(deg > p.final_max_degree, as_tuple=False).view(-1).tolist():
            inc_ids = ((row == node) | (col == node)).nonzero(as_tuple=False).view(-1)
            if inc_ids.numel() <= p.final_max_degree:
                continue
            sel = torch.argsort(length_all[inc_ids])[:p.final_max_degree]
            keep = inc_ids[sel]
            drop = torch.tensor(list(set(inc_ids.tolist()) - set(keep.tolist())), device=row.device, dtype=torch.long)
            keep_mask[drop] = False
        edge_all = edge_all[:, keep_mask]

    # ---- NEW: post-fix ensure min degree (after pruning) ----
    row_pf, col_pf = edge_all
    deg_pf = torch.bincount(row_pf, minlength=N) + torch.bincount(col_pf, minlength=N)
    need_pf = (deg_pf < p.min_degree).nonzero(as_tuple=False).view(-1)
    if need_pf.numel() > 0 and p.knn_backup_k > 0:
        with torch.no_grad():
            D = torch.cdist(pos2[need_pf], pos2)   # [M, N]
            D[torch.arange(need_pf.numel()), need_pf] = 1e9
            if p.knn_max_radius and p.knn_max_radius > 0:
                D = D.masked_fill(D > p.knn_max_radius, 1e9)
            k = min(max(p.knn_backup_k, p.min_degree), max(1, N-1))
            nn_idx = torch.topk(D, k=k, largest=False).indices  # [M, k]
        # (선택) mutual 검사 & budget 적용 (상세 구현은 이전과 동일 패턴)
        src_list, dst_list = [], []
        for r, i in enumerate(need_pf.tolist()):
            js = nn_idx[r]
            # 가까운 순 정렬 + per-node budget
            di = D[r]
            order = torch.argsort(di[js])
            js = js[order][:p.backup_per_node_budget]
            # 이미 final_max_degree에 걸린 노드는 skip (cap 유지)
            if deg_pf[i] >= p.final_max_degree:
                continue
            # cap을 넘지 않도록 필요한 개수만 추가
            quota = max(0, p.min_degree - int(deg_pf[i]))
            if quota <= 0 or js.numel() == 0:
                continue
            js = js[:quota]
            src_list.append(torch.full((js.numel(),), i, dtype=torch.long, device=pos2.device))
            dst_list.append(js)
            deg_pf[i] += js.numel()
        if src_list:
            extra = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
            extra = torch.cat([extra, extra.flip(0)], dim=1)
            # 길이 하드캡 재적용
            if p.length_hard_cap and p.length_hard_cap > 0:
                dvec_extra = pos2[extra[1]] - pos2[extra[0]]
                len_extra = dvec_extra.norm(dim=1)
                extra = extra[:, len_extra <= p.length_hard_cap]
            edge_all = torch.unique(torch.cat([edge_all, extra], dim=1), dim=1)

    edge_all = _postfix_repair_min_degree(
    edge_all=edge_all,
    pos2=pos2,
    min_degree=p.min_degree,          # ex) 3
    max_degree=p.final_max_degree,    # ex) 48~64
    knn_k=p.knn_backup_k,             # ex) 6
    knn_radius=p.knn_max_radius,      # ex) 0.08
    length_cap=p.length_hard_cap,     # ex) 0.12
    max_iters=3
)

    row, col = edge_all
    # drop self-loops
    mask_self = edge_all[0] != edge_all[1]
    edge_all = edge_all[:, mask_self]
    # drop near-duplicate edges by sorting unique already done above
    row, col = edge_all
    dvec = pos2[col] - pos2[row]
    dist = dvec.norm(dim=1, keepdim=True).clamp_min(1e-12)
    dist = dvec.norm(dim=1, keepdim=True)
    # drop zero-length (or tiny) edges defensively
    nonzero = (dist.view(-1) > 1e-9)
    edge_all = edge_all[:, nonzero]
    dvec = dvec[nonzero]
    dist = dist[nonzero]
    dist = dist.clamp_min(1e-12)
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

    # Primary edge_attr schema (backward compatible):
    #   [0]=dist (edge length), [1]=dir_x, [2]=dir_y, [3]=cos_n, [4]=is_surface_pair
    edge_attr = torch.cat([dist, dir_xy, cos_n, surf_pair], dim=1)

    # Auxiliary schema for physics/divergence computations:
    #   edge_attr_dxdy = [dx, dy, dist]
    # This directly stores the edge vector components (in the same units as data.pos)
    edge_attr_dxdy = torch.cat([dvec, dist], dim=1)
    data.edge_index = edge_all
    data.edge_attr = edge_attr
    # Attach auxiliary edge attributes useful for physics losses
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

    # === NEW: prune isolated nodes before returning ===
    # try:
    #     from utils_prune import prune_isolated_nodes
    #     data = prune_isolated_nodes(
    #         data,
    #         node_attr_keys=['pos','x','y','node_area',
    #                         'is_wall','is_inlet','is_outlet','is_farfield']
    #     )
    # except Exception as e:
    #     print(f"[Warn] prune_isolated_nodes failed: {e}")

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
                'edge_attr_dxdy': getattr(d, 'edge_attr_dxdy', None),
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
                             dataset_train: AirfRANS, dataset_test: Optional[AirfRANS]) -> Dict[str, Any]:
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
                               dataset_train: AirfRANS, dataset_test: Optional[AirfRANS]) -> Dict[str, Any]:
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