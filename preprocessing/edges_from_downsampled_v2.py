#!/usr/bin/env python
"""
Build edges for downsampled AirfRANS graphs — V2 (step 2 of 2).

Changes from V1:
  - Uses preprocess_airfrans_edges_v2 (corrected degree, QA reports).
  - Prints per-split QA summary after processing.
"""
import os, argparse, sys
from typing import Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from preprocessing.config_v2 import EdgeConfigV2
from torch_geometric.data import Data
from tqdm import tqdm

# Ensure project root is on sys.path for src imports
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from src.preprocessing import build_bc_masks_airfrans


def _import_preprocess_module(path: Optional[str] = None):
    import importlib.util, sys
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'src', 'edge_construction.py')
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location('pre_air_v2', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not import preprocess module from {path}')
    pre_air = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pre_air
    spec.loader.exec_module(pre_air)
    return pre_air


def _parse_args() -> 'EdgeConfigV2':
    from preprocessing.config_v2 import EdgeConfigV2
    defaults = EdgeConfigV2()
    ap = argparse.ArgumentParser(description='Add edges to downsampled graphs — V2.')
    ap.add_argument('--in-dir', type=str, default=defaults.in_dir)
    ap.add_argument('--out-dir', type=str, default=defaults.out_dir)
    ap.add_argument('--task', type=str, default=defaults.task, choices=['scarce', 'full'])
    ap.add_argument('--global-radius', type=float, default=defaults.global_radius)
    ap.add_argument('--surface-radius', type=float, default=defaults.surface_radius)
    ap.add_argument('--max-num-neighbors', type=int, default=defaults.max_num_neighbors)
    ap.add_argument('--surface-ring', action='store_true', default=defaults.surface_ring)
    ap.add_argument('--denormalize', action='store_true', default=defaults.denormalize)
    ap.add_argument('--min-degree', type=int, default=defaults.min_degree)
    ap.add_argument('--knn-backup-k', type=int, default=defaults.knn_backup_k)
    ap.add_argument('--knn-max-radius', type=float, default=defaults.knn_max_radius)
    ap.add_argument('--max-isolated-fraction', type=float, default=defaults.max_isolated_fraction)
    ap.add_argument('--max-low-degree-fraction', type=float, default=defaults.max_low_degree_fraction)
    ap.add_argument('--qa-fail-fast', action='store_true', default=defaults.qa_fail_fast)
    args = ap.parse_args()
    return EdgeConfigV2(
        in_dir=args.in_dir, out_dir=args.out_dir, task=args.task,
        global_radius=args.global_radius, surface_radius=args.surface_radius,
        max_num_neighbors=args.max_num_neighbors, surface_ring=args.surface_ring,
        denormalize=args.denormalize, min_degree=args.min_degree,
        knn_backup_k=args.knn_backup_k, knn_max_radius=args.knn_max_radius,
        max_isolated_fraction=args.max_isolated_fraction,
        max_low_degree_fraction=args.max_low_degree_fraction,
        qa_fail_fast=args.qa_fail_fast,
    )


def run(cfg: 'EdgeConfigV2'):
    pre_air = _import_preprocess_module()

    params = pre_air.Params(
        root='.', preset='scarce', task=cfg.task, include_test=False,
        global_radius=cfg.global_radius, surface_radius=cfg.surface_radius,
        max_num_neighbors=cfg.max_num_neighbors, surface_ring=cfg.surface_ring,
        output_dir='__unused__', rebuild=True, limit=None, workers=0,
        use_processes=False, aoa_min=None, aoa_max=None, aoa_index=2,
        filter_contains=None, sequential=True, chunk_size=1,
        mem_highwater=100.0, gc_interval=1, max_active_futures=0,
        min_degree=cfg.min_degree, knn_backup_k=cfg.knn_backup_k,
        knn_max_radius=cfg.knn_max_radius,
        max_isolated_fraction=cfg.max_isolated_fraction,
        max_low_degree_fraction=cfg.max_low_degree_fraction,
        qa_fail_fast=cfg.qa_fail_fast,
    )

    in_root = os.path.join(cfg.in_dir, cfg.task)
    out_root = os.path.join(cfg.out_dir, cfg.task)

    for split in ('train', 'test'):
        src = os.path.join(in_root, split)
        dst = os.path.join(out_root, split)
        os.makedirs(dst, exist_ok=True)
        files = sorted([f for f in os.listdir(src)
                        if f.startswith('graph_') and f.endswith('.pt')]) \
            if os.path.isdir(src) else []
        saved = 0
        qa_reports = []
        for fn in tqdm(files, desc=f'Edges -> {dst}'):
            path = os.path.join(src, fn)
            try:
                d = torch.load(path, map_location='cpu', weights_only=False)
                if not isinstance(d, Data):
                    d = Data(**d)
                d2 = d.clone()
                if cfg.denormalize:
                    try:
                        x_val = getattr(d2, 'x', None)
                        if (hasattr(d2, 'x_norm_params') and d2.x_norm_params is not None
                                and isinstance(x_val, torch.Tensor)):
                            xn = d2.x_norm_params
                            x_mean = xn.get('mean')
                            x_scale = xn.get('scale')
                            if x_mean is not None and x_scale is not None:
                                d2.x = x_val * torch.as_tensor(x_scale, dtype=x_val.dtype) \
                                        + torch.as_tensor(x_mean, dtype=x_val.dtype)
                        if (hasattr(d2, 'pos') and d2.pos is not None
                                and hasattr(d2, 'pos_norm_params')
                                and d2.pos_norm_params is not None):
                            pn = d2.pos_norm_params
                            p_mean = pn.get('mean')
                            p_scale = pn.get('scale')
                            if p_mean is not None and p_scale is not None:
                                d2.pos = d2.pos * torch.as_tensor(p_scale, dtype=d2.pos.dtype) \
                                         + torch.as_tensor(p_mean, dtype=d2.pos.dtype)
                        y_val = getattr(d2, 'y', None)
                        if (isinstance(y_val, torch.Tensor)
                                and hasattr(d2, 'y_norm_params')
                                and d2.y_norm_params is not None):
                            yn = d2.y_norm_params
                            y_mean = yn.get('mean')
                            y_scale = yn.get('scale')
                            if y_mean is not None and y_scale is not None:
                                d2.y = y_val * torch.as_tensor(y_scale) \
                                       + torch.as_tensor(y_mean)
                    except Exception as ee:
                        print(f'[edge] warning: denormalize failed for {path}: {ee}')

                d2, qa = pre_air.build_edges_for_graph(d2, params)
                qa_reports.append(qa)
                pre_air.check_qa_thresholds(qa, params, graph_id=fn)

                if (hasattr(d2, 'edge_index') and d2.edge_index is not None
                        and d2.edge_index.dtype != torch.long):
                    d2.edge_index = d2.edge_index.long()
                # Bake BC masks into the graph so all downstream code has them
                d2 = build_bc_masks_airfrans(d2)
                _N = d2.num_nodes
                _wall_n = int(getattr(d2, 'is_wall', torch.zeros(0)).sum())
                if _wall_n == 0:
                    print(f'[bc] WARNING: {fn} has 0 wall nodes out of {_N}')

                orig_idx = None
                try:
                    oi = getattr(d, 'orig_index', None)
                    if oi is not None:
                        orig_idx = int(oi.item() if hasattr(oi, 'item') else int(oi))
                except Exception:
                    orig_idx = None
                out_name = fn if orig_idx is None else f'graph_{orig_idx:06d}.pt'
                torch.save(d2, os.path.join(dst, out_name))
                saved += 1
            except Exception as e:
                print(f'[edge] failed for {path}: {e}')

        print(f'[V2] Saved edges for {split}: {saved} files in {dst}')
        pre_air.print_qa_summary(qa_reports, split)
        # BC mask summary: verify masks were baked into saved graphs
        _bc_check_files = sorted([f for f in os.listdir(dst) if f.startswith('graph_') and f.endswith('.pt')])[:3]
        for _cf in _bc_check_files:
            _cg = torch.load(os.path.join(dst, _cf), map_location='cpu', weights_only=False)
            _cn = _cg.num_nodes
            _masks = {k: int(getattr(_cg, k, torch.zeros(0)).sum()) for k in ['is_wall', 'is_inlet', 'is_outlet', 'is_farfield']}
            print(f'  [bc verify] {_cf}: N={_cn} {_masks}')


def main():
    run(_parse_args())


if __name__ == '__main__':
    main()
