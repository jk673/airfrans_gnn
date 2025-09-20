#!/usr/bin/env python
"""
Build edges for downsampled AirfRANS graphs (step 2 of 2).

Loads per-graph .pt files from downsample_airfrans.py and adds edge_index/edge_attr using
preprocess_airfrans_edges.build_edges_for_graph with the default radii and surface ring.
"""
import os, argparse
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def _import_preprocess_module(path: str = 'preprocess_airfrans_edges.py'):
    import importlib.util, sys
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location('pre_air', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not import preprocess module from {path}')
    pre_air = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = pre_air  # type: ignore[attr-defined]
    spec.loader.exec_module(pre_air)  # type: ignore[union-attr]
    return pre_air


def main():
    ap = argparse.ArgumentParser(description='Add edges to downsampled graphs.')
    ap.add_argument('--in-dir', type=str, required=True, help='Input root containing <task>/{train,test}/graph_*.pt')
    ap.add_argument('--out-dir', type=str, required=True, help='Output root for graphs with edges')
    ap.add_argument('--task', type=str, default='scarce', choices=['scarce','full'])
    ap.add_argument('--global-radius', type=float, default=0.02)
    ap.add_argument('--surface-radius', type=float, default=0.01)
    ap.add_argument('--max-num-neighbors', type=int, default=48)
    ap.add_argument('--surface-ring', action='store_true', default=True)
    ap.add_argument('--denormalize', action='store_true', help='If set, denormalize pos/x/y using attached norm params if present')
    # NEW: degree floor & knn backup knobs
    ap.add_argument('--min-degree', type=int, default=2)
    ap.add_argument('--knn-backup-k', type=int, default=4)
    ap.add_argument('--knn-max-radius', type=float, default=0.05)
    args = ap.parse_args()

    pre_air = _import_preprocess_module('preprocess_airfrans_edges.py')
    class EdgeParams(pre_air.Params):
        pass
    params = EdgeParams(root='.', preset='scarce', task=args.task, include_test=False,
                        global_radius=args.global_radius, surface_radius=args.surface_radius,
                        max_num_neighbors=args.max_num_neighbors, surface_ring=args.surface_ring,
                        output_dir='__unused__', rebuild=True, limit=None, workers=0, use_processes=False,
                        aoa_min=None, aoa_max=None, aoa_index=2, filter_contains=None, sequential=True,
                        chunk_size=1, mem_highwater=100.0, gc_interval=1, max_active_futures=0,
                        min_degree=args.min_degree, knn_backup_k=args.knn_backup_k, knn_max_radius=args.knn_max_radius)
    
    in_root = os.path.join(args.in_dir, args.task)
    out_root = os.path.join(args.out_dir, args.task)
    for split in ('train','test'):
        src = os.path.join(in_root, split)
        dst = os.path.join(out_root, split)
        os.makedirs(dst, exist_ok=True)
        files = sorted([f for f in os.listdir(src) if f.startswith('graph_') and f.endswith('.pt')]) if os.path.isdir(src) else []
        saved = 0
        for i, fn in enumerate(tqdm(files, desc=f'Edges -> {dst}')):
            path = os.path.join(src, fn)
            try:
                d = torch.load(path, map_location='cpu', weights_only=False)
                if not isinstance(d, Data):
                    d = Data(**d)
                # Clone safely
                d2 = d.clone()
                # Optionally denormalize positions and node features/targets if norm params are attached
                if args.denormalize:
                    try:
                        # Denormalize x and pos (positions are typically in x[:,:2] or d.pos)
                        if hasattr(d2, 'x_norm_params') and d2.x_norm_params is not None and isinstance(getattr(d2, 'x', None), torch.Tensor):
                            xn = d2.x_norm_params
                            x_mean = xn['mean']; x_scale = xn['scale']
                            # Convert to tensors, match dtype if available
                            x_mean_t = torch.as_tensor(x_mean)
                            x_scale_t = torch.as_tensor(x_scale)
                            d2.x = d2.x * x_scale_t + x_mean_t
                        if hasattr(d2, 'pos') and d2.pos is not None and hasattr(d2, 'pos_norm_params') and d2.pos_norm_params is not None:
                            pn = d2.pos_norm_params
                            p_mean = pn['mean']; p_scale = pn['scale']
                            p_mean_t = torch.as_tensor(p_mean, dtype=d2.pos.dtype)
                            p_scale_t = torch.as_tensor(p_scale, dtype=d2.pos.dtype)
                            d2.pos = d2.pos * p_scale_t + p_mean_t
                        # Targets (y)
                        if isinstance(getattr(d2, 'y', None), torch.Tensor) and hasattr(d2, 'y_norm_params') and d2.y_norm_params is not None:
                            yn = d2.y_norm_params
                            y_mean = yn['mean']; y_scale = yn['scale']
                            y_mean_t = torch.as_tensor(y_mean)
                            y_scale_t = torch.as_tensor(y_scale)
                            d2.y = d2.y * y_scale_t + y_mean_t
                    except Exception as ee:
                        print(f'[edge] warning: denormalize failed for {path}: {ee}')

                d2 = pre_air.build_edges_for_graph(d2, params)
                if hasattr(d2, 'edge_index') and d2.edge_index is not None and d2.edge_index.dtype != torch.long:
                    d2.edge_index = d2.edge_index.long()
                # Decide output filename based on original dataset index if present
                orig_idx = None
                try:
                    oi = getattr(d, 'orig_index', None)
                    if oi is not None:
                        orig_idx = int(oi.item() if hasattr(oi, 'item') else int(oi))
                except Exception:
                    orig_idx = None
                out_name = fn if orig_idx is None else f'graph_{orig_idx:06d}.pt'
                # Ensure auxiliary edge attributes are saved
                torch.save(d2, os.path.join(dst, out_name))
                saved += 1
            except Exception as e:
                print(f'[edge] failed for {path}: {e}')
        print(f'Saved edges for {split}: {saved} files in {dst}')


if __name__ == '__main__':
    main()
