#!/usr/bin/env python
import os
import argparse
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch_geometric.data import Batch

from ddp_training_old import load_consolidated_graphs
from smoke_model_ddp import SmokeCFDModel
from smoke_loss import AdjustableSmokeLoss


def collate_graphs(batch_list):
    if len(batch_list) == 1:
        return batch_list[0]
    return Batch.from_data_list(batch_list)


def main():
    parser = argparse.ArgumentParser(description="DDP smoke test for multi-GPU torchrun")
    parser.add_argument("--data-dir", type=str, default="prebuilt_edges/scarce/train", help="Directory of prebuilt graph .pt files (from preprocess_edges_offline.py)")
    parser.add_argument("--limit", type=int, default=4, help="Number of graph files to load for smoke test")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--node-dim", type=int, default=5, help="Expected node feature dimension")
    parser.add_argument("--out-dim", type=int, default=4, help="Output channels")
    parser.add_argument("--loss-mse", type=float, default=1.0)
    parser.add_argument("--loss-l1", type=float, default=0.0)
    parser.add_argument("--loss-pw", type=float, default=0.0, help="Pressure-WSS consistency weight")
    parser.add_argument("--backend", type=str, default=None, help="Process group backend (nccl/gloo). Defaults to nccl on Linux with CUDA, else gloo.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # DDP setup via torchrun
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = world_size > 1

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Backend selection: prefer NCCL when CUDA available; else gloo
    chosen_backend = args.backend
    if chosen_backend is None:
        chosen_backend = "nccl" if torch.cuda.is_available() else "gloo"

    if use_ddp:
        dist.init_process_group(backend=chosen_backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[DDP Smoke] world_size={world_size}, device={device}, backend={chosen_backend}")

    # Load a few graphs to CPU
    graphs = load_consolidated_graphs(data_dir=args.data_dir, file_pattern="*graph_*.pt", limit=args.limit)
    if len(graphs) == 0:
        if rank == 0:
            print(f"No prebuilt graphs found in '{args.data_dir}'. Falling back to inline AirfRANS subset + edge build.")
        # Fallback: tiny inline pipeline
        try:
            from torch.utils.data import Subset
            from torch_geometric.datasets import AirfRANS
            from torch_geometric.data import Data as _Data
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location('pre_air', os.path.abspath('preprocess_airfrans_edges.py'))
            pre_air = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = pre_air
            spec.loader.exec_module(pre_air)
            class EdgeParams(pre_air.Params):
                pass
            p = EdgeParams(root='Dataset', preset='scarce', task='scarce', include_test=False, global_radius=0.02, surface_radius=0.01, max_num_neighbors=48, surface_ring=True,
                           output_dir='__inline__', rebuild=True, limit=None, workers=0, use_processes=False, aoa_min=None, aoa_max=None, aoa_index=2, filter_contains=None, sequential=True,
                           chunk_size=1, mem_highwater=100.0, gc_interval=1, max_active_futures=0)
            try:
                ds_train = AirfRANS(root='Dataset', train=True, task='scarce')
            except TypeError:
                ds_train = AirfRANS(root='Dataset', train=True)
            n_take = min(len(ds_train), max(1, args.limit))
            ds_small = Subset(ds_train, list(range(n_take)))
            graphs = [pre_air.build_edges_for_graph(_Data(**{k: v for k,v in ds_small[i]}), p) for i in range(n_take)]
        except Exception as e:
            if rank == 0:
                print(f"Inline fallback failed: {e}")
            if use_ddp:
                dist.destroy_process_group()
            return
    for i, g in enumerate(graphs):
        if hasattr(g, 'x') and g.x is not None and g.x.is_cuda:
            graphs[i] = g.cpu()

    # Tiny split
    split = max(1, int(len(graphs) * 0.8))
    train_graphs = graphs[:split]
    val_graphs = graphs[split:] if split < len(graphs) else graphs[:1]

    # Wrap in trivial dataset
    from torch.utils.data import Dataset as _TorchDataset
    class GraphListDataset(_TorchDataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    train_ds = GraphListDataset(train_graphs)
    val_ds = GraphListDataset(val_graphs)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_graphs,
    )

    # Model and loss
    model = SmokeCFDModel(node_feat_dim=args.node_dim, hidden_dim=args.hidden_dim, output_dim=args.out_dim, num_layers=args.layers)
    model = model.to(device)
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank], output_device=rank)

    loss_fn = AdjustableSmokeLoss(weights={
        'mse': args.loss_mse,
        'l1': args.loss_l1,
        'pressure_wss': args.loss_pw,
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def _step(batch):
        batch = batch.to(device)
        # sanity: ensure presence of y
        if not hasattr(batch, 'y') or batch.y is None:
            raise RuntimeError("Batch has no 'y' target; ensure your consolidated graphs include .y")
        pred = model(batch)
        target = batch.y.to(device)
        comp = loss_fn(pred, target, batch)
        return pred, target, comp

    # Train loop (very short)
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        model.train()
        t0 = time.time()
        tr_loss = 0.0
        tr_steps = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            try:
                _, _, comp = _step(batch)
                comp['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += float(comp['total'].item())
                tr_steps += 1
                if rank == 0 and i % max(1, len(train_loader)//5 or 1) == 0:
                    print(f"  step {i+1}/{len(train_loader)}: loss={comp['total'].item():.4f}")
            except Exception as e:
                if rank == 0:
                    print(f"  train batch {i} error: {e}")
                continue
        if rank == 0:
            dt = time.time() - t0
            avg = tr_loss / max(1, tr_steps)
            print(f" train avg loss: {avg:.4f} (time {dt:.1f}s)")

        # quick val
        model.eval()
        vl_loss = 0.0
        vl_steps = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                try:
                    _, _, comp = _step(batch)
                    vl_loss += float(comp['total'].item())
                    vl_steps += 1
                except Exception as e:
                    if rank == 0:
                        print(f"  val batch {i} error: {e}")
                    continue
        if rank == 0 and vl_steps > 0:
            print(f" val avg loss: {vl_loss / vl_steps:.4f}")

    # Save checkpoint from rank 0
    if rank == 0:
        state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({'model': state, 'meta': vars(args)}, 'smoke_ddp_checkpoint.pt')
        print("[DDP Smoke] Saved checkpoint: smoke_ddp_checkpoint.pt")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
