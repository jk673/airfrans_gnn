# DDP Smoke Test (torchrun on Linux)

This repo includes a minimal, multi-GPU-ready smoke test using PyTorch Distributed Data Parallel (DDP) and PyG. It now supports a prebuilt-edges workflow to avoid rebuilding edges on the fly.

## Files
- `ddp_smoke_test.py` — Runnable entry point for torchrun. Loads a few graph `.pt` files, trains briefly, validates, and saves a checkpoint. Defaults to `prebuilt_edges/scarce/train`.
- `preprocess_edges_offline.py` — Offline builder that reads AirfRANS and writes per-graph `.pt` with edges to `prebuilt_edges/<task>/{train,test}`. Normalization is NOT applied here.
- `smoke_model_ddp.py` — Minimal model used by the smoke test.
- `smoke_loss.py` — Adjustable loss with MSE/L1 and a tiny pressure–WSS consistency term.

## Prepare data (Linux)
1) Generate prebuilt graphs once:

```bash
python preprocess_edges_offline.py \
	--root /workspace/airfrans \
	--task scarce \
	--out-dir prebuilt_edges \
	--global-radius 0.02 --surface-radius 0.01 --max-num-neighbors 48 --surface-ring
```

To accelerate, enable parallel workers (I/O-bound friendly; avoid oversubscription):

```bash
python preprocess_edges_offline.py \
	--root /workspace/airfrans \
	--task scarce \
	--out-dir prebuilt_edges \
	--global-radius 0.02 --surface-radius 0.01 --max-num-neighbors 48 --surface-ring \
	--workers 8 --mp-chunksize 8
```

This creates:
```
prebuilt_edges/
	scarce/
		train/graph_000000.pt ...
		test/graph_000000.pt  ...
```

## Run (Linux)

- Single process (sanity):

```bash
python ddp_smoke_test.py --data-dir prebuilt_edges/scarce/train --limit 4 --epochs 1
```

- Multi-GPU on a single node (e.g., 2 GPUs):

```bash
torchrun --standalone --nproc_per_node=2 ddp_smoke_test.py --data-dir prebuilt_edges/scarce/train --limit 8 --epochs 1
```

Notes:
- Default backend on Linux with CUDA is `nccl` (auto-selected). Override with `--backend gloo` if needed.
- The script saves `smoke_ddp_checkpoint.pt` on rank 0.
- If `--data-dir` has no `.pt` files, the script falls back to a tiny inline AirfRANS subset and builds edges in-process.

## Adjustable knobs
- Model: `--node-dim`, `--hidden-dim`, `--layers`, `--out-dim`
- Loss weights: `--loss-mse`, `--loss-l1`, `--loss-pw`

## Environment versions
If you run into torchrun/DDP issues, please share:
- Python version
- PyTorch version + CUDA toolkit used
- torch-geometric + companion packages (torch-scatter, torch-sparse, pyg-lib) versions

I can provide a pinned list based on your Linux server environment.
