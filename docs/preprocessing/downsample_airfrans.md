# downsample_airfrans.py

Step 1 of the data preparation pipeline. Downsamples raw AirfRANS graphs from 100k+ nodes to a target range (default 15k-30k) using adaptive voxel sampling while preserving all surface (airfoil) nodes. No edges are created at this stage.

## Usage

```bash
python downsample_airfrans.py --root Dataset [OPTIONS]
```

## Options

### Required

| Option | Type | Description |
|--------|------|-------------|
| `--root` | `str` | Path to the raw AirfRANS dataset directory (e.g. `Dataset`). |

### Dataset

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--task` | `str` | `scarce` | AirfRANS task split. Choices: `scarce`, `full`. Determines which subset of simulations to use. |
| `--out-dir` | `str` | `downsampled_graphs` | Base output directory. Files are saved under `<out-dir>/<task>/train/` and `<out-dir>/<task>/test/`. |
| `--limit-train` | `int` | `None` | Process only the first N training samples. Useful for quick testing. When `None`, all training samples are processed. |
| `--limit-test` | `int` | `None` | Process only the first N test samples. Useful for quick testing. When `None`, all test samples are processed. |

### Downsampling

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--target-min-nodes` | `int` | `15000` | Lower bound of the target node count after downsampling. The adaptive algorithm tries to keep the result within `[min, max]`. |
| `--target-max-nodes` | `int` | `30000` | Upper bound of the target node count after downsampling. |
| `--voxel-frac` | `float` | `0.01` | Initial voxel size as a fraction of the estimated chord length. Smaller values retain more nodes; larger values are more aggressive. |
| `--voxel-iters` | `int` | `5` | Maximum number of binary-search-style iterations to converge the node count into the target range. |

## Examples

```bash
# Default: downsample scarce split, target 15k-30k nodes
python downsample_airfrans.py --root Dataset

# Full split with tighter target range
python downsample_airfrans.py --root Dataset --task full \
    --target-min-nodes 20000 --target-max-nodes 25000

# Quick test run with 5 training + 2 test samples
python downsample_airfrans.py --root Dataset \
    --limit-train 5 --limit-test 2

# Custom output directory and more aggressive downsampling
python downsample_airfrans.py --root Dataset --out-dir my_output \
    --voxel-frac 0.02 --voxel-iters 10
```

## Output Structure

```
<out-dir>/<task>/
  train/
    graph_000000.pt
    graph_000001.pt
    ...
  test/
    graph_000000.pt
    ...
```

Each `.pt` file is a `torch_geometric.data.Data` object containing:

| Field | Shape | Description |
|-------|-------|-------------|
| `x` | `[N, 5]` | Node features (inlet velocity, wall distance, wall normals) |
| `y` | `[N, 4]` | Ground-truth targets (u, v, pressure, nu_t) |
| `pos` | `[N, 2]` | Node positions (x, y coordinates) |
| `surf` | `[N]` | Surface mask (bool), if present in source data |
| `orig_index` | scalar | Index in the original AirfRANS dataset for alignment |

## Algorithm

1. Estimate chord length from the x-extent of node positions.
2. Compute `voxel_size = chord * voxel_frac`.
3. Identify surface nodes (wall distance near zero or non-zero wall normals) -- these are always kept.
4. Apply voxel grid to volume (non-surface) nodes, keeping one representative per voxel.
5. If the resulting node count is outside `[target-min-nodes, target-max-nodes]`, adjust `voxel_frac` proportionally and repeat (up to `voxel-iters` times).
6. Save the best result.

## Next Step

After downsampling, build edges with:

```bash
python build_edges_from_downsampled.py --in-dir downsampled_graphs --out-dir prebuilt_edges --task scarce
```
