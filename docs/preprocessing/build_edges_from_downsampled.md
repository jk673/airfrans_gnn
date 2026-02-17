# build_edges_from_downsampled.py

Step 2 of the data preparation pipeline. Loads downsampled per-graph `.pt` files produced by `downsample_airfrans.py` and adds `edge_index` / `edge_attr` using radius-graph construction with KNN backup for isolated nodes.

## Usage

```bash
python build_edges_from_downsampled.py --in-dir <input> --out-dir <output> [OPTIONS]
```

## Options

### Required

| Option | Type | Description |
|--------|------|-------------|
| `--in-dir` | `str` | Input directory containing `<task>/{train,test}/graph_*.pt` (output of `downsample_airfrans.py`). |
| `--out-dir` | `str` | Output directory for graphs with edges added. |

### Dataset

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--task` | `str` | `scarce` | AirfRANS task split. Choices: `scarce`, `full`. |

### Edge Construction

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--global-radius` | `float` | `0.02` | Radius for connecting all node pairs. Two nodes within this distance get an edge. |
| `--surface-radius` | `float` | `0.01` | Tighter radius used exclusively among surface (airfoil) nodes for finer boundary resolution. |
| `--max-num-neighbors` | `int` | `48` | Maximum number of neighbors per node in the radius graph. |
| `--surface-ring` | flag | `True` | Create ring connectivity along the airfoil surface. Enabled by default. |

### Isolated Node Recovery

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--min-degree` | `int` | `2` | Minimum acceptable node degree. Nodes below this threshold receive KNN backup edges. |
| `--knn-backup-k` | `int` | `4` | Number of nearest neighbors to connect for under-connected nodes. |
| `--knn-max-radius` | `float` | `0.05` | Maximum search radius for KNN backup edges. |

### Denormalization

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--denormalize` | flag | `False` | Reverse normalization on `pos`, `x`, and `y` before building edges. Only takes effect when the graph has `x_norm_params`, `pos_norm_params`, or `y_norm_params` attached. |

## Examples

```bash
# Default: build edges for scarce split
python build_edges_from_downsampled.py \
    --in-dir downsampled_graphs --out-dir prebuilt_edges_v2 --task scarce

# Wider radius with stronger isolated-node recovery
python build_edges_from_downsampled.py \
    --in-dir downsampled_graphs --out-dir prebuilt_edges_v2 \
    --global-radius 0.03 --min-degree 3 --knn-backup-k 6

# Full split with denormalization
python build_edges_from_downsampled.py \
    --in-dir downsampled_graphs --out-dir prebuilt_edges_v2 \
    --task full --denormalize
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

Each `.pt` file is the input graph augmented with:

| Field | Shape | Description |
|-------|-------|-------------|
| `edge_index` | `[2, E]` | COO edge connectivity (dtype `torch.long`) |
| `edge_attr` | `[E, 5]` | Primary edge features for message passing (see below) |
| `edge_attr_dxdy` | `[E, 3]` | Auxiliary edge features for physics loss (see below) |
| `edge_meta` | `dict` | Metadata: radii used, edge/surface-edge counts, schema descriptions |

All original fields (`x`, `y`, `pos`, `surf`, `orig_index`) are preserved.

### Edge Feature Schemas

Two edge feature tensors are stored per graph, each serving a different purpose.

**`edge_attr`** — Primary features (model input for message passing)

Given an edge from node `row` to node `col`:

| Index | Name | Computation |
|-------|------|-------------|
| 0 | `dist` | Euclidean distance `‖pos[col] - pos[row]‖` |
| 1 | `dir_x` | Normalized direction x-component `dx / dist` |
| 2 | `dir_y` | Normalized direction y-component `dy / dist` |
| 3 | `cos_n` | Cosine similarity between wall normals of both nodes (`x[:,3:5]`). Zero if normals are unavailable. |
| 4 | `is_surface_pair` | `1.0` if both endpoints are surface nodes, `0.0` otherwise |

**`edge_attr_dxdy`** — Auxiliary features (physics loss)

| Index | Name | Computation |
|-------|------|-------------|
| 0 | `dx` | Raw x-displacement `pos[col].x - pos[row].x` |
| 1 | `dy` | Raw y-displacement `pos[col].y - pos[row].y` |
| 2 | `dist` | Euclidean distance `‖(dx, dy)‖` |

`edge_attr` uses unit-normalized directions and geometric context for the GNN encoder, while `edge_attr_dxdy` preserves raw displacements in physical coordinates for finite-difference-style gradient approximations in the continuity and momentum losses.

## How It Works

1. Loads `preprocess_airfrans_edges.py` at runtime via `importlib`.
2. For each graph, builds a radius graph using `global-radius` for all nodes and `surface-radius` for surface nodes.
3. Optionally adds surface ring edges along the airfoil boundary.
4. Checks node degrees; any node below `min-degree` gets KNN backup edges (up to `knn-backup-k` neighbors within `knn-max-radius`).
5. Ensures `edge_index` dtype is `torch.long`.
6. Saves the result using `orig_index` for filename alignment with the original dataset.

## Previous Step

Downsampled graphs must be prepared first:

```bash
python downsample_airfrans.py --root Dataset --task scarce --out-dir downsampled_graphs
```
