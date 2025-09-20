# Downsampling and Edge Building Pipeline

This guide explains how to generate downsampled AirfRANS graphs and then attach edges to them for fast training and analysis. It covers:

- What each script does
- Expected input/output folder structure
- Recommended parameters and defaults
- Exact commands to run on Windows PowerShell
- Tips to ensure index alignment with the raw dataset

## Overview

There are two scripts used in sequence:

1) downsample_airfrans.py
   - Loads AirfRANS graphs (no edges) from the dataset
   - Applies adaptive voxel-based downsampling while preserving surface nodes
   - Saves per-graph .pt files with fields: x, y, pos (and optionally surf)
   - Adds a metadata field orig_index that records the original dataset index

2) build_edges_from_downsampled.py
   - Reads the downsampled graphs
   - Builds edge_index (and edge_attr) using preprocess_airfrans_edges.py
   - Saves per-graph .pt files under prebuilt_edges with the same index as the raw dataset (graph_{orig_index}.pt when available)

This preserves index alignment so that graph_000123.pt corresponds to the 123rd item of the raw AirfRANS dataset. The notebook loader can then retrieve graphs in the exact same order as the raw dataset split.

## Prerequisites

- Python 3.10/3.11
- PyTorch and PyTorch Geometric installed (in your conda environment)
- The AirfRANS dataset in the folder Dataset/ (default used by the scripts)
- Windows PowerShell is assumed below

If you use a conda environment, activate it first (replace the path with your env):

```powershell
# Example: activate your environment (adjust to your env name)
conda activate pyg5090
```

## 1) Downsample graphs (no edges)

This script saves downsampled graphs to downsampled_graphs/<task>/{train,test}/graph_XXXXXX.pt, embedding orig_index for alignment.

Key arguments:
- --root: Dataset root (default: Dataset)
- --task: scarce or full (default: scarce)
- --out-dir: Output root (default: downsampled_graphs)
- --limit-train / --limit-test: Optional caps on how many graphs to process
- --target-min-nodes / --target-max-nodes: Node count target band after downsampling
- --voxel-frac / --voxel-iters: Controls voxel size search

Run (Windows PowerShell):

```powershell
python .\downsample_airfrans.py --root Dataset --task scarce --out-dir downsampled_graphs \
  --target-min-nodes 15000 --target-max-nodes 30000 --voxel-frac 0.01 --voxel-iters 5
```

Output structure:

```
downsampled_graphs/
  scarce/
    train/
      graph_000000.pt
      graph_000001.pt
      ...
    test/
      graph_000000.pt
      graph_000001.pt
      ...
```

Each .pt contains: x, y, pos, optional surf, and orig_index (torch.long).

## 2) Build edges on downsampled graphs

This script reads downsampled graphs and saves graphs with edges under prebuilt_edges/<task>/{train,test}/. It preserves orig_index and names files graph_{orig_index}.pt when available to maintain exact index alignment.

Key arguments:
- --in-dir: Input root (the downsampled_graphs directory)
- --out-dir: Output root for prebuilt graphs with edges
- --task: scarce or full (default: scarce)
- --global-radius, --surface-radius: Radii for radius graph construction
- --max-num-neighbors: Cap on neighbors
- --surface-ring: Include ring edges along the surface (on by default)

Run (Windows PowerShell):

```powershell
python .\build_edges_from_downsampled.py `
  --in-dir .\downsampled_graphs `
  --out-dir .\prebuilt_edges `
  --task scarce `
  --global-radius 0.02 `
  --surface-radius 0.01 `
  --surface-ring `
  --max-num-neighbors 48
  --min-degree 2
  --knn-backup-k 4
  --knn-max-radius 0.05
```

Output structure:

```
prebuilt_edges/
  scarce/
    train/
      graph_000000.pt  # aligned to raw dataset idx 0
      graph_000001.pt  # aligned to raw dataset idx 1
      ...
    test/
      graph_000000.pt
      graph_000001.pt
      ...
```

## Alignment Guarantees and Notebook Loader

- The downsample script writes orig_index inside each file.
- The edge builder uses orig_index to rename outputs to graph_{orig_index}.pt.
- In the notebook, the prebuilt loader retrieves graphs by the exact raw indices (ids_train/ids_val), so train_edges[i] aligns with train_raw[i].
- If the files were created before this change and lack orig_index or correct filenames, you can either:
  - Rebuild the downsampled + edges with the commands above, or
  - Rely on the notebook's fallback: it scans files and tries to match by orig_index when filenames don’t match. Missing indices will raise a clear error.

## Parameter Tips

- global-radius ~ 0.02 and surface-radius ~ 0.01 are good starting points for 2D Airfoil meshes.
- max-num-neighbors 48 is a balanced default; reduce if you see memory pressure.
- For very large meshes, increase voxel-frac to downsample more aggressively, or raise target-max-nodes if VRAM allows.

## Troubleshooting

- ModuleNotFoundError: No module named 'torch': Make sure your conda env with PyTorch is activated before running scripts.
- FileNotFoundError: Missing graphs for indices: This indicates some indices weren’t produced or files are misnamed. Re-run the pipeline to regenerate files with orig_index and aligned filenames.
- Memory errors during edge building: Lower max-num-neighbors or increase radii slightly to reduce neighbor counts; you can also process in smaller limits using --limit-train/--limit-test in the downsampling step.

## Repro Checklist

1) Activate your PyTorch/PyG environment
2) Run downsample_airfrans.py to create downsampled_graphs with orig_index
3) Run build_edges_from_downsampled.py to create prebuilt_edges aligned by index
4) Point the notebook to prebuilt_edges and run the preprocessing cell
5) Visual compare should now show matching airfoil shapes for the same indices
