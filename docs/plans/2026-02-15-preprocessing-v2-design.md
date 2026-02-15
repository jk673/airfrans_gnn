# Preprocessing V2 Design

## Context
Based on the survey (`docs/preprocessing/survey_2026-02-15.md`), the current preprocessing has 4 key issues.

## Changes

### 1. Degree calculation fix (High)
- **Current**: `bincount(row) + bincount(col)` on bidirectional edges → ~2x inflation
- **V2**: `bincount(row)` only (edges are already bidirectional, so row count = undirected degree)
- **Files**: `src/preprocess_airfrans_edges_v2.py`

### 2. Voxel representative: gradient-based selection (Medium)
- **Current**: First-hit (arbitrary first node in voxel)
- **V2**: Node with highest y-deviation from voxel mean (proxy for flow gradient importance)
- **Fallback**: Centroid-nearest if y is unavailable
- **Files**: `preprocessing/downsample_airfrans_v2.py`

### 3. orig_index robustness (Medium)
- **Current**: Loop index `i` → breaks with shuffled/custom subsets
- **V2**: Use `subset.indices[i]` when dataset is a `Subset`
- **Files**: `preprocessing/downsample_airfrans_v2.py`

### 4. Per-graph QA report (New)
- Metrics: isolated count, low-degree count, degree stats, edge length stats
- Fail-fast thresholds in config (max_isolated_fraction, max_low_degree_fraction)
- **Files**: `src/preprocess_airfrans_edges_v2.py`, `preprocessing/config_v2.py`

## File mapping
| Original | V2 |
|----------|----|
| `preprocessing/downsample_airfrans.py` | `preprocessing/downsample_airfrans_v2.py` |
| `src/preprocess_airfrans_edges.py` | `src/preprocess_airfrans_edges_v2.py` |
| `preprocessing/config.py` | `preprocessing/config_v2.py` |
| `preprocessing/edges_from_downsampled.py` | `preprocessing/edges_from_downsampled_v2.py` |
