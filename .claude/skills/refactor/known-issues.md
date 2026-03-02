# Known Refactoring Targets

## Code Duplication (CRITICAL)

### `_extract_dxdy_length()` (~70 lines, duplicated)
- `src/airfrans_utils.py`
- `src/navier_stokes_physics_loss.py`
- **Action**: Extract to `src/edge_utils.py`, import from both

### `_half_edges()` (~8 lines, duplicated)
- `src/airfrans_utils.py`
- `src/navier_stokes_physics_loss.py`
- **Action**: Move to `src/edge_utils.py`

### `_valid_edges()` (~4 lines, duplicated)
- `src/airfrans_utils.py`
- `src/navier_stokes_physics_loss.py`
- **Action**: Move to `src/edge_utils.py`

## Bare Exception Handling

### `src/navier_stokes_physics_loss.py`
- Multiple `except Exception:` with silent pass
- 100+ line try-except blocks
- **Action**: Use specific exception types, narrow scope

### `src/utils.py`
- `_prep_graph_for_norm()` catches all exceptions silently
- **Action**: Catch specific errors, add logging

## Large Config Object

### `SmokeCfg` in `src/training_common.py`
- 80+ fields in single dataclass
- **Action**: Decompose into `ModelConfig`, `PhysicsConfig`, `TrainingConfig`, `DataConfig`

## Magic Numbers

| Value | File | Purpose |
|-------|------|---------|
| `1.5` | `src/airfrans_utils.py` | Edge schema detection |
| `0.02` | `src/airfrans_utils.py` | Inlet quantile |
| `0.90` | `src/airfrans_utils.py` | Farfield quantile |
| `1e-4` | `src/airfrans_utils.py` | Wall distance threshold |
| `0.12` | `src/preprocess_airfrans_edges.py` | Edge length hard cap |

## Incomplete Implementations

| Module | Issue |
|--------|-------|
| `src/multigraph_convolution.py` | `SpatialPyramidPooling._adaptive_pool()` is just global pooling |
| `src/global_context_processor.py` | `Set2Set` has hardcoded 3 LSTM steps |
| `src/turbulent_modeling_physics_loss.py` | Duplicate imports at top |
