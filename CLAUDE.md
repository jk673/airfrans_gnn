# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AirfRANS GNN is a physics-informed Graph Neural Network framework for aerodynamic surrogate modeling of 2D airfoil CFD. It predicts velocity (u, v), pressure (Cp), and turbulent viscosity (nu_t) over airfoil meshes using message-passing neural networks with RANS-based physics loss.

## Environment Setup

```bash
# Python 3.11 required
uv sync   # uses pyproject.toml with PyTorch CUDA 12.4 index and prebuilt PyG wheels
```

Key dependencies: PyTorch, PyTorch Geometric, torch-scatter, torch-sparse, wandb, scipy.

## 문서 참조 규칙 (Context7)

- **PyTorch**: Context7 library ID `/pytorch/pytorch` 사용. 버전 명시 시 해당 버전 문서 우선.
- **PyTorch Geometric**: Context7 library ID `/pyg-team/pytorch_geometric` 사용.

## Commands

### Data Preparation Pipeline

```bash
# Step 1: Downsample raw AirfRANS graphs (100k+ nodes → 15-30k)
python preprocessing/downsample_airfrans_v2.py --root Dataset --task scarce --out-dir downsampled_graphs_v2

# Step 2: Build multi-scale radius-graph edges
python preprocessing/edges_from_downsampled_v2.py --in-dir downsampled_graphs_v2 --out-dir prebuilt_edges_v2 --task scarce
```

### Training

CLI scripts (preferred):
- `scripts/train.py` — Main training pipeline with full CLI args
- `scripts/run_experiment.py` — Train + benchmark score + experiment logging
- `scripts/train_multiscale.py` — Multi-scale model variant
- `scripts/optuna_hpo.py` — Hyperparameter optimization with Optuna
- `scripts/eval_cp.py` — Evaluate Cp relative L2 from a checkpoint
- `scripts/reset_experiment_docs.py` — Reset experiment tracking and Optuna docs

Jupyter notebooks (interactive exploration):
- `notebooks/00_preprocessing.ipynb` — Data preprocessing and inspection
- `notebooks/01_trainer.ipynb` — Main training pipeline
- `notebooks/02_optuna_training.ipynb` — Hyperparameter optimization with Optuna
- `notebooks/02_trainer_multi_scale.ipynb` — Multi-scale model variant
- `notebooks/03_eval_cp_relative_l2.ipynb` — Surface pressure evaluation

### Tests

```bash
pytest tests/ -v
pytest tests/test_physics_loss_batching.py::test_physics_loss_with_increased_batch_size -v  # single test
```

## Architecture

### Pipeline Flow

Raw AirfRANS data → `preprocessing/downsample_airfrans_v2.py` (adaptive voxel sampling, preserves surface nodes) → `preprocessing/edges_from_downsampled_v2.py` (radius-graph edges with KNN backup) → Training via `scripts/train.py` or notebooks (normalize with `src/data.py`, train with `src/training.py` + physics loss, evaluate)

### Model: `EnhancedCFDModelWithGlobalContext` (defined in `src/global_context_processor.py`)

- **Input**: 7D node features (freestream velocity, wall distance, wall normals, position) + 10D edge features (after enrichment)
- **Output**: 4D predictions (u, v, pressure, nu_t)
- **Architecture**: Node/Edge encoders → 14 message-passing layers → optional `GlobalContextProcessor` (attention-based) → output decoder
- Configuration lives in `SmokeCfg` dataclass in `src/config.py`

### Physics-Informed Loss (`src/physics_loss.py`)

Combined loss with curriculum learning:
- **Data loss**: MSE on predictions
- **Continuity loss**: Conservative divergence (∇·u = 0)
- **Momentum loss**: RANS momentum balance with skew-symmetric convection
- **BC loss**: No-slip walls, inlet/outlet/farfield conditions

Physics weights ramp up over training via linear or cosine curriculum schedule. `src/preprocessing.py:prepare_airfrans_graph_for_physics()` precomputes node areas, boundary masks, wall normals, and inlet velocities needed by the physics loss.

### Key Module Responsibilities

| File | Purpose |
|------|---------|
| `src/config.py` | `SmokeCfg` dataclass, config file loading, CLI parsing (`parse_args`, `create_config_from_args`) |
| `src/data.py` | `StandardScaler`, `NormalizedDataset`, `DataBundle`, `load_and_prepare_data`, `collate_pyg` |
| `src/training.py` | `train_epoch`, `run_epoch`, `compute_loss_with_physics`, `train_with_scheduler`, LR scheduler, wandb init |
| `src/preprocessing.py` | Physics preprocessing (BC masks, node areas, wall normals, edge geometry) |
| `src/physics_loss.py` | `NavierStokesPhysicsLoss` — RANS physics loss with curriculum scheduling |
| `src/turbulent_physics_loss.py` | `EnhancedPhysicsLoss` — turbulence model extensions |
| `src/global_context_processor.py` | `EnhancedCFDModelWithGlobalContext` — attention-based global context model |
| `src/multigraph_convolution.py` | Multi-scale and dilated graph convolutions |
| `src/force_coefficients.py` | Lift/drag coefficient integration from surface pressure |
| `src/utils.py` | Graph utilities (`prep_graph`, `validate_edges`, `_valid_edges`, `_prep_graph_for_norm`) |
| `src/benchmark.py` | `ExperimentTracker`, FLOW-GLIDE comparison table generation |
| `src/diagnostics.py` | Diagnostic plots and statistics (`plot_inlet_bc_velocity`, `print_diagnostic_stats`) |
| `src/metrics.py` | Surface mask detection, force coefficient computation |
| `src/prediction.py` | Model inference helpers |
| `src/visualization.py` | Prediction vs. ground-truth plotting |
| `src/edge_construction.py` | Radius-graph edge building, degree floor enforcement, edge features, QA reports |
| `src/ddp_utils.py` | Distributed Data Parallel helpers (`setup_ddp`, `_is_ddp`, `_unwrap_model`, etc.) |
| `preprocessing/edges_from_downsampled_v2.py` | Edge construction wrapper (CLI) |
| `docs/benchmark/benchmark_reference.json` | FLOW-GLIDE 논문의 10개 baseline 메트릭 |
| `scripts/score_benchmark.py` | CLI 벤치마크 스코어링 (6개 메트릭 계산 + 비교 테이블) |

### Edge Attribute Schema

Base 5D edge features: `[dist, dir_x, dir_y, cos_n, is_surface_pair]`. Two raw orderings are auto-detected:
- `[dist, dir_x, dir_y]` (default) or `[dx, dy, dist]` (dxdy format)
- Detection heuristic in `src/edge_construction.py` based on column value ranges

After `_prep_graph_for_norm`, `enrich_edge_features()` in `src/data.py` extends 5D → 10D by appending: `log_dist, edge_angle, relative_sdf, min_sdf, has_boundary_node`.

### Batching

Uses PyG `Batch.from_data_list()` — multiple variable-size graphs concatenated into one with `batch` tensor for per-graph operations. `NormalizedDataset` in `src/data.py` handles feature normalization with `StandardScaler`.

## Data Artifacts

- `Dataset/` — Raw AirfRANS dataset
- `downsampled_graphs_v2/<task>/{train,test}/graph_*.pt` — Downsampled graphs
- `prebuilt_edges_v2/<task>/{train,test}/graph_*.pt` — Graphs with precomputed edges

Prebuilt graphs are aligned to the original dataset via `orig_index` field.

## Benchmark Scoring

훈련 완료 후 FLOW-GLIDE 기준 6개 메트릭을 계산하고 비교 테이블을 생성한다.

### 실행 방법

```bash
# 기본 (best checkpoint, scarce task, hidden=128, layers=14)
python scripts/score_benchmark.py \
  --checkpoint checkpoints/best.pt \
  --task scarce \
  --hidden 128 --layers 14 \
  --model-name "Ours"

# 커스텀 설정
python scripts/score_benchmark.py \
  --checkpoint checkpoints/best.pt \
  --task full \
  --hidden 256 --layers 16 \
  --model-name "PhysicsGNN v3"
```

### 메트릭 정의 (FLOW-GLIDE, Su et al. 2025)

| Metric | Description | Direction |
|--------|-------------|-----------|
| Volume Rel. L₂ | Relative L2 over all nodes, 4 channels (ux, uy, p, nu_t) | ↓ |
| Surface Rel. L₂ | Relative L2 over surface nodes, pressure only | ↓ |
| CD Rel. Error | Mean \|CD_pred − CD_gt\| / \|CD_gt\| | ↓ |
| CL Rel. Error | Mean \|CL_pred − CL_gt\| / \|CL_gt\| | ↓ |
| ρ_D (Spearman) | Rank correlation of drag coefficient | ↑ |
| ρ_L (Spearman) | Rank correlation of lift coefficient | ↑ |

### 출력 파일 (score_benchmark.py 실행 시 생성)

- `docs/benchmark/results.json` — 메트릭 값 + 메타데이터
- `docs/benchmark/results.md` — FLOW-GLIDE 비교 마크다운 테이블

### 참고

- `docs/benchmark/benchmark_reference.json` — FLOW-GLIDE 논문의 10개 baseline 메트릭
- `docs/benchmark/scoring_guide.md` — 상세 워크플로우 가이드
- `--hidden`/`--layers`는 체크포인트 훈련 시 사용한 값과 반드시 일치해야 함

## Notebook 수정 규칙

- `.ipynb` 파일 수정 시 **NotebookEdit 도구를 사용하지 않는다** (디스크 반영 안 되는 버그 있음).
- 대신 **Python으로 JSON을 직접 읽고 수정 후 `json.dump`로 저장**한다.
- source 필드는 줄 단위 리스트이며, 마지막 줄을 제외한 각 줄 끝에 `\n`을 붙인다.
- 수정 후 반드시 파일을 다시 읽어 변경 사항이 디스크에 반영되었는지 검증한다.
