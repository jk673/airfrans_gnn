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

## PyTorch 문서 참조 규칙 (Context7)

- PyTorch 관련 질문(API 사용법, 설치, 설정, 예제 코드, 버전별 차이, deprecated 여부 등)에는 **항상 Context7 MCP를 사용**한다.
- 기본은 **PyTorch 최신 stable 문서** 기준으로 답변한다.
- Context7 library ID는 `/pytorch/pytorch`로 고정하여 사용한다.
- 버전이 명시된 경우(예: PyTorch 2.1, 2.2, 2.5)에는 해당 버전 문서를 우선 참조한다.
- PyTorch 관련 코드 생성, 디버깅, 성능 최적화 질문에도 동일하게 적용한다.

**프롬프트 예시:**
```python
# torch.nn.functional.grid_sample의 align_corners 파라미터가 무엇인지 설명해줘
# PyTorch 2.5에서 torch.compile 사용 예제를 보여줘
# DataLoader의 num_workers 설정 시 권장사항은?
```

## PyTorch Geometric 문서 참조 규칙 (Context7)

- PyTorch Geometric 관련 질문(GNN 레이어, 데이터 구조, 메시지 패싱, 배치 처리, 변환 등)에는 **항상 Context7 MCP를 사용**한다.
- Context7 library ID는 `/pyg-team/pytorch_geometric`로 고정하여 사용한다.
- PyG의 `Data`, `Batch`, `MessagePassing`, 각종 Conv 레이어(GCNConv, GATConv 등) 사용법 질문에 적용한다.
- 그래프 전처리, 에지 구성, 노드/에지 피처 처리 관련 질문에도 동일하게 적용한다.

**프롬프트 예시:**
```python
# MessagePassing 클래스의 propagate 메서드는 어떻게 동작하나?
# torch_geometric.nn.GATConv의 heads 파라미터 설명
# Batch.from_data_list()로 여러 그래프 배치 처리하는 방법
# radius_graph와 knn_graph의 차이점은?
```

## Commands

### Data Preparation Pipeline

```bash
# Step 1: Downsample raw AirfRANS graphs (100k+ nodes → 15-30k)
python preprocessing/downsample_airfrans.py --root Dataset --task scarce --out-dir downsampled_graphs

# Step 2: Build multi-scale radius-graph edges
python preprocessing/build_edges_from_downsampled.py --in-dir downsampled_graphs --out-dir prebuilt_edges --task scarce
```

### Training

Training runs through Jupyter notebooks:
- `01_trainer.ipynb` — Main training pipeline
- `02_optuna_training.ipynb` — Hyperparameter optimization with Optuna
- `02_trainer_multi_scale.ipynb` — Multi-scale model variant

### Tests

```bash
pytest tests/ -v
pytest tests/test_continuity_loss.py::test_continuity_zero_divergence_gt_only -v  # single test
```

## Architecture

### Pipeline Flow

Raw AirfRANS data → `preprocessing/downsample_airfrans.py` (adaptive voxel sampling, preserves surface nodes) → `preprocessing/build_edges_from_downsampled.py` (radius-graph edges with KNN backup) → Training notebook (normalize, train with physics loss, evaluate)

### Model: `EnhancedCFDModelWithGlobalContext` (defined in `01_trainer.ipynb`)

- **Input**: 7D node features (freestream velocity, wall distance, wall normals, position) + 5D edge features
- **Output**: 4D predictions (u, v, pressure, nu_t)
- **Architecture**: Node/Edge encoders → 14 message-passing layers → optional `GlobalContextProcessor` (attention-based) → output decoder
- Configuration lives in `SmokeCfg` dataclass inside `01_trainer.ipynb`

### Physics-Informed Loss (`src/navier_stokes_physics_loss.py`)

Combined loss with curriculum learning:
- **Data loss**: MSE on predictions
- **Continuity loss**: Conservative divergence (∇·u = 0)
- **Momentum loss**: RANS momentum balance with skew-symmetric convection
- **BC loss**: No-slip walls, inlet/outlet/farfield conditions

Physics weights ramp up over training via linear or cosine curriculum schedule. `src/airfrans_utils.py:prepare_airfrans_graph_for_physics()` precomputes node areas, boundary masks, wall normals, and inlet velocities needed by the physics loss.

### Key Module Responsibilities

| File | Purpose |
|------|---------|
| `preprocessing/downsample_airfrans.py` | Adaptive voxel downsampling with surface preservation |
| `preprocessing/build_edges_from_downsampled.py` | Edge construction wrapper |
| `src/preprocess_airfrans_edges.py` | Radius-graph edge building, degree floor enforcement, edge feature computation |
| `src/airfrans_utils.py` | Physics preprocessing (areas, BC masks, wall normals) |
| `src/navier_stokes_physics_loss.py` | RANS physics loss with curriculum scheduling |
| `src/turbulent_modeling_physics_loss.py` | Turbulence model extensions |
| `src/global_context_processor.py` | Attention-based global context with cross-attention and Set2Set pooling |
| `src/multigraph_convolution.py` | Multi-scale and dilated graph convolutions |
| `src/force_coefficients_calculation.py` | Lift/drag coefficient integration from surface pressure |
| `src/utils.py` / `src/utils_prune.py` | Graph utilities and isolated node pruning |

### Edge Attribute Schema

The codebase supports two edge feature orderings detected automatically:
- `[dist, dir_x, dir_y]` (default)
- `[dx, dy, dist]` (dxdy format)

Detection heuristic is in `src/preprocess_airfrans_edges.py` based on column value ranges.

### Batching

Uses PyG `Batch.from_data_list()` — multiple variable-size graphs concatenated into one with `batch` tensor for per-graph operations. The `NormalizedDataset` class in the training notebook handles feature normalization with StandardScaler.

## Data Artifacts

- `Dataset/` — Raw AirfRANS dataset
- `downsampled_graphs/<task>/{train,test}/graph_*.pt` — Downsampled graphs
- `prebuilt_edges/<task>/{train,test}/graph_*.pt` — Graphs with precomputed edges

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

### 출력 파일

- `benchmark_results.json` — 메트릭 값 + 메타데이터
- `benchmark_results.md` — FLOW-GLIDE 비교 마크다운 테이블

### 참고

- `benchmark_reference.json` — FLOW-GLIDE 논문의 10개 baseline 메트릭
- `docs/benchmark/benchmark_scoring.md` — 상세 워크플로우 가이드
- `--hidden`/`--layers`는 체크포인트 훈련 시 사용한 값과 반드시 일치해야 함
