# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AirfRANS GNN is a physics-informed Graph Neural Network framework for aerodynamic surrogate modeling of 2D airfoil CFD. It predicts velocity (u, v), pressure (Cp), and turbulent viscosity (nu_t) over airfoil meshes using message-passing neural networks with RANS-based physics loss.

## Environment Setup

```bash
# Python 3.11 required
uv sync   # uses pyproject.toml with PyTorch CUDA 12.8 index and prebuilt PyG wheels
```

Key dependencies: PyTorch 2.8.0+cu128, PyTorch Geometric, torch-scatter, torch-sparse, wandb, scipy.
RTX 50 시리즈 (Blackwell, sm_120) 지원을 위해 CUDA 12.8 사용.

## 문서 참조 규칙 (Context7)

- **PyTorch**: Context7 library ID `/pytorch/pytorch` 사용. 버전 명시 시 해당 버전 문서 우선.
- **PyTorch Geometric**: Context7 library ID `/pyg-team/pytorch_geometric` 사용.

## Commands

### Data Preparation Pipeline

```bash
# Step 1: Downsample raw AirfRANS graphs (100k+ nodes → 15-30k)
python preprocessing/downsample_airfrans_v2.py --root Dataset --task scarce --out-dir Dataset/processed_data/downsampled-graphs

# Step 2: Build multi-scale radius-graph edges
python preprocessing/edges_from_downsampled_v2.py --in-dir Dataset/processed_data/downsampled-graphs --out-dir Dataset/processed_data/prebuilt_edges --task scarce
```

### Training

CLI scripts (preferred):
- `scripts/train.py` — Main training pipeline with full CLI args
- `scripts/run_experiment.py` — Train + benchmark score + experiment logging
- `scripts/train_multiscale.py` — Multi-scale model variant
- `scripts/optuna_hpo.py` — Hyperparameter optimization with Optuna
- `scripts/eval_cp.py` — Evaluate Cp relative L2 from a checkpoint
- `scripts/reset_experiment_docs.py` — Reset experiment tracking and Optuna docs

Integrated dashboard:
- `python dashboard/app.py` — Browser-based training dashboard (config, live charts, experiments, GPU monitor)

Jupyter notebooks (interactive exploration):
- `notebooks/00_preprocessing.ipynb` — Data preprocessing and inspection
- `notebooks/01_trainer.ipynb` — Main training pipeline
- `notebooks/02_optuna_training.ipynb` — Hyperparameter optimization with Optuna
- `notebooks/02_trainer_multi_scale.ipynb` — Multi-scale model variant
- `notebooks/03_eval_cp_relative_l2.ipynb` — Surface pressure evaluation

### Dashboard

```bash
python dashboard/app.py              # http://localhost:5000
python dashboard/app.py --port 8080  # custom port
```

### Tests

```bash
pytest tests/ -v
pytest tests/test_physics_loss_batching.py::test_physics_loss_with_increased_batch_size -v  # single test
```

## Architecture

### Pipeline Flow

Raw AirfRANS data → `preprocessing/downsample_airfrans_v2.py` (adaptive voxel sampling, preserves surface nodes) → `preprocessing/edges_from_downsampled_v2.py` (radius-graph edges with KNN backup) → Training via `scripts/train.py` or notebooks (normalize with `src/data.py`, train with `src/training.py` + physics loss, evaluate)

### Model: `EnhancedCFDModelWithGlobalContext` (defined in `src/model.py`)

- **Input**: 7D node features (freestream velocity, wall distance, wall normals, position) + 10D edge features (after enrichment)
- **Output**: 4D predictions (u, v, pressure, nu_t)
- **Architecture**: Node/Edge encoders → 14 message-passing layers → optional `GlobalContextProcessor` (attention-based) → output decoder
- Configuration lives in `Config` dataclass in `scripts/main.py` or CLI args in `scripts/train.py`

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
| `src/data.py` | `StandardScaler`, `NormalizedDataset`, `DataBundle`, `load_and_prepare_data`, `collate_pyg` |
| `src/model.py` | `EnhancedCFDModelWithGlobalContext` — attention-based global context model |
| `src/training.py` | `train_epoch`, `run_epoch`, `compute_loss_with_physics`, `train_with_scheduler`, LR scheduler, wandb init |
| `src/physics_loss.py` | `NavierStokesPhysicsLoss` — RANS physics loss with curriculum scheduling (continuity/momentum/bc start→target weight ramping) |
| `src/pipeline.py` | Declarative training API: `load_airfrans_data`, `convert_to_pyg`, `build_model`, `build_physics_loss`, `Trainer` (with `on_epoch_end` callback), `LiveDashboard` |
| `src/benchmark.py` | `ExperimentTracker`, FLOW-GLIDE comparison table, `score_test_set`, `run_benchmark_and_log_experiment` |
| `preprocessing/edges_from_downsampled_v2.py` | Edge construction wrapper (CLI) |
| `docs/benchmark/benchmark_reference.json` | FLOW-GLIDE 논문의 10개 baseline 메트릭 |

### Dashboard (`dashboard/`)

| File | Purpose |
|------|---------|
| `dashboard/app.py` | Flask server + API routes (`/api/config`, `/api/start`, `/api/stop`, `/api/status`, `/api/gpu`, `/api/experiments`) |
| `dashboard/runner.py` | `TrainingSession` — background thread로 학습 실행, thread-safe 상태 관리 |
| `dashboard/templates/index.html` | SPA frontend (Config/Training/Experiments 3탭 + GPU 모니터 플로팅 패널) |
| `docs/dashboard/README.md` | Dashboard 설계 문서 |

### Edge Feature Pipeline

엣지 피처는 오프라인 전처리 → 로드 타임 enrichment 두 단계로 생성된다.

**Stage 1 — 오프라인 전처리** (`src/edge_construction.py:build_edges_for_graph`)

`Dataset/processed_data/prebuilt_edges/`에 저장. 두 가지 텐서를 저장:

| 텐서 | 차원 | 피처 구성 |
|------|------|-----------|
| `edge_attr` | 5D | `[dist, dir_x, dir_y, cos_n, is_surface_pair]` |
| `edge_attr_dxdy` | 3D | `[dx, dy, dist]` (물리 손실 전용) |

`cos_n`: 양 끝 노드의 법선벡터 코사인 유사도 (`x[:, 3:5]` 사용).
`is_surface_pair`: 양 끝 모두 표면 노드면 1.

**Stage 2 — 로드 타임 enrichment** (`src/data.py:load_and_prepare_data`)

```
prep_graph() → _prep_graph_for_norm() → enrich_edge_features()
```

- `_prep_graph_for_norm`: 노드 x 5D → 7D (pos[:,:2] append)
- `enrich_edge_features`: edge_attr **5D → 10D** (edge_attr.size(1)==5 일 때만 실행)

추가되는 5개 피처:

| 인덱스 | 이름 | 계산 |
|--------|------|------|
| 5 | `log_dist` | `log(dist + 1e-8)` |
| 6 | `edge_angle` | `atan2(dy, dx) / π` ∈ [-1, 1] |
| 7 | `relative_sdf` | `(sdf[col] - sdf[row]) / (dist + 1e-8)` |
| 8 | `min_sdf` | `min(sdf[row], sdf[col])` |
| 9 | `has_boundary_node` | 양 끝 중 하나라도 표면 노드면 1 |

`sdf = x[:, 2]` (wall distance).

**최종 모델 입력**: 노드 7D (normalized) + 엣지 10D (raw) + `edge_attr_dxdy` 3D (physics loss)

상세 문서: `docs/preprocessing/edge_feature_pipeline.md`

### Batching

Uses PyG `Batch.from_data_list()` — multiple variable-size graphs concatenated into one with `batch` tensor for per-graph operations. `NormalizedDataset` in `src/data.py` handles feature normalization with `StandardScaler`.

## Data Artifacts

```
Dataset/
  raw_data/                                    — Raw AirfRANS download (AirfRANS.pt, manifest.json)
  raw/                                         — Symlink → raw_data/ (PyG compatibility)
  processed/                                   — PyG internal cache (auto-generated)
  processed_data/
    downsampled-graphs/<task>/{train,test}/    — Downsampled graphs (step 1 output)
    prebuilt_edges/<task>/{train,test}/        — Graphs with precomputed edges (step 2 output)
```

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
