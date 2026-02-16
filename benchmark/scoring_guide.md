# Benchmark Scoring Guide

## 에이전트에게 줄 프롬프트 (복사해서 사용)

### 기본형 (훈련 직후)

```
훈련이 끝났습니다. CLAUDE.md의 "Benchmark Scoring" 섹션을 읽고,
best checkpoint로 벤치마크 스코어링을 실행해주세요.

결과를 FLOW-GLIDE 비교 테이블 형식으로 정리하고,
Transolver와 FLOW-GLIDE 대비 개선/열화 여부를 분석해주세요.
```

### 상세형 (체크포인트 경로 지정)

```
훈련이 끝났습니다. 다음 조건으로 벤치마크 스코어링을 실행해주세요:

- 체크포인트: checkpoints/best_epoch120.pt
- Task: full
- 모델 이름: "PhysicsGNN v3"
- Hidden: 256, Layers: 16

CLAUDE.md의 "Benchmark Scoring" 섹션에 따라
scripts/score_benchmark.py를 실행하고, 결과를
FLOW-GLIDE 논문 비교 테이블 형식으로 정리해주세요.
benchmark_results.json과 benchmark_results.md도 저장해주세요.
```

### 자동 파이프라인형 (훈련→스코어링 연속)

```
다음 순서로 실행해주세요:

1. scripts/train.py로 full task 훈련 실행
2. 훈련 완료 후, 저장된 best checkpoint 경로를 확인
3. CLAUDE.md "Benchmark Scoring" 섹션에 따라
   scripts/score_benchmark.py를 실행
4. 결과를 FLOW-GLIDE 비교 테이블로 정리
5. benchmark_results.json, benchmark_results.md 저장
```

---

## 에이전트가 실행하는 워크플로우

에이전트가 위 프롬프트를 받으면 CLAUDE.md를 읽고 다음을 수행합니다:

### Step 1: 체크포인트 확인
```bash
ls -la checkpoints/
# → best.pt, last.pt 등 확인
```

### Step 2: 훈련 설정 확인
```bash
# checkpoint에서 config 추출 (가능한 경우)
python -c "
import torch
ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
if isinstance(ckpt, dict):
    for k in ['task', 'hidden', 'layers', 'epoch', 'val_loss']:
        print(f'{k}: {ckpt.get(k, \"N/A\")}')
"
```

### Step 3: 스코어링 실행
```bash
python scripts/score_benchmark.py \
  --checkpoint checkpoints/best.pt \
  --task full \
  --hidden 128 --layers 14 \
  --model-name "Ours"
```

### Step 4: 결과 확인
```bash
cat benchmark_results.json
cat benchmark_results.md
```

### Step 5: 비교 분석 제시

에이전트는 결과를 다음 형식으로 보고합니다:

```
## 벤치마크 결과

| Model | Volume Rel.L₂ ↓ | Surface Rel.L₂ ↓ | CD Rel.Err ↓ | CL Rel.Err ↓ | ρ_D ↑ | ρ_L ↑ |
|---|---|---|---|---|---|---|
| Transolver | 0.0100 | 0.0352 | 0.6316 | 0.1122 | 0.8750 | 0.9946 |
| FLOW-GLIDE | 0.0038 | 0.0063 | 0.5072 | 0.1029 | 0.9286 | 0.9964 |
| **Ours** | **0.XXXX** | **0.XXXX** | **X.XXXX** | **X.XXXX** | **X.XXXX** | **X.XXXX** |

### 분석
- Volume Rel. L₂: Transolver 대비 XX% 개선/열화
- Surface Rel. L₂: FLOW-GLIDE 대비 XX% 개선/열화
- ρ_D: 드래그 순위 예측 ...
```

---

## 핵심 파일 구조

```
airfrans_gnn/
├── CLAUDE.md                    # ← 에이전트가 읽는 주 문서 (Benchmark Scoring 섹션)
├── benchmark/
│   ├── benchmark_reference.json # ← FLOW-GLIDE 비교 테이블 (10개 모델)
│   ├── scoring_guide.md         # ← 이 파일 (상세 워크플로우 가이드)
│   ├── results.json             # ← 스코어링 결과 (실행 후 생성)
│   └── results.md               # ← 마크다운 비교 테이블 (실행 후 생성)
├── scripts/
│   └── score_benchmark.py       # ← CLI 벤치마크 스코어링
└── src/
    ├── metrics.py               # ← relative_l2, compute_force_coefficients 등
    └── force_coefficients_calculation.py
```

## 메트릭 정의 상세

### Volume Rel. L₂ (FLOW-GLIDE 방식)

```python
# 전체 테스트셋의 모든 노드, 4채널을 하나로 합쳐서 계산
vol_rel_l2 = sqrt(Σ_graphs Σ_nodes Σ_channels (pred - gt)²)
           / sqrt(Σ_graphs Σ_nodes Σ_channels gt²)
```

주의: per-graph 평균이 아니라, 모든 그래프의 노드를 하나의 풀로 합쳐서
global relative L2를 계산합니다. 이것이 FLOW-GLIDE 논문의 정의입니다.

### Surface Rel. L₂

```python
# 표면 노드의 압력(p/ρ) 채널만 사용
surf_rel_l2 = sqrt(Σ_graphs Σ_surf_nodes (p_pred - p_gt)²)
            / sqrt(Σ_graphs Σ_surf_nodes p_gt²)
```

### Force Coefficients

```python
# Per-graph: 표면 압력 적분 → CD, CL 계산
# CD relative error = mean_graphs(|CD_pred - CD_gt| / |CD_gt|)
# CL relative error = mean_graphs(|CL_pred - CL_gt| / |CL_gt|)
# ρ_D = spearmanr(CD_gt_all, CD_pred_all)
# ρ_L = spearmanr(CL_gt_all, CL_pred_all)
```

## 주의사항

1. **메트릭 혼동 방지**: AirfRANS 원본 논문(NeurIPS 2022)은 정규화 필드의 MSE를 사용.
   FLOW-GLIDE는 물리 공간의 Relative L₂를 사용. 직접 비교 불가.

2. **`--hidden`/`--layers` 불일치**: 체크포인트와 CLI 인자가 다르면 `state_dict` 로드 실패.
   훈련 시 사용한 정확한 값을 전달해야 함.

3. **Task 일치**: `--task`는 `prebuilt_edges/<task>/test/`의 데이터를 로드함.
   훈련 task와 반드시 일치해야 함.

4. **배치 사이즈**: `--batch-size 1` 권장. >1이면 force coefficient 계산 시
   batch 내 그래프 분리가 필요하며, 스크립트가 이를 처리함.
