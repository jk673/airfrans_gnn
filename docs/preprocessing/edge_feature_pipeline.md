# 엣지 피처 파이프라인

AirfRANS GNN에서 엣지 피처가 생성되고 최종 훈련에 사용되기까지의 전체 흐름을 설명한다.

---

## 전체 흐름 요약

```
[prebuilt_edges_v2/*.pt]
        │
        │  edge_attr:     [dist, dir_x, dir_y, cos_n, is_surf_pair]  (5D)
        │  edge_attr_dxdy: [dx, dy, dist]                            (3D)
        │
        ▼  prep_graph()          → 5D edge_attr 검증/확인
        ▼  _prep_graph_for_norm() → x: 5D → 7D  (pos2 append)
        ▼  enrich_edge_features() → edge_attr: 5D → 10D
        │
        ▼  StandardScaler fit on x (7D), y (4D)
        ▼  NormalizedDataset → x, y 정규화
        │
[모델 입력]
  node_feat:      7D  (normalized)
  edge_feat:      10D (raw, not normalized)
  edge_attr_dxdy: 3D  (physics loss 전용)
```

---

## Stage 1: 오프라인 전처리

**실행 스크립트**: `preprocessing/edges_from_downsampled_v2.py`
**핵심 함수**: `src/edge_construction.py:build_edges_for_graph()`
**출력**: `prebuilt_edges_v2/<task>/{train,test}/graph_*.pt`

### 엣지 연결 구조 생성 순서

1. **글로벌 반경 그래프**
   `radius_graph(pos2, r=global_radius, max_num_neighbors=64)`
   기본값 `global_radius=0.02` (chord 길이 기준 2%)

2. **표면 전용 반경 그래프**
   `surf` 마스크로 표면 노드만 추출 후 `radius_graph(pos_surf, r=surface_radius)`
   기본값 `surface_radius=0.01` — 표면 구조를 더 촘촘하게 연결

3. **양방향화 + 중복 제거**
   `edge.flip(0)`으로 역방향 엣지 추가 → `torch.unique(dim=1)`

4. **KNN 백업**
   degree < `min_degree(=2)` 노드에 가장 가까운 이웃 엣지 추가
   반경 상한 `knn_max_radius=0.05`, 길이 상한 `length_hard_cap=0.12`

5. **Degree 캡 pruning**
   degree > `final_max_degree(=64)` 노드에서 가장 긴 엣지부터 제거

6. **사후 최소 degree 보정**
   `_postfix_repair_min_degree()` — 최대 3회 반복하여 최소 degree 보장

### 저장되는 엣지 피처

| 텐서 | 차원 | 피처 구성 | 용도 |
|------|------|-----------|------|
| `edge_attr` | 5D | `[dist, dir_x, dir_y, cos_n, is_surface_pair]` | 모델 입력 (5→10D로 확장) |
| `edge_attr_dxdy` | 3D | `[dx, dy, dist]` | 물리 손실 전용 |

**`edge_attr` 피처 상세**

| 인덱스 | 이름 | 계산 |
|--------|------|------|
| 0 | `dist` | `‖pos[col] - pos[row]‖` |
| 1 | `dir_x` | `(pos[col,0] - pos[row,0]) / dist` |
| 2 | `dir_y` | `(pos[col,1] - pos[row,1]) / dist` |
| 3 | `cos_n` | `dot(n_row, n_col) / (‖n_row‖·‖n_col‖)` — 법선벡터 코사인 유사도 (`x[:, 3:5]` 사용) |
| 4 | `is_surface_pair` | 양 끝 노드가 모두 표면 노드이면 1, 아니면 0 |

---

## Stage 2: 훈련 데이터 로드

**핵심 함수**: `src/data.py:load_and_prepare_data()`

```python
train_prepped = [enrich_edge_features(_prep_graph_for_norm(g)) for g in train_edges_subset]
```

### Step A: `_prep_graph_for_norm(g)` — `src/utils.py:101`

노드 피처 `x`가 5D(`[u_inf, v_inf, sdf, nx, ny]`)이면 `pos[:, :2]`(x, y 좌표)를 append하여 7D로 확장.

```
x: [u_inf, v_inf, sdf, nx, ny]  (5D)
         ↓ with_pos2()
x: [u_inf, v_inf, sdf, nx, ny, px, py]  (7D)
```

- `edge_attr`가 5D가 아니면 `ensure_edge_features(d, want_dim=5)`로 재계산 (prebuilt 사용 시 해당 없음)

### Step B: `enrich_edge_features(data)` — `src/data.py:124`

`edge_attr.size(1) == 5`인 경우에만 실행. 5개 피처를 추가하여 **10D**로 확장.

| 인덱스 | 이름 | 계산 | 의미 |
|--------|------|------|------|
| 0–4 | (기존 5D) | — | 위 Stage 1 참조 |
| **5** | `log_dist` | `log(dist + 1e-8)` | 로그 스케일 거리 — 넓은 거리 범위를 부드럽게 인코딩 |
| **6** | `edge_angle` | `atan2(dy, dx) / π` ∈ [-1, 1] | 엣지 방향 각도 (정규화) |
| **7** | `relative_sdf` | `(sdf[col] - sdf[row]) / (dist + 1e-8)` | 벽면 거리(SDF) 기울기 |
| **8** | `min_sdf` | `min(sdf[row], sdf[col])` | 엣지 경계 근접도 |
| **9** | `has_boundary_node` | 양 끝 중 하나라도 표면 노드면 1 | 경계 인접 엣지 식별 |

`sdf`는 `x[:, 2]` — wall distance (signed distance function).

---

## Stage 3: 최종 모델 입력

**관련 코드**: `scripts/train.py:127–142`

```python
node_dim = 7
edge_dim = data_bundle.edge_dim  # = 10
model = EnhancedCFDModelWithGlobalContext(
    node_feat_dim=node_dim,   # 7
    edge_feat_dim=edge_dim,   # 10
    ...
)
```

### 모델에 입력되는 피처

| 피처 텐서 | 차원 | 정규화 | 사용처 |
|-----------|------|--------|--------|
| `x` | 7D | StandardScaler 적용 | 노드 인코더 입력 |
| `edge_attr` | 10D | 적용 안 함 | 엣지 인코더 입력 |
| `edge_attr_dxdy` | 3D | 적용 안 함 | `NavierStokesPhysicsLoss` (발산·모멘텀 계산) |

### 노드 피처 7D 상세

| 인덱스 | 이름 | 설명 |
|--------|------|------|
| 0 | `u_inf` | 자유류 x방향 속도 |
| 1 | `v_inf` | 자유류 y방향 속도 |
| 2 | `sdf` | 벽면 거리 (wall distance) |
| 3 | `nx` | 벽면 법선벡터 x 성분 |
| 4 | `ny` | 벽면 법선벡터 y 성분 |
| 5 | `px` | 노드 x 좌표 |
| 6 | `py` | 노드 y 좌표 |

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `src/edge_construction.py` | 엣지 구조 생성 + 5D/3D `edge_attr` 저장 |
| `src/utils.py` | `prep_graph`, `_prep_graph_for_norm`, `ensure_edge_features` |
| `src/data.py` | `enrich_edge_features` (5D→10D), `load_and_prepare_data`, `NormalizedDataset` |
| `preprocessing/edges_from_downsampled_v2.py` | Stage 1 CLI 래퍼 |
| `prebuilt_edges_v2/` | 전처리된 엣지 파일 저장 위치 |
