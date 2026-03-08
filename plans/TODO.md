# AirfRANS GNN — 개발 TODO 리스트

> 작성일: 2026-02-16
> 대상 레포: `airfrans_gnn`

---

## 목차

1. [Surface Cp Visualization 버그 수정](#1-surface-cp-visualization-버그-수정)
2. [BC Loss 업데이트](#2-bc-loss-업데이트)
3. [Continuity Loss 재검토](#3-continuity-loss-재검토)
4. [Momentum Loss 재검토](#4-momentum-loss-재검토)
5. [Normalization / Denormalization 문서화](#5-normalization--denormalization-문서화)
6. [Attention Mechanism 조사 및 반영](#6-attention-mechanism-조사-및-반영)
7. [기타 추천 사항](#7-기타-추천-사항)

---

## 1. Surface Cp Visualization 버그 수정

### 증상
`01_trainer.ipynb` → Results Visualization 섹션에서 surface Cp distribution 그래프의 prediction 곡선이 **flat** (거의 일정한 값)으로 나옴. Ground truth는 정상적인 Cp 분포를 보여줌.

### 관련 파일
- `src/visualization.py` → `plot_surface_pressure()`, `plot_pred_vs_gt()`
- `01_trainer.ipynb` → 마지막 Results visualization 셀

### 근본 원인 분석 (3가지 가능성)

#### 가능성 A: Denormalization 누락 또는 이중 적용
`plot_surface_pressure()`는 **이미 denormalized된** `y_pred_denorm`과 `y_true_denorm`을 인자로 받는다. 만약 노트북에서 호출할 때:
- 모델 output (normalized)을 그대로 넘기면 → prediction이 표준화된 좁은 범위에 몰려 flat하게 보임
- 이미 denormalize된 값을 또 denormalize하면 → 스케일이 왜곡됨

**확인 방법**: 노트북 셀에서 `y_pred`의 실제 값 범위를 출력:
```python
print(f"y_pred raw: min={y_pred.min():.4f}, max={y_pred.max():.4f}")
print(f"y_pred denorm: min={y_pred_denorm.min():.4f}, max={y_pred_denorm.max():.4f}")
print(f"y_true denorm: min={y_true_denorm.min():.4f}, max={y_true_denorm.max():.4f}")
```

#### 가능성 B: Surface mask가 normalized features에서 생성됨
`plot_surface_pressure()` 내부에서 surface mask를 `d.x[:, 3:5]`로 만든다:
```python
normals = xvars[:, 3:5]
surf_mask = (normals.abs().sum(dim=1) > 0)
```
만약 `d.x`가 **normalized 상태**라면 모든 노드의 normal 컬럼이 non-zero가 되어 **모든 노드가 surface로 분류**된다. 이 경우 volume 노드까지 포함되어 pressure 분포가 뭉개짐.

**확인 방법**:
```python
surf_mask = (d.x[:, 3:5].abs().sum(dim=1) > 0)
print(f"Surface nodes: {surf_mask.sum()} / {d.num_nodes} total")
# 만약 surf_mask.sum() ≈ d.num_nodes 이면 이 문제
```

#### 가능성 C: `plot_pred_vs_gt()`의 Cp 계산에서 q 값 오류
`plot_pred_vs_gt()`는 Cp = p / q 를 계산하는데, q = 0.5 * V²를 `x_phys[0, 0:2]`에서 구한다. 만약 `x_scaler`가 None으로 전달되면 normalized features에서 q를 계산하여 비물리적인 값이 됨.

### 수정 방안

```python
# 01_trainer.ipynb 셀 수정안 (핵심 로직)

# 1. 모델 prediction (normalized space)
with torch.no_grad():
    y_pred_norm = model(dm_norm)

# 2. Denormalize
y_pred_denorm = y_scaler.inverse(y_pred_norm.cpu())
y_true_denorm = y_scaler.inverse(dm_norm.y.cpu())

# 3. Surface mask는 RAW features에서 추출
d_raw = graphs_test[idx]  # 원본 그래프 (정규화 전)
x_raw = d_raw.x
normals_raw = x_raw[:, 3:5]
surf_mask = (normals_raw.abs().sum(dim=1) > 0)
print(f"Surface nodes: {surf_mask.sum()} / {d_raw.num_nodes}")

# 4. 시각화 호출 (denormalized 데이터 전달)
plot_surface_pressure(d_raw, y_pred_denorm=y_pred_denorm, y_true_denorm=y_true_denorm)
```

또한 `plot_surface_pressure()`에 방어 로직 추가:
```python
def plot_surface_pressure(d, y_pred_denorm, y_true_denorm):
    # RAW features에서 surface mask (d.x가 normalized일 수 있으므로 방어)
    normals = d.x[:, 3:5]
    surf_mask = (normals.abs().sum(dim=1) > 0)
    
    # 방어: surface가 전체의 50% 이상이면 normalized features 의심
    if surf_mask.sum() > 0.5 * d.num_nodes:
        print(f"⚠️ WARNING: {surf_mask.sum()}/{d.num_nodes} nodes classified as surface. "
              f"This likely means d.x is normalized. Use RAW features for mask.")
```

### 체크리스트
- [ ] 노트북에서 `y_pred` 값 범위 확인 (normalized vs denormalized)
- [ ] `surf_mask` 개수 확인 (전체 노드의 ~5-10%만 surface여야 정상)
- [ ] `plot_surface_pressure()`에 경고 로직 추가
- [ ] `plot_pred_vs_gt()`에 `x_scaler`, `y_scaler` 정상 전달 확인
- [ ] 수정 후 Cp 분포가 GT와 유사한 형태인지 확인

---

## 2. BC Loss 업데이트

### 현재 상태
`src/navier_stokes_physics_loss.py` → `NavierStokesPhysicsLoss._bc_loss()` 메서드가 존재하지만 여러 문제가 있음.

### 관련 파일
- `src/navier_stokes_physics_loss.py` → `_bc_loss()`
- `src/normalizer.py` → `NormalizedDataset.__getitem__()` (BC mask 생성)
- `src/airfrans_utils.py` → `build_bc_masks_airfrans()`

### 발견된 문제점

#### 문제 1: Scaled space에서 BC를 적용하는 비일관성
`_bc_loss()`는 `pred_scaled` (U_ref와 L_ref로 무차원화된 값)를 입력으로 받는다. 이 space에서:
- Wall no-slip: `u* = u/U_ref = 0` → 올바름 ✅
- Inlet: `target = [[1.0, 0.0]]` (하드코딩) → **AoA ≠ 0인 경우 틀림** ❌
- Farfield: `target = [[1.0, 0.0]]` (하드코딩) → **AoA ≠ 0인 경우 틀림** ❌
- Outlet: `p* = 0` → 대략 맞지만, 실제 outlet 압력은 정확히 0이 아닐 수 있음 ⚠️

#### 문제 2: `inlet_u` 처리 로직 오류
```python
if inlet_u is not None:
    inlet_u = inlet_u.to(device)
    if inlet_u.size(0) != mask_in.sum():
        inlet_u_target = inlet_u[mask_in]  # ← 이미 mask 적용된 데이터에 또 mask 적용
    else:
        inlet_u_target = inlet_u
```
`inlet_u`는 `build_bc_masks_airfrans()`에서 전체 노드 크기 `[N, 2]`로 저장되거나, inlet 노드만 `[n_inlet, 2]`로 저장될 수 있다. 이 분기 로직이 두 경우를 제대로 처리하는지 검증 필요.

#### 문제 3: BC mask의 scaled space 미반영
BC targets는 **dimensional space**에서 정의되지만, `pred_scaled`는 이미 `U_ref`로 나뉜 상태. `_bc_loss()`에서 inlet target도 scaled space로 변환해야 함.

#### 문제 4: Weight가 너무 낮음
`self.bc_w`의 기본값이 `0.0`이라 명시적으로 설정하지 않으면 BC loss가 아예 계산되지 않음.

### 수정 방안

```python
def _bc_loss(self, pred_scaled: torch.Tensor, data: Any, Uref_local: float) -> torch.Tensor:
    """
    Soft BC penalties in dimensionless space.
    pred_scaled: u*=u/Uref, p*=p/Uref², nu_t*=nu_t/(Uref*Lref)
    """
    device = pred_scaled.device
    N = pred_scaled.size(0)
    u = pred_scaled[:, :2]
    p = pred_scaled[:, 2] if pred_scaled.size(1) >= 3 else torch.zeros(N, device=device)
    loss_terms = []

    # 1) Wall no-slip: u* = 0  (항상 정확)
    is_wall = getattr(data, 'is_wall', None)
    if is_wall is not None:
        mask_w = is_wall.bool().to(device)
        if mask_w.any() and mask_w.size(0) == N:
            loss_terms.append((u[mask_w] ** 2).mean())

    # 2) Inlet: u* = u_inlet / Uref  (AoA 고려)
    is_inlet = getattr(data, 'is_inlet', None)
    if is_inlet is not None:
        mask_in = is_inlet.bool().to(device)
        if mask_in.any() and mask_in.size(0) == N:
            inlet_u_phys = getattr(data, 'inlet_u', None)
            if inlet_u_phys is not None:
                # inlet_u는 physical velocity [N,2] 또는 [n_inlet,2]
                inlet_u_phys = inlet_u_phys.to(device)
                if inlet_u_phys.size(0) == N:
                    target_scaled = inlet_u_phys[mask_in] / max(Uref_local, 1e-12)
                elif inlet_u_phys.size(0) == mask_in.sum():
                    target_scaled = inlet_u_phys / max(Uref_local, 1e-12)
                else:
                    target_scaled = torch.ones_like(u[mask_in])
                    target_scaled[:, 1] = 0.0
                loss_terms.append(((u[mask_in] - target_scaled) ** 2).mean())
            else:
                # Fallback: (1,0) in scaled space → 영속적 AoA=0 가정
                target = torch.tensor([[1.0, 0.0]], device=device).expand_as(u[mask_in])
                loss_terms.append(0.1 * ((u[mask_in] - target) ** 2).mean())

    # 3) Farfield: 약한 constraint (pressure ≈ 0)
    is_far = getattr(data, 'is_farfield', None)
    if is_far is not None:
        mask_f = is_far.bool().to(device)
        if mask_f.any() and mask_f.size(0) == N:
            loss_terms.append(0.05 * (p[mask_f] ** 2).mean())

    # 4) Outlet: p* ≈ 0
    is_out = getattr(data, 'is_outlet', None)
    if is_out is not None:
        mask_o = is_out.bool().to(device)
        if mask_o.any() and mask_o.size(0) == N:
            loss_terms.append(0.1 * (p[mask_o] ** 2).mean())

    if len(loss_terms) == 0:
        return torch.zeros(1, device=device)
    return torch.stack(loss_terms).sum()  # sum이 mean보다 적절 (각 항이 이미 mean)
```

### 체크리스트
- [ ] `_bc_loss()`에 `Uref_local` 파라미터 추가 → inlet target을 scaled space로 변환
- [ ] `inlet_u` 크기 분기 로직 수정 (`[N,2]` vs `[n_inlet,2]` 구분)
- [ ] Farfield target에서 AoA 고려 (하드코딩 `[1,0]` 제거)
- [ ] `bc_loss_weight` 기본값을 `0.0 → 0.5~1.0`으로 변경 또는 문서화
- [ ] `forward()`에서 `_bc_loss()` 호출 시 `Uref_local` 전달
- [ ] 테스트 작성: `tests/test_bc_loss.py` (wall=0, inlet match, outlet p≈0 검증)
- [ ] `build_bc_masks_airfrans()`에서 `inlet_u` 형식 확인 및 통일

---

## 3. Continuity Loss 재검토

### 현재 상태
`conservative_divergence()` 함수가 FV-style divergence를 계산. 이를 `_continuity_loss()`에서 호출.

### 관련 파일
- `src/navier_stokes_physics_loss.py` → `conservative_divergence()`, `_continuity_loss()`
- `tests/test_continuity_loss.py`
- `tests/test_edge_attr_denorm_and_divergence.py`

### 검증해야 할 사항

#### 3.1 Normal vector 방향 일관성
Conservative divergence는 edge의 face-normal flux를 합산한다:
```
flux_e = u_face · n_e * |e|
```
여기서 normal `n = (dy/|e|, -dx/|e|)`. 만약 edge direction (`dx, dy`)이 일관되지 않으면 (같은 edge pair에 대해 `i→j`와 `j→i`의 dx,dy가 반대부호), `_half_edges()`의 `row < col` 필터가 일관된 방향을 보장하는지 확인 필요.

**검증 방법**:
```python
# GT velocity에 대해 divergence가 0에 가까운지 확인
div_gt = conservative_divergence(velocity=gt_velocity, ...)
print(f"div(GT) mean={div_gt.mean():.6f}, std={div_gt.std():.6f}, max={div_gt.abs().max():.6f}")
# 기대: mean ≈ 0, std < 0.01 (무차원)
```

#### 3.2 Node area 정확성
`node_area`가 없을 때 `(perimeter²)/(4π)`로 근사하는데, 이 근사가 실제 dual cell 면적과 얼마나 다른지 확인.

```python
# node_area가 있는 경우와 없는 경우의 divergence 비교
div_with_area = conservative_divergence(..., node_area=data.node_area)
div_without_area = conservative_divergence(..., node_area=None)
print(f"Ratio: {(div_with_area / div_without_area).median():.4f}")
```

#### 3.3 Dimensional scaling 정합성
`conservative_divergence()`에서 `Lref`로 geometry를 무차원화하는데, `_continuity_loss()`에서 이미 `pos_scaled`를 전달한다면 이중 scaling이 발생할 수 있음.

```python
# _continuity_loss에서:
div = conservative_divergence(
    velocity=u_scaled[:, :2],     # 이미 /Uref
    ...
    pos=pos_scaled,               # 이미 /Lref
    Lref=self.Lref,               # ← 또 /Lref?
)
```
**확인**: `Lref` 파라미터가 `conservative_divergence()` 내에서 `length = length * (1/Lref)`를 적용하므로, `pos_scaled`가 이미 `/Lref`라면 `Lref=1.0`을 전달해야 함.

#### 3.4 Edge attribute schema 자동 감지 신뢰성
`_extract_dxdy_length()`의 heuristic:
```python
is_default = (col1 <= 1.5) and torch.all(edge_attr[:, 0] > 0)
```
이 조건이 모든 그래프에서 올바르게 작동하는지 확인. 특히 매우 작은 edge (`dx ≈ 0`)가 있을 때.

### 체크리스트
- [ ] GT velocity에 대한 divergence 검증 (결과 ≈ 0 확인)
- [ ] `_continuity_loss()`에서 `Lref` 이중 적용 여부 확인
- [ ] `_half_edges()` normal 방향 일관성 테스트
- [ ] `node_area` 유무에 따른 divergence 비교
- [ ] `div_area_floor_factor`, `div_min_degree` 파라미터 효과 검증
- [ ] 기존 테스트 `test_continuity_loss.py` 전부 통과 확인

---

## 4. Momentum Loss 재검토

### 현재 상태
`_momentum_loss()` 메서드가 RANS momentum residual을 계산:
```
residual = convection + ∇p - viscous_diffusion = 0
```

### 관련 파일
- `src/navier_stokes_physics_loss.py` → `_momentum_loss()`, `weighted_gradient()`, `weighted_laplacian()`
- `tests/check_momentum_loss_on_gt.py`

### 검증해야 할 사항

#### 4.1 Weighted Laplacian의 정확도
현재 Laplacian은 "gradient of gradient" (2-pass):
```python
def weighted_laplacian(field, ...):
    gx, gy = weighted_gradient(field, ...)        # 1st pass
    dgxdx, _ = weighted_gradient(gx, ...)         # 2nd pass (x방향)
    _, dgydy = weighted_gradient(gy, ...)          # 2nd pass (y방향)
    return dgxdx + dgydy
```
이 방식은 unstructured mesh에서 **수치 확산/잡음이 크게 증폭**될 수 있음. 1차 gradient의 오차가 2차에서 배가됨.

**검증 방법**: 알려진 해석해 (예: `f(x,y) = sin(πx)sin(πy)`, Δf = -2π²f)에 대해 Laplacian 오차 측정.

#### 4.2 Skew-symmetric convection 항
```python
conv_u = 0.5 * (u*du/dx + v*du/dy) + 0.5 * (d(u²)/dx + d(uv)/dy)
```
이론적으로 올바르지만, `weighted_gradient`로 `u²`의 gradient를 구할 때 비선형 곱의 gradient가 정확한지 확인 필요.

#### 4.3 Eddy viscosity 처리
```python
nu_t = F.softplus(pred_scaled[:, 3])  # 항상 양수 보장
visc_u = (mol_coeff + nu_t) * lap_u + dnutdx * dudx + dnutdy * dudy
```
`dnutdx`, `dnutdy`는 `weighted_gradient(nu_t, ...)`로 구하는데, `nu_t`가 softplus 후이므로 gradient가 smooth한지 확인.

#### 4.4 GT에 대한 momentum residual 크기
```bash
python tests/check_momentum_loss_on_gt.py
```
GT velocity/pressure에 대한 momentum residual이 충분히 작은지 (수치 오차 수준) 확인. 만약 크다면 discrete operator 자체의 문제.

### 개선 방안 (필요 시)

**옵션 A: Direct Laplacian** (2-pass 대신)
```python
def direct_laplacian(field, edge_index, edge_attr, num_nodes, ...):
    """Δf(i) ≈ Σ_j w_ij (f_j - f_i) / r_ij²"""
    row, col = edge_index
    df = field[col] - field[row]
    inv_r2 = 1.0 / (length * length + eps)
    lap = scatter_add(w * df * inv_r2, row, dim=0, dim_size=num_nodes) * 2
    # *2 factor: unstructured mesh에서 Laplacian = 2d * weighted average of df/r²
    return lap / scatter_add(w, row, ...).clamp_min(1)
```

**옵션 B: Momentum loss weight를 낮게 유지**
Discrete operator의 정확도가 낮으면, momentum loss weight를 작게 유지하고 data loss에 의존.

### 체크리스트
- [ ] `check_momentum_loss_on_gt.py` 실행 → GT residual 크기 확인
- [ ] 해석해에 대한 `weighted_gradient()` 정확도 테스트
- [ ] 해석해에 대한 `weighted_laplacian()` 정확도 테스트 (vs direct Laplacian)
- [ ] `_momentum_loss()`에서 `Lref` 이중 적용 여부 확인 (continuity와 동일 이슈)
- [ ] Skew-symmetric convection의 에너지 보존 특성 검증
- [ ] `debug=True, debug_level=2`로 훈련 실행하여 각 항의 크기 비교

---

## 5. Normalization / Denormalization 문서화

### 목표
데이터 흐름 전체에서 normalization/denormalization이 **언제**, **어디서**, **왜** 일어나는지 명확하게 정리.

### 현재 구조 요약

```
[Raw AirfRANS Data]
  x: [N, 7] = [Vx∞, Vy∞, wall_dist, nx, ny, pos_x, pos_y]
  y: [N, 4] = [u, v, p, nu_t]
  pos: [N, 2] = [x, y]
      │
      ▼
[StandardScaler.fit()] ← train set의 x, y에 대해 한 번만 fit
  x_scaler.mean, x_scaler.std → [7]
  y_scaler.mean, y_scaler.std → [4]
      │
      ▼
[NormalizedDataset.__getitem__(idx)]
  1. dm.x = x_scaler.transform(raw_x)     ← normalized
  2. dm.y = y_scaler.transform(raw_y)     ← normalized
  3. dm.x_norm_params = {mean, scale}      ← 나중에 denorm용 저장
  4. dm.y_norm_params = {mean, scale}
  5. dm.edge_attr = raw edge_attr          ← NOT normalized (물리 좌표계)
  6. dm.pos = raw pos                      ← NOT normalized
  7. BC masks: build_bc_masks_airfrans(d_raw)  ← RAW features로 생성!
      │
      ▼
[Model forward pass]
  input: dm.x (normalized), dm.edge_attr (raw)
  output: y_pred (normalized space)
      │
      ▼
[Loss computation: NavierStokesPhysicsLoss.forward()]
  1. MSE loss: predictions vs targets (both normalized)     ← normalized space
  2. Physics loss preparation:
     a. preds_phys = denorm(predictions, y_norm_params)     ← denormalized
     b. targs_phys = denorm(targets, y_norm_params)         ← denormalized
     c. pos_phys = data.pos (already raw)                   ← raw
     d. x_phys = denorm(data.x, x_norm_params)             ← denormalized (for U∞ inference)
  3. Dimensional scaling:
     pred_scaled = preds_phys / [Uref, Uref, Uref², Uref*Lref]
     pos_scaled = pos_phys / Lref
  4. Continuity/Momentum: pred_scaled, pos_scaled 사용
  5. BC loss: pred_scaled 사용
      │
      ▼
[Visualization / Evaluation]
  y_pred_denorm = y_scaler.inverse(y_pred)    ← denormalized
  x_phys = x_scaler.inverse(dm.x)            ← denormalized (for Cp, surface mask)
```

### 핵심 주의사항

| 데이터 | Normalized? | 사용처 |
|--------|-------------|--------|
| `dm.x` (model input) | ✅ Yes | GNN forward pass |
| `dm.y` (training target) | ✅ Yes | MSE data loss |
| `dm.edge_attr` | ❌ No | GNN message passing, physics loss |
| `dm.pos` | ❌ No | Physics loss 좌표 |
| `dm.node_area` | ❌ No | Divergence 계산 |
| BC masks (`is_wall` 등) | N/A (boolean) | Physics loss, BC loss |
| `inlet_u` | ❌ No (physical) | BC loss target |
| `preds_phys` (physics loss 내부) | ❌ Denormalized | Continuity/Momentum |
| `pred_scaled` (physics loss 내부) | 무차원 (/Uref) | 실제 residual 계산 |

### ⚠️ 자주 발생하는 실수

1. **Surface mask를 normalized `dm.x`에서 생성** → 모든 노드가 surface로 분류됨
2. **Physics loss에서 edge_attr를 normalize** → 기하학적 의미 파괴
3. **Visualization에서 denormalize 누락** → 예측이 flat하게 보임
4. **`Lref` 이중 적용** → `pos_scaled = pos/Lref`를 전달한 뒤 내부에서 또 `/Lref`

### 체크리스트
- [ ] 위 flow chart를 `docs/plans/normalization_flow.md`로 저장
- [ ] Mermaid diagram 추가 (flow chart 시각화)
- [ ] `NormalizedDataset` docstring에 normalization 범위 명시
- [ ] `NavierStokesPhysicsLoss` docstring에 input space 명시
- [ ] 각 visualization 함수의 expected input space 명시

---

## 6. Attention Mechanism 조사 및 반영

### 현재 상태
`src/global_context_processor.py`에 attention-based global context가 구현됨:
- `PositionalEncoding` (sinusoidal)
- `MultiHeadSelfAttention` (scaled dot-product)
- `CrossAttention` (nodes ↔ global tokens)
- `GlobalContextProcessor` (Set2Set pooling → self-attention → cross-attention)

현재 `EnhancedCFDModelWithGlobalContext` (`01_trainer.ipynb`)에서 optional로 사용.

### 조사할 방향

#### 6.1 Graph Transformer 아키텍처 (문헌 조사)

| Method | 핵심 아이디어 | 적용 가능성 |
|--------|-------------|-------------|
| **Graph Attention Networks v2 (GATv2)** | Dynamic attention (key-query 분리 후 concat → LeakyReLU) | Message passing layer 교체 가능 |
| **GraphGPS** (Rampášek et al., 2022) | Local MPNN + Global attention 병렬 | 현재 구조와 유사하나 더 체계적 |
| **Mesh Transformer** (Lam et al., 2023) | 기상예측용 mesh attention, multi-scale | 대규모 mesh에 적합 |
| **GRIT** (Ma et al., 2023) | Random walk structural encoding + transformer | Positional encoding 개선 |
| **Transolver** (Wu et al., 2024) | Physics-attention with learnable slicing | CFD-specific attention |

#### 6.2 현재 구현의 개선점

**문제 1: Scalability**
현재 `GlobalContextProcessor`는 전체 노드를 attention에 넣는데, AirfRANS 그래프는 15k-30k 노드. Self-attention의 O(N²) 복잡도가 문제.

**개선안**: Attention을 cluster/region 단위로 수행:
```python
# 1. Virtual node / readout token 방식
#    K개의 global token이 모든 노드로부터 cross-attention으로 정보 수집
#    → global token끼리 self-attention → 다시 노드로 broadcast

# 2. Local window attention
#    K-hop neighborhood 내에서만 attention 수행
```

**문제 2: Positional Encoding**
현재 sinusoidal encoding은 sequence 기반이라 graph 구조를 반영 못함.

**개선안**: 
- Laplacian Positional Encoding (LPE): graph Laplacian의 고유벡터 사용
- Random Walk PE (RWPE): random walk 확률 사용
- Spatial PE: 실제 좌표 `(x, y)` 기반 encoding

**문제 3: 현재 사용 여부 불확실**
`01_trainer.ipynb`의 `SmokeCfg`에서 `use_global_context`가 어떻게 설정되어 있는지 확인 필요.

### 구현 우선순위 제안

1. **단기 (쉬움)**: GATv2Conv로 message passing layer 교체 테스트
2. **중기**: Virtual node + cross-attention (현재 구조 확장)
3. **장기**: GraphGPS-style 아키텍처 (local MPNN + global attention 병렬)

### 체크리스트
- [ ] 문헌 조사: Transolver, GraphGPS, MeshGraphNets attention 변형
- [ ] `01_trainer.ipynb`에서 `use_global_context=True`로 ablation 실험
- [ ] GATv2Conv 교체 실험 (PyG의 `GATv2Conv` 사용)
- [ ] Laplacian PE 구현 및 효과 측정
- [ ] Virtual node 방식 구현 (scalability 개선)
- [ ] Attention weight 시각화 → 물리적으로 의미 있는 패턴인지 확인

---

## 7. 기타 추천 사항

### 7.1 Edge Attribute Schema 통일
현재 두 가지 schema (`[dist, dir_x, dir_y]` vs `[dx, dy, dist]`)를 heuristic으로 자동 감지하는데, 이는 잠재적 오류 원인. **하나로 통일** 권장.

```python
# 권장: [dx, dy, dist] 형식으로 통일
# preprocess_airfrans_edges.py에서 생성 시 통일
# _extract_dxdy_length()의 auto-detect 로직은 backward compatibility로 유지
```

**체크리스트**:
- [ ] `build_edges_from_downsampled.py` 출력 schema 확인
- [ ] 새로 생성하는 edge는 항상 `[dx, dy, dist]`
- [ ] 기존 데이터 마이그레이션 스크립트 작성 (필요 시)

### 7.2 Physics Loss Debug 모드 활용
이미 강력한 debug 시스템이 있으므로 적극 활용:

```python
physics_loss = NavierStokesPhysicsLoss(
    debug=True,
    debug_level=2,     # 상세 통계
    debug_every=50,    # 50 step마다
)

# 훈련 후 확인
print(physics_loss.last_debug)
# → div 통계, gradient 크기, edge length, 경고 메시지 등
```

### 7.3 Force Coefficient 계산 검증
`src/force_coefficients_calculation.py`의 lift/drag 계산이 benchmark scoring과 일치하는지 확인.

**체크리스트**:
- [ ] GT 데이터로 Cl, Cd 계산 → 알려진 값과 비교
- [ ] Surface pressure integration 정확도 검증
- [ ] `scripts/score_benchmark.py`와의 일관성 확인

### 7.4 Multi-scale Model 검토
`02_trainer_multi_scale.ipynb`과 `src/multigraph_convolution.py`가 존재. 현재 활용도 확인 및 main pipeline과의 통합 검토.

### 7.5 Wandb Logging 개선
Physics loss의 개별 항 (continuity, momentum, BC, data) 및 curriculum ramp 값을 wandb에 기록하여 훈련 progress 모니터링 개선.

```python
# 이미 losses dict에 포함되어 있으므로, trainer에서 wandb.log()에 전달하면 됨
wandb.log({
    "loss/total": losses["total_loss"].item(),
    "loss/mse": losses["mse_loss"].item(),
    "loss/continuity": losses["continuity_loss"].item(),
    "loss/momentum": losses["momentum_loss"].item(),
    "physics/cont_weight": losses["cont_weight_used"].item(),
    "physics/mom_weight": losses["mom_weight_used"].item(),
    "physics/uref": losses["uref_used"].item(),
})
```

### 7.6 Benchmark 자동화
훈련 완료 후 자동으로 `score_benchmark.py` 실행하여 FLOW-GLIDE 메트릭 계산하는 파이프라인 구축.

### 7.7 테스트 커버리지 확대
현재 테스트가 물리 loss에 집중. 추가 테스트 권장:

| 테스트 대상 | 파일 | 내용 |
|------------|------|------|
| BC mask 생성 | `test_bc_masks.py` | RAW vs normalized features에서 mask 차이 |
| Normalization round-trip | `test_normalization.py` | `inverse(transform(x)) ≈ x` |
| Surface ordering | `test_surface_ordering.py` | `order_surface()` 결과 검증 |
| Force coefficients | `test_force_coefficients.py` | 알려진 airfoil의 Cl, Cd |
| Model I/O shapes | `test_model_shapes.py` | 다양한 그래프 크기에서 shape 검증 |

---

## 우선순위 정리

| 순위 | 항목 | 난이도 | 영향도 | 소요 시간 (추정) |
|------|------|--------|--------|-----------------|
| 🔴 1 | Surface Cp Visualization 버그 | 쉬움 | 높음 (디버깅 필수) | 1-2시간 |
| 🔴 2 | Normalization 문서화 | 쉬움 | 높음 (모든 작업의 기초) | 2-3시간 |
| 🟡 3 | Continuity Loss 재검토 | 보통 | 높음 | 3-5시간 |
| 🟡 4 | Momentum Loss 재검토 | 어려움 | 높음 | 5-8시간 |
| 🟡 5 | BC Loss 업데이트 | 보통 | 중간 | 3-4시간 |
| 🟢 6 | Attention Mechanism | 어려움 | 중간 | 1-2주 |
| 🟢 7 | 기타 추천 | 다양 | 다양 | 지속적 |

> **권장 순서**: 1 → 5 (문서화) → 3 (continuity) → 4 (momentum) → 2 (BC loss) → 6 (attention) → 7 (기타)
> 
> Visualization 버그를 먼저 수정해야 이후 physics loss 개선 결과를 시각적으로 확인할 수 있고,
> Normalization 문서화를 통해 전체 데이터 흐름을 이해해야 나머지 작업을 안전하게 진행할 수 있음.
