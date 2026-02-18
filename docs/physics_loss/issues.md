# Physics Loss Issues Report

Date: 2026-02-18
Updated: 2026-02-18

## Overview

`src/physics_loss.py`의 `NavierStokesPhysicsLoss` 클래스에서 발견된 문제점 목록.
심각도 순으로 정리.

---

## Critical Bugs — 수정 완료

### 1. `F.softplus(nu_t)` — 무차원화 후 적용으로 점성 ~700배 팽창 ✅

**위치:** `_momentum_loss()` (원래 line 651-652)

**현상:**
```python
nu_t = pred_scaled[:, 3]   # nu_t_phys / (U_ref * L_ref) → ~1e-4 ~ 1e-2
nu_t = F.softplus(nu_t)    # softplus(작은값) ≈ ln(2) ≈ 0.693
```

`pred_scaled[:, 3]`는 이미 `nu_t_phys / (U_ref * L_ref)`로 무차원화된 값(~1e-4 ~ 1e-2).
`F.softplus(x)`는 x→0일 때 ln(2) ≈ 0.693을 반환하므로, 실제 무차원 난류 점성(~0.001)이
~0.693으로 팽창 (약 700배). 점성항이 대류/압력항을 완전히 지배하여 모멘텀 잔차가 무의미.

**수정:** `_apply_dimensional_scaling_with_Uref`에서 물리 공간(스케일링 전)에
`F.softplus` 적용. `torch.cat` 기반 구현으로 autograd 호환성 보장.

---

### 2. Farfield/Inlet fallback BC에서 `[1,0] / U_ref` 이중 나눗셈 ✅

**위치:** `_bc_loss()` (farfield, inlet fallback)

**현상:**
```python
# u_far = pred_scaled[:, :2] = u_phys / U_ref  (이미 무차원)
target_u = torch.tensor([[1.0, 0.0]]) / scale_u  # = [1/U_ref, 0]
# Loss: (u_phys/U_ref - 1/U_ref)^2 → u_phys → 1 m/s 강제 (실제 ~30 m/s)
```

무차원 공간에서 타겟은 `U_inf / U_ref ≈ [cos(α), sin(α)]`이어야 하는데,
`[1/U_ref, 0]`으로 설정되어 물리적으로 완전히 틀린 타겟을 강제.

**수정:** `_get_freestream_scaled` 헬퍼 추가. `x_phys[:, 0:2]`에서 실제 자유류 속도를
읽어 `U_ref`로 나눔. Issue #3도 동시에 해결 (AoA 반영).

---

## Significant Issues — 수정 완료

### 3. 자유류 방향 `[1, 0]` 하드코딩 — 받음각(AoA) 무시 ✅

**위치:** `_bc_loss()` farfield/inlet fallback

AirfRANS 데이터셋은 다양한 받음각을 포함하지만, BC 타겟이 항상 `[1, 0]` (x방향).

**수정:** Issue #2와 함께 해결. `_get_freestream_scaled`가 `x_phys[:, 0:2]`에서
실제 자유류 속도를 사용하므로 AoA가 자동 반영.

### 4. Wall BC에서 `nu_t = 0` 조건 누락 ✅

**위치:** `_bc_loss()` wall section

```python
wall_loss = (u_wall ** 2).mean()  # no-slip만 적용
# nu_t = 0 at wall: 누락
```

**수정:** `pred_scaled[:, 3]` (nu_t)에 대해 `(nut_wall ** 2).mean()` 페널티 추가.

### 5. 배치 내 서로 다른 유동 조건에 단일 `U_ref` 적용 ✅

**위치:** `forward()`

배치에 받음각/자유류 속도가 다른 그래프가 섞이면 단일 U_ref로의 무차원화가
개별 그래프에 맞지 않아 physics loss 왜곡.

**수정:** `Uref_local`을 스칼라에서 per-node 텐서 `[N]`으로 변경.
`x_phys[:, :2].norm(dim=1)`로 계산 — AirfRANS에서 자유류 속도 feature는
그래프 내 모든 노드에 동일하므로, per-node = per-graph.
`_apply_dimensional_scaling_with_Uref`, `_bc_loss`, `_get_freestream_scaled`,
`mol_coeff` 모두 텐서 U_ref 지원.

### 6. Continuity — `node_area` 미계산 (proxy 사용) ✅

**위치:** `conservative_divergence()` + `NormalizedDataset`

CLI 학습 파이프라인에서 `prepare_airfrans_graph_for_physics` 미호출로
`node_area=None`, perimeter² / (4π) 근사로 대체.

**수정:** `NormalizedDataset.__getitem__`에서 `estimate_node_area(d_raw)` 호출 추가.

### 10. `Lref` double-scaling 위험 (edge_attr → pos 폴백 시) ✅

**위치:** `_continuity_loss`, `_momentum_loss`

`weighted_gradient`에 `pos_scaled` (이미 /Lref)와 `Lref`를 동시 전달.
`edge_attr` schema 감지 실패 시 pos 폴백으로 이중 스케일링.

**수정:** `pos_phys` (raw physical positions)를 전달하도록 변경.
Lref 스케일링은 `weighted_gradient` 내부에서만 수행.

---

## Deferred Issues (보류)

### 7. Momentum — Two-pass Laplacian 오차 증폭

**위치:** `weighted_laplacian()` (line 256-278)

gradient를 두 번 적용하여 Laplacian 계산. 비정렬 메시에서
1차 gradient 오차가 2차에서 제곱으로 증폭.

**상태:** 표준 approach이며 대안(direct Laplacian stencil) 구현은 validation 필요. 보류.

### 9. 점성 항 단순화 — (∇u)^T 항 누락

RANS 점성 항 `∇·[ν_eff(∇u + (∇u)^T)]` 중 `(∇u)^T` 누락.
∇·u = 0이면 소멸하지만, 학습 초기에는 연속 방정식 미충족으로 오차 유발.

**상태:** ∇·u ≈ 0 조건에서 유효한 근사. 학습 후기에는 영향 미미. 보류.

---

## Design Limitations (설계 한계)

### 8. Radius graph에서 Green-Gauss divergence 정합성

`conservative_divergence`는 에지가 닫힌 control volume을 형성한다고 가정하지만,
radius graph/KNN graph 에지는 proper dual cell을 구성하지 않음.
이산 발산 연산자의 conservation 성질이 보장되지 않음.

**상태:** 근본적 설계 한계. Radius graph 기반 GNN의 공통 문제.
정확한 해결은 Voronoi dual mesh 구성이 필요하며 현재 아키텍처와 호환 어려움.
