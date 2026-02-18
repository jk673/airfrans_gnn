# Momentum Loss 설계 문서

> **파일**: `src/physics_loss.py` — `NavierStokesPhysicsLoss._momentum_loss()`
>
> 이 문서는 프로젝트의 momentum loss가 **어떤 물리 방정식을 풀고 있는지**,
> **왜 그렇게 설계했는지**, 그리고 **코드가 수식과 어떻게 대응하는지**를 설명합니다.

---

## 1. 우리가 풀고 싶은 방정식

### 1.1 Steady Incompressible RANS

에어포일 주변 유동은 **정상(Steady)** 상태의 **비압축성(Incompressible) RANS** 방정식으로 기술됩니다.

$$
\underbrace{\mathbf{u} \cdot \nabla \mathbf{u}}_{\text{Convection}}
+ \underbrace{\nabla \tilde{p}}_{\text{Pressure gradient}}
- \underbrace{\nabla \cdot \left[ \left(\frac{1}{Re} + \nu_t^*\right) \nabla \mathbf{u} \right]}_{\text{Viscous diffusion}}
= \mathbf{0}
$$

여기서 모든 변수는 **무차원(dimensionless)** 입니다 (2절 참조).

| 기호 | 의미 | 단위 (무차원화 전) |
|------|------|------------------|
| $\mathbf{u} = (u, v)$ | 속도 벡터 | m/s |
| $\tilde{p}$ | 압력 (밀도로 나눈 값) | m²/s² |
| $\nu_t$ | 난류 점성 (eddy viscosity) | m²/s |
| $Re = U_\infty L / \nu$ | 레이놀즈 수 | - |

### 1.2 각 항의 물리적 의미

```
┌──────────────────────────────────────────────────────────┐
│  Convection (관성력)     유체가 스스로를 밀어내는 힘        │
│  + Pressure gradient     압력 차이에 의한 힘               │
│  - Viscous diffusion     점성이 속도를 골고루 퍼뜨리는 효과  │
│  = 0                     정상 상태 → 힘이 평형              │
└──────────────────────────────────────────────────────────┘
```

- **Convection** $(\mathbf{u} \cdot \nabla \mathbf{u})$: 유체 입자가 속도장을 따라 이동하면서 겪는 속도 변화.
- **Pressure gradient** $(\nabla \tilde{p})$: 고압 → 저압으로 유체를 미는 힘.
- **Viscous diffusion**: 분자 점성($1/Re$)과 난류 점성($\nu_t^*$)이 속도 차이를 균일화. RANS에서는 Reynolds stress를 eddy viscosity 모델($\nu_t$)로 근사.

---

## 2. 무차원화 (Nondimensionalization)

GNN 예측값은 **물리 단위(m/s, Pa, ...)**로 denormalize된 뒤, 아래 규칙으로 무차원화됩니다.

### 2.1 Reference Scales

| Scale | 기호 | 결정 방법 |
|-------|------|----------|
| 속도 | $U_\infty$ | inlet 마스크 노드의 속도 크기 median (동적 추정) |
| 길이 | $L_{ref}$ | 코드 길이 `chord_length` (기본값 1.0) |

### 2.2 Scaling Rules

$$
u^* = \frac{u}{U_\infty}, \quad
v^* = \frac{v}{U_\infty}, \quad
\tilde{p}^* = \frac{\tilde{p}}{U_\infty^2}, \quad
\nu_t^* = \frac{\nu_t}{U_\infty L_{ref}}, \quad
\mathbf{x}^* = \frac{\mathbf{x}}{L_{ref}}
$$

이렇게 하면 모멘텀 방정식의 분자 점성 계수가 $1/Re$로 자연스럽게 나타납니다:

$$
\frac{1}{Re} = \frac{\nu}{U_\infty L_{ref}}
$$

> **코드 위치**: `_apply_dimensional_scaling_with_Uref()` (line 532)

### 2.3 mol_coeff 계산

```python
# dynamic_re_from_data=True (기본값)
mol_coeff = nu_molecular / (U_ref * L_ref)   # = 1/Re

# dynamic_re_from_data=False
mol_coeff = 1.0 / Re                         # 고정 Re 사용
```

---

## 3. Momentum Loss 상세 설계

### 3.1 x-방향 모멘텀 (u-momentum)

무차원 변수를 대입하면 (이하 $*$ 표기 생략):

$$
\underbrace{u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}}_{\text{Convection}}
+ \underbrace{\frac{\partial p}{\partial x}}_{\text{Pressure}}
- \underbrace{\left(\frac{1}{Re} + \nu_t\right) \Delta u
  - \frac{\partial \nu_t}{\partial x} \frac{\partial u}{\partial x}
  - \frac{\partial \nu_t}{\partial y} \frac{\partial u}{\partial y}}_{\text{Viscous}}
= 0
$$

y-방향도 동일한 구조 ($u \to v$, $\partial p / \partial x \to \partial p / \partial y$).

### 3.2 Residual 정의

**Residual** = 방정식의 좌변 값. 정확한 해에서는 0이어야 합니다.

$$
R_u = \text{Conv}_u + \frac{\partial p}{\partial x} - \text{Visc}_u
$$

$$
R_v = \text{Conv}_v + \frac{\partial p}{\partial y} - \text{Visc}_v
$$

**Momentum loss** = 이 residual의 크기를 최소화:

$$
\mathcal{L}_{\text{mom}} = \frac{1}{2N} \sum_{i=1}^{N} \left( R_{u,i}^2 + R_{v,i}^2 \right)
$$

(또는 Huber loss 사용 시 smooth L1으로 대체)

---

## 4. Convection: Skew-Symmetric Form

### 4.1 왜 Skew-Symmetric인가?

Convection을 이산화하는 방법은 여러 가지인데, 이 프로젝트에서는 **skew-symmetric form**을 사용합니다.

| Form | 수식 | 특징 |
|------|------|------|
| Advective | $\mathbf{u} \cdot \nabla \mathbf{u}$ | 직관적이지만 에너지 비보존 |
| Divergence | $\nabla \cdot (\mathbf{u} \otimes \mathbf{u})$ | 보존형이지만 연속방정식 오차 유입 |
| **Skew-symmetric** | $\frac{1}{2}[\mathbf{u} \cdot \nabla \mathbf{u} + \nabla \cdot (\mathbf{u} \otimes \mathbf{u})]$ | **두 장점의 평균** |

> 연속 공간에서는 비압축성($\nabla \cdot \mathbf{u} = 0$)이면 세 형태가 동일합니다.
> 하지만 비정형 메시의 이산 근사에서는 $\nabla \cdot \mathbf{u} \neq 0$이 약간 남기 때문에
> skew-symmetric form이 **가짜 에너지 생성/소멸을 방지**하여 안정적입니다.

### 4.2 수식 전개 (x-방향)

**Advective form:**

$$
(\mathbf{u} \cdot \nabla u) = u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}
$$

**Divergence form** ($\nabla \cdot (\mathbf{u} u)$ 는 $\mathbf{u}$의 x-성분에 대해):

$$
\nabla \cdot (u \mathbf{u}) = \frac{\partial (u^2)}{\partial x} + \frac{\partial (uv)}{\partial y}
$$

**Skew-symmetric (평균):**

$$
\text{Conv}_u = \frac{1}{2} \left[
  \underbrace{u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}}_{\text{advective}}
  + \underbrace{\frac{\partial (u^2)}{\partial x} + \frac{\partial (uv)}{\partial y}}_{\text{divergence}}
\right]
$$

### 4.3 코드 대응

```python
# Advective form
conv_u_std = u * dudx + v * dudy

# Divergence form: ∂(u²)/∂x + ∂(uv)/∂y
u2 = u * u;  uv = u * v
du2dx, _ = weighted_gradient(u2, ...)
_, duvdy  = weighted_gradient(uv, ...)

# Skew-symmetric average
conv_u = 0.5 * (conv_u_std + (du2dx + duvdy))
```

---

## 5. Viscous Term: Product Rule Expansion

### 5.1 수식

완전한 viscous term은:

$$
\nabla \cdot \left[ (\nu_{\text{mol}} + \nu_t) \nabla u \right]
$$

여기서 $\nu_{\text{mol}} = 1/Re$ (무차원 분자 점성).

이를 **product rule**로 전개하면:

$$
\nabla \cdot \left[ (\nu_{\text{mol}} + \nu_t) \nabla u \right]
= \underbrace{(\nu_{\text{mol}} + \nu_t) \Delta u}_{\text{(A) 균일 점성 확산}}
+ \underbrace{\nabla \nu_t \cdot \nabla u}_{\text{(B) 점성 변화 효과}}
$$

유도 과정 (2D):

$$
\frac{\partial}{\partial x}\left[(\nu_{\text{mol}}+\nu_t)\frac{\partial u}{\partial x}\right]
+ \frac{\partial}{\partial y}\left[(\nu_{\text{mol}}+\nu_t)\frac{\partial u}{\partial y}\right]
$$

$$
= (\nu_{\text{mol}}+\nu_t)\frac{\partial^2 u}{\partial x^2}
+ \frac{\partial \nu_t}{\partial x}\frac{\partial u}{\partial x}
+ (\nu_{\text{mol}}+\nu_t)\frac{\partial^2 u}{\partial y^2}
+ \frac{\partial \nu_t}{\partial y}\frac{\partial u}{\partial y}
$$

$$
= (\nu_{\text{mol}}+\nu_t) \underbrace{\left(\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}\right)}_{\Delta u}
+ \frac{\partial \nu_t}{\partial x}\frac{\partial u}{\partial x}
+ \frac{\partial \nu_t}{\partial y}\frac{\partial u}{\partial y}
$$

> **참고**: $\nu_{\text{mol}}$은 공간 상수이므로 $\nabla \nu_{\text{mol}} = 0$.
> gradient 항에는 $\nabla \nu_t$만 남습니다.

### 5.2 nu_t의 양수 보장

모델이 출력하는 $\nu_t$ 값은 음수일 수 있습니다. 물리적으로 점성은 양수여야 하므로 **softplus**를 적용합니다:

$$
\nu_t^+ = \text{softplus}(\nu_t) = \ln(1 + e^{\nu_t})
$$

softplus는 ReLU와 달리 0 근방에서 미분 가능하므로 gradient flow가 끊기지 않습니다.

### 5.3 코드 대응

```python
nu_t = F.softplus(nu_t)   # 양수 보장

# Laplacian: Δu (2-pass weighted gradient, 6절 참조)
lap_u = weighted_laplacian(u, ...)
lap_v = weighted_laplacian(v, ...)

# ∇ν_t
dnutdx, dnutdy = weighted_gradient(nu_t, ...)

# (A) + (B) 조합
visc_u = (mol_coeff + nu_t) * lap_u + dnutdx * dudx + dnutdy * dudy
visc_v = (mol_coeff + nu_t) * lap_v + dnutdx * dvdx + dnutdy * dvdy
```

---

## 6. 이산 미분 연산자 (Discrete Differential Operators)

비정형 메시에서는 유한차분을 직접 쓸 수 없으므로, **가중 최소제곱 기반 그래프 연산자**를 사용합니다.

### 6.1 Weighted Gradient

노드 $i$에서의 스칼라 장 $f$의 gradient 근사:

$$
\left(\frac{\partial f}{\partial x}\right)_i
\approx
\frac{\displaystyle\sum_{j \in \mathcal{N}(i)} w_{ij} \, (f_j - f_i) \, \frac{\Delta x_{ij}}{r_{ij}^2}}
     {\displaystyle\sum_{j \in \mathcal{N}(i)} w_{ij}}
$$

여기서:
- $\mathcal{N}(i)$: 노드 $i$의 이웃 (그래프 edge로 연결된 노드)
- $\Delta x_{ij} = x_j - x_i$, $\Delta y_{ij} = y_j - y_i$
- $r_{ij} = \sqrt{\Delta x_{ij}^2 + \Delta y_{ij}^2}$ (에지 길이)
- $w_{ij}$: 거리 기반 가중치

**가중치 선택** (`weight_mode`):

| Mode | $w_{ij}$ | 특징 |
|------|----------|------|
| `rbf` (기본) | $\exp\!\left(-r_{ij}^2 / h^2\right)$, $h = \bar{r}$ | 가까운 이웃 강조, smooth |
| `inv_r2` | $1 / r_{ij}^2$ | 더 날카로운 locality |

> **코드 위치**: `weighted_gradient()` (line 190)

### 6.2 Weighted Laplacian (2-Pass)

Laplacian $\Delta f = \partial^2 f / \partial x^2 + \partial^2 f / \partial y^2$ 를 **gradient의 divergence**로 근사합니다:

$$
\Delta f \approx \frac{\partial g_x}{\partial x} + \frac{\partial g_y}{\partial y}
$$

여기서 $(g_x, g_y) = \nabla f$ 는 6.1절의 weighted gradient로 먼저 계산합니다.

```
Pass 1:  (gx, gy) = weighted_gradient(f)
Pass 2:  dgx/dx   = weighted_gradient(gx)  → x-성분만 사용
         dgy/dy   = weighted_gradient(gy)  → y-성분만 사용
Δf ≈ dgx/dx + dgy/dy
```

> **알려진 한계**: 2-pass 방식은 1st pass의 수치 오차가 2nd pass에서 증폭될 수 있습니다.
> 특히 비등방(anisotropic) 메시에서 Laplacian이 과도하게 smooth되거나
> 크기가 왜곡될 수 있습니다. 직접 2차 도함수를 계산하는 방법이 대안이지만,
> 현재 구현에서는 안정성과 구현 간결성을 위해 2-pass를 사용합니다.

> **코드 위치**: `weighted_laplacian()` (line 254)

### 6.3 Edge Halving

입력 그래프는 양방향 edge $(i \to j, j \to i)$를 모두 가집니다.
flux/gradient 계산 시 중복 카운팅을 방지하기 위해 $i < j$ 인 edge만 사용하고,
결과를 양쪽 노드에 **symmetric하게 누적**합니다.

```python
# 한 방향만 유지
mask = row < col
edge_index = edge_index[:, mask]

# 양쪽 노드에 누적 (symmetric accumulation)
num_x = scatter_add(gx_edge, row, ...) + scatter_add(gx_edge, col, ...)
```

---

## 7. 전체 Momentum Loss 흐름

```
                    ┌──────────────┐
                    │  GNN Output  │  (normalized space)
                    └──────┬───────┘
                           │ denormalize (y_norm_params)
                           ▼
                    ┌──────────────┐
                    │ Physical     │  u [m/s], v [m/s],
                    │ Predictions  │  p [m²/s²], nu_t [m²/s]
                    └──────┬───────┘
                           │ ÷ U_ref, L_ref (무차원화)
                           ▼
                    ┌──────────────┐
                    │ Scaled       │  u*, v*, p*, nu_t*
                    │ Predictions  │  (dimensionless)
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌───────────┐ ┌────────┐ ┌──────────┐
        │ Convection│ │Pressure│ │ Viscous  │
        │ (skew)    │ │Gradient│ │ Diffusion│
        └─────┬─────┘ └───┬────┘ └────┬─────┘
              │            │           │
              └────────────┼───────────┘
                           ▼
                    ┌──────────────┐
                    │   Residual   │  R_u, R_v
                    │  = Conv + ∇p │
                    │    - Visc    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  MSE or      │
                    │  Huber Loss  │  → L_mom
                    └──────────────┘
```

---

## 8. Curriculum Learning (가중치 스케줄)

Momentum loss는 처음부터 강하게 적용하지 않습니다.
모델이 기본적인 데이터 패턴을 먼저 학습한 뒤, 점진적으로 물리 제약을 강화합니다.

### 8.1 가중치 램프

$$
w_{\text{mom}}(t) = w_0 + (w_{\text{target}} - w_0) \cdot r(t)
$$

여기서 $r(t) \in [0, 1]$은 ramp factor:

**Linear ramp:**

$$
r(t) = \text{clamp}\left(\frac{t - t_{\text{start}}}{T_{\text{ramp}}}, \ 0, \ 1\right)
$$

**Cosine ramp:**

$$
r(t) = \frac{1}{2}\left[1 - \cos\left(\pi \cdot \text{clamp}\left(\frac{t - t_{\text{start}}}{T_{\text{ramp}}}, 0, 1\right)\right)\right]
$$

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `momentum_loss_weight` ($w_0$) | 시작 가중치 | 0.05 |
| `momentum_target_weight` ($w_{\text{target}}$) | 목표 가중치 | 0.20 |
| `mom_ramp_start_epoch` ($t_{\text{start}}$) | 램프 시작 에포크 | 공용 값 사용 |
| `mom_ramp_epochs` ($T_{\text{ramp}}$) | 램프 기간 (에포크) | 공용 값 사용 |
| `ramp_mode` | linear 또는 cosine | linear |

### 8.2 Total Loss에서의 위치

$$
\mathcal{L}_{\text{total}} =
  w_{\text{data}} \cdot \mathcal{L}_{\text{MSE}}
+ w_{\text{cont}}(t) \cdot \mathcal{L}_{\text{continuity}}
+ w_{\text{mom}}(t) \cdot \mathcal{L}_{\text{momentum}}
+ w_{\text{bc}}(t) \cdot \mathcal{L}_{\text{BC}}
$$

---

## 9. 코드 참조 (Quick Reference)

| 기능 | 함수/메서드 | 파일 위치 |
|------|-----------|----------|
| Momentum residual 계산 | `_momentum_loss()` | `src/physics_loss.py:642` |
| Weighted gradient | `weighted_gradient()` | `src/physics_loss.py:190` |
| Weighted Laplacian | `weighted_laplacian()` | `src/physics_loss.py:254` |
| Edge halving | `_half_edges()` | `src/physics_loss.py:28` |
| dx, dy 추출 | `_extract_dxdy_length()` | `src/physics_loss.py:39` |
| 무차원화 | `_apply_dimensional_scaling_with_Uref()` | `src/physics_loss.py:532` |
| Softplus nu_t | `F.softplus()` | `src/physics_loss.py:656` |
| Huber/MSE 선택 | `_quad_or_huber()` | `src/physics_loss.py:578` |
| Curriculum ramp | `_ramp_factor()` | `src/physics_loss.py:500` |
| Total loss 조합 | `forward()` | `src/physics_loss.py:858` |
