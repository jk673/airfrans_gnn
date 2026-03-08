# Physics Loss Full Audit Report — 2026-03-08

> 5인 전문가 팀에 의한 `src/physics_loss.py` 종합 감사 보고서.
> 각 전문가의 분석 결과를 별도 섹션으로 구성.

---

## Table of Contents

1. [T1 — 연구자 (Literature Survey)](#t1--연구자-literature-survey)
2. [T2 — 레포 스카우터 (GitHub Repository Analysis)](#t2--레포-스카우터-github-repository-analysis)
3. [T3 — 과학자 (Physics & Numerical Correctness)](#t3--과학자-physics--numerical-correctness)
4. [T4 — 소프트웨어 엔지니어 (Code Structure & Performance)](#t4--소프트웨어-엔지니어-code-structure--performance)
5. [T5 — ML 엔지니어 (Machine Learning Optimization)](#t5--ml-엔지니어-machine-learning-optimization)
6. [팀 리더 — 통합 결론 및 실행 계획](#팀-리더--통합-결론-및-실행-계획)

---

# T1 — 연구자 (Literature Survey)

## 1. 벤치마크 참조 논문

### AirfRANS Dataset (Bonnet et al., NeurIPS 2022)
- 2D NACA 에어포일 위 정상 비압축 RANS 시뮬레이션 (OpenFOAM, Spalart-Allmaras)
- 4가지 태스크: full, scarce, Reynolds 외삽, 받음각 외삽
- GraphSAGE 및 PointNet 기본선 제시

### FLOW-GLIDE (Su et al., Applied Sciences 2025) — 현재 SOTA
- volume_rel_l2 = 0.0038, surface_rel_l2 = 0.0063 (이전 SOTA 대비 62%, 82% 감소)
- MAG-BLOCK: 로컬 메시지 패싱 + 글로벌 셀프 어텐션 교차 구조
- 4개 MAG-BLOCK (각 3 MP + 1 attention)
- **물리 손실 미사용** — 순수 데이터 기반

### Transolver (Wu et al., ICML 2024 Spotlight)
- Physics-Attention: 도메인을 학습 가능한 슬라이스로 적응적 분할
- AirfRANS volume_rel_l2 = 0.0100

### B-GNN (arXiv 2503.18638, March 2025)
- 표면 메시 전용 GNN. 물리 기반 입력 (로컬 Re, 패널법 Cp)
- 모델 83% 축소, 훈련 데이터 87% 축소, 일반화 7배 향상

## 2. 물리 손실 이산화 방법론

### 2.1 연속 방정식 (발산 계산)

| 접근법 | 특징 | 사용처 |
|--------|------|--------|
| **보존형 FVM** | 면-법선 플럭스 scatter_add → 셀 잔차 | Gen-FVGN, 우리 구현 (의도) |
| **비보존형** | du/dx + dv/dy (WLSQ 기울기) | Gen-FVGN (대안), PhysicsNeMo |
| **관련 행렬** | -(B^T * g) | OCGNN |
| **하드 제약** | 스트림 함수 u=∂ψ/∂y, v=-∂ψ/∂x | Neural Conservation Laws (NeurIPS 2022) |

**핵심 발견**: Green-Gauss 발산 정리 기울기는 비구조 메시에서 0차 정확도. 최소제곱 기울기가 1차 이상 보장.

### 2.2 기울기 및 라플라시안

- **PhyMPGN** (ICLR 2025): 이산 Laplace-Beltrami 연산자를 MP 내 학습 가능 블록으로 인코딩
- **OCGNN** (2024): 관련 행렬 B를 통한 기울기-발산 쌍대성, 에너지 보존 보장
- 직접 그래프 라플라시안 `L*f = sum w*(f_j-f_i)`가 2-패스보다 오차 증폭 방지

### 2.3 대류항

| 형태 | 특성 |
|------|------|
| 비대칭 (우리 구현) | 이산 수준에서 운동 에너지 보존. 문헌에서 안정성 확인 |
| 표준 `u·∇u` | 단순하나 에너지 비보존 |
| 보존형 (FVM 플럭스) | Gen-FVGN 사용. 질량/운동량 보존 |
| FFV-PINN (2025) | 간소화 FVM + 잔차 보정항 |

### 2.4 에디 점성(ν_t) 처리

1. 예측 ν_t를 물리 손실에 사용 (우리 방식) — 오차 전파 위험
2. 정답 ν_t 사용 — 단순, 오차 전파 없음
3. ν_t에 stop-gradient — 초기 훈련 안정성 (문헌 권장)

## 3. 적응형 손실 가중치 방법론

| 방법 | 논문 | 핵심 |
|------|------|------|
| **GradNorm** | Chen et al., ICML 2018 | 기울기 노름 균형화. 불안정할 수 있음 |
| **ReLoBRaLo** | Bischof & Kraus, CMAME 2025 | 상대 진행도 기반, 기울기 계산 불필요. GradNorm/SoftAdapt 능가 |
| **불확실성 가중치** | Kendall et al., CVPR 2018 | 학습 가능 로그-분산으로 자동 균형 |
| **NTK 분석** | Wang et al., JCP 2022 | 수렴률 불균형의 이론적 기초. 왜 균등 가중이 실패하는지 규명 |
| **PECANN** | JCP 2022/2025 | 증강 라그랑주 + 조건부 적응 페널티 |
| **다목적 손실 균형** | ETH Zurich, CMAME 2025 | 스케일 + 립시츠 상수 동시 균형 |
| **동적 커리큘럼** | NeurIPS ML4PS 2024 | 난이도 인식 스케줄링. 균일 에폭 가정의 결함 해결 |

## 4. 수치 안정성 알려진 문제

1. **기울기 경직성**: 2차 도함수의 2-패스 추정이 오차 증폭
2. **스케일 불균형**: 연속 잔차 O(0.01) vs 운동량 잔차 O(1.0) — NTK 분석에서 확인
3. **저차수 노드**: deg < 3에서 기울기/발산 추정 불안정
4. **작은 이중 면적**: 인위적으로 큰 발산값 생성
5. **엣지 길이 분포**: 경계층 vs 원방에서 10-100배 차이

## 5. 최첨단 혁신 (2024-2025)

| 혁신 | 핵심 |
|------|------|
| **등변 GNN** (Kurz et al., 2025) | NS의 회전/반사/병진 대칭을 아키텍처에 내장 |
| **From Zero to Turbulence** (ICLR 2024) | 생성 모델로 초기 과도 단계 건너뛰기 |
| **Graph Transformer 역문제** (Jan 2025) | 희소 표면 측정→전체 유동장 복원 |
| **FVM 스킴 자동 발견** (2025) | GNN으로 최적 플럭스 계수 발견 |
| **Physics-informed KAN + ALM** (Nature Sci Rep 2025) | Kolmogorov-Arnold 네트워크 + 증강 라그랑주 |

## 6. 비차원화

- 비차원화로 조건수 κ >> 1 → κ ~ 1-10 개선 (훈련 안정성 핵심)
- 통계 정규화 + 방정식 변환이 가장 효과적 (Zhu et al., JCP 2025)
- **비차원화만으로 기울기 실패 해결 불충분** — 적응형 가중치 여전히 필요

## 주요 출처

- AirfRANS (Bonnet et al., NeurIPS 2022) — arxiv.org/abs/2212.07564
- FLOW-GLIDE (Su et al., Applied Sciences 2025) — mdpi.com/2076-3417/15/19/10834
- MeshGraphNets (Pfaff et al., ICLR 2021) — arxiv.org/abs/2010.03409
- CFDGCN (Belbute-Peres et al., ICML 2020) — arxiv.org/abs/2007.04439
- Transolver (Wu et al., ICML 2024) — arxiv.org/abs/2402.02366
- PhyMPGN (ICLR 2025) — arxiv.org/abs/2410.01337
- OCGNN (2024) — arxiv.org/abs/2512.11860
- Neural Conservation Laws (NeurIPS 2022) — arxiv.org/abs/2210.01741
- FFV-PINN (CMAME 2025) — sciencedirect.com/S0045782525004116
- FVGN (2023) — arxiv.org/abs/2309.10050
- GradNorm (Chen et al., 2018)
- ReLoBRaLo (Bischof & Kraus, CMAME 2025) — arxiv.org/abs/2110.09813
- Uncertainty Weighting (Kendall et al., CVPR 2018) — arxiv.org/abs/1705.07115
- NTK PINNs (Wang et al., JCP 2022) — arxiv.org/abs/2007.14527
- Causal PINN (Wang et al., CMAME 2024) — arxiv.org/abs/2203.07404
- PECANN (JCP 2022/2025) — arxiv.org/abs/2306.04904
- Non-dim for PINNs (Zhu et al., JCP 2025)
- B-GNN (March 2025) — arxiv.org/abs/2503.18638
- Equivariant GNN Turbulence (Kurz et al., 2025) — arxiv.org/abs/2504.07741

---

# T2 — 레포 스카우터 (GitHub Repository Analysis)

## 1. 핵심 저장소 분석

### 1A. Gen-FVGN (Finite Volume Graph Network) — 가장 관련성 높음

**Repos**: [Litianyu141/Gen-FVGN-steady](https://github.com/Litianyu141/Gen-FVGN-steady)

**기울기 계산 (WLSQ)**:
- Taylor 전개 변위 행렬을 노드별로 구성
- 2차, 3차, 4차 다항식 지원
- 가중치: 역거리 `w = 1/r^n` (n=3,4,5) 또는 MLS RBF `w = exp(-(r/r_max)^2)`
- `torch.linalg.solve(A, B)`로 풀 least-squares 시스템 풀이
- **경계 처리**: 고스트 포인트 — 반사(벽: `mirror_phi = -out_phi`), 복제(유입/유출)

**발산 계산**:
- 보존형: `scatter_add(matmul(uv_face, cells_face_surface_vec), cells_index)` — 면 법선 벡터 사용
- 비보존형: `du/dx + dv/dy` (WLSQ 기울기 × 셀 면적)

**운동량 계산**:
- 보존형: `J_flux = (convection_flux + P_flux - vis_flux) * surface_vec`
- 비보존형: `conv_cell + grad_p - viscosity_force = 0`

**손실 결합**:
```python
loss = params.loss_press * loss_press + params.loss_cont * loss_cont + params.loss_mom * (loss_mom_x + loss_mom_y)
loss = torch.mean(torch.log(loss))  # log-space 평균 — 수치 안정성
```

**우리 구현과 비교**:
- 우리: `sum(w*(f_j-f_i)*d/r²)/sum(w)` (1차 가중 평균). Gen-FVGN: 풀 최소제곱 시스템 (고차 정확도)
- 우리: 엣지 방향 × 길이로 면 벡터 근사. Gen-FVGN: 메시에서 올바른 셀-면 표면 벡터 사용
- Gen-FVGN의 `torch.mean(torch.log(loss))` — 아웃라이어 잔차 영향 감소

### 1B. NVIDIA PhysicsNeMo — 가장 완성도 높은 프레임워크

**Repos**: [NVIDIA/physicsnemo-sym](https://github.com/NVIDIA/physicsnemo-sym)

**기울기 (least squares)**:
- 가중치: `w = 1/(||dv||^2 + eps)` (eps=1e-8, 우리 `1/r^2`와 동일 계열)
- 조립: `A = sum(dv⊗dv * w²)`, `B = sum(dv * w² * du)`
- 풀이: `torch.linalg.lstsq(A + λI, B)` (λ=1e-6 정규화) — **우리보다 수치적으로 강건**

**물리 잔차 (Stokes 예시)**:
```python
phy_informer = PhysicsInformer(
    required_outputs=["continuity", "momentum_x", "momentum_y"],
    equations=node_pde, grad_method="least_squares",
)
```

**손실 가중치 (Stokes MeshGraphNet fine-tuning)**:
```
loss = 1*loss_u + 1*loss_v + 1*loss_p
     + 10*loss_u_in + 10*loss_v_in + 10*loss_u_noslip + 10*loss_v_noslip
     + 1*loss_mom_u + 1*loss_mom_v + 10*loss_cont
```
연속 10배, BC 10배 > 데이터/운동량. 고정 가중치 사용.

**기울기 방법 지원**: `autodiff`, `least_squares`, `meshless_finite_difference`, `finite_difference`, `spectral`

### 1C. 기타 관련 저장소

| 저장소 | 특징 | 물리 손실 |
|--------|------|----------|
| [mario-linov/graphs4cfd](https://github.com/mario-linov/graphs4cfd) | 다중 스케일 회전 등변 GNN, U-Net 구조 | 없음 (순수 데이터 + BC 패널티) |
| [Extrality/AirfRANS](https://github.com/Extrality/AirfRANS) | 공식 기본선 | 없음 (MSE 또는 표면/체적 가중 MSE) |
| MeshGraphNets (다수 PyTorch 포트) | 15 MP 블록, 잔차 연결 | 없음 (다음 스텝 가속도 MSE) |
| [locuslab/cfd-gcn](https://github.com/locuslab/cfd-gcn) | 미분 가능 PDE 솔버 + GNN 하이브리드 | 솔버 내장 |
| [sungyongs/dpgn](https://github.com/sungyongs/dpgn) | 라플라시안 행렬 곱 `L.mm(h)` | 확산/파동 전용 |
| [echowve/phygnnet](https://github.com/echowve/phygnnet) | 이산 미분 (autograd 대신) | Burgers 방정식 |

### 1D. 손실 균형 저장소

| 저장소 | 방법 |
|--------|------|
| [rbischof/relative_balancing](https://github.com/rbischof/relative_balancing) | ReLoBRaLo — EMA + 랜덤 룩백 + softmax 온도 |
| [levimcclenny/SA-PINNs](https://github.com/levimcclenny/SA-PINNs) | 영역별 학습 가능 적응 가중치 |
| [dr-aheydari/SoftAdapt](https://github.com/dr-aheydari/SoftAdapt) | 손실 변화율 기반 적응 |

## 2. 패턴 비교 요약

### 발산 계산
| 접근 | 방법 | 사용처 |
|------|------|--------|
| 면-법선 플럭스 + scatter | `sum(u_face * n * len) / area` | **우리** (법선 방향 오류), Gen-FVGN (정확) |
| 기울기 기반 | `du/dx + dv/dy` (WLSQ) | Gen-FVGN (비보존), PhysicsNeMo |
| 관련 행렬 | `-(B^T * g)` | OCGNN |

### 기울기 계산
| 접근 | 가중 | 시스템 | 사용처 |
|------|------|--------|--------|
| 가중 평균 | RBF 또는 1/r² | 시스템 풀이 없음 | **우리** |
| WLSQ (풀) | 1/r^n (n=3-5) | `torch.linalg.solve(A,B)` | Gen-FVGN |
| 최소제곱 | 1/r² | `torch.linalg.lstsq(A+λI, B)` | PhysicsNeMo |

### 비차원화
| 접근 | 사용처 |
|------|--------|
| 동적 U_ref (입구/원방) + L_ref | **우리** — 다른 레포보다 정교 |
| 고정 Re | 대부분 PINN |
| 심볼릭 비차원 모듈 | PhysicsNeMo |
| 없음 (차원 유지) | Gen-FVGN, Graphs4CFD |

### 수치 안정성 트릭
| 트릭 | 사용처 |
|------|--------|
| 면적 하한 (median 기반) | **우리** |
| 최소 차수 필터링 | **우리** |
| Huber 손실 | **우리** |
| softplus(ν_t) | **우리** |
| log-space 손실 평균 | Gen-FVGN |
| 정규화 lstsq (λ·I) | PhysicsNeMo |
| 고스트 포인트 경계 | Gen-FVGN |

**결론**: 우리 구현은 대부분의 공개 물리 기반 GNN보다 정교하나, 기울기 정확도(WLSQ 풀 시스템)와 적응형 가중치에서 개선 여지 있음.

---

# T3 — 과학자 (Physics & Numerical Correctness)

## A. 발산 / 연속 방정식

### A1. 법선 방향: **오류** (근본적)

**파일**: `src/physics_loss.py:168-169`

```python
# 현재 (오류): 엣지 접선 방향
nx = dx / length
ny = dy / length
```

2D 정점-중심 유한체적법에서 이중 제어체적의 면 법선은 엣지에 **수직**이어야 합니다.
엣지 벡터 `(dx, dy)`의 수직 외향 법선은 `(dy, -dx)/length`입니다.

코드 주석(L167):
```
# Old: (dy/length, -dx/length) was perpendicular → computed -vorticity, not divergence
```
이 판단이 잘못되었을 가능성이 높습니다. 이전 수직 법선이 적절한 이중 메시에서의 발산 계산에 올바른 선택이었습니다.

**현재 코드가 실제로 계산하는 것**:
$$\text{div}_i \approx \frac{1}{A_i} \sum_j \frac{1}{2}(\mathbf{u}_i + \mathbf{u}_j) \cdot \hat{\mathbf{e}}_{ij} \cdot |\mathbf{e}_{ij}|$$
이는 속도가 엣지 방향과 얼마나 정렬되는지를 측정하는 것으로, 발산 ∇·u가 **아닙니다**.

**단, 중요한 미묘함**: 반경 그래프(radius graph)는 Delaunay 삼각분할이 아니므로, 수직 법선도 기하학적으로 엄밀하지 않습니다. 두 접근 모두 근사이나, 수직 법선이 FVM 발산에 훨씬 가깝습니다.

**참조**: Ferziger & Perić, *Computational Methods for Fluid Dynamics*, Ch. 4; Moukalled et al., *The Finite Volume Method in CFD*, Ch. 8

### A2. 면 중심 속도: 정확
`u_face = 0.5*(u_i + u_j)` — 표준 2차 중심 차분 (Moukalled et al., Eq. 8.37)

### A3. 이중 면적 추정: 의문스러움
`A_i = P_i²/(4π)` — 등주부등식을 등식으로 적용. 원의 면적을 반환하므로 **체계적 과대 추정**.
비등방성 메시(벽면 근처 늘어진 요소)에서 과대 추정이 심각.
→ 발산값이 체계적으로 **과소 추정**됨.

### A4. 엣지 반감 + 부호 누적: **구조적으로 정확**
```python
div = scatter_add(flux, row) - scatter_add(flux, col)
```
보존 특성 올바르게 구현: 내부 면의 플럭스가 한 제어체적에 더해지고 인접 체적에서 빼짐.

## B. 기울기 계산

### B1. 가중 최소제곱: 원리적으로 정확, 단순화됨
$$\frac{\partial f}{\partial x}\bigg|_i \approx \frac{\sum_j w_{ij} (f_j - f_i) \frac{dx_{ij}}{r_{ij}^2}}{\sum_j w_{ij}}$$

정확한 가중 최소제곱은 정규 방정식 풀이 필요:
$$\nabla f_i = \mathbf{M}_i^{-1} \mathbf{b}_i$$
여기서 $\mathbf{M}_i = \sum_j w_{ij} \mathbf{d}_{ij} \mathbf{d}_{ij}^T / r_{ij}^4$.

구현은 $\mathbf{M}_i \approx (\sum_j w_{ij}) \mathbf{I}$로 가정 — 등방성 스텐실에서만 정확.
비등방성 스텐실(벽면 경계층)에서 x-y 교차 결합 손실 → 상당한 오차.

**참조**: Mavriplis, "Revisiting the Least-Squares Procedure" (AIAA-2003-3986)

### B2. RBF 가중치: 합리적
`w = exp(-r²/h²)`, h = mean edge length — 표준 가우시안 RBF 커널.

### B3. 대칭 누적: 내부 노드에서 정확
수학적 검증: 엣지 (i,j)에서 `df`와 `dx` 부호가 모두 반전되므로 곱은 동일.
경계 노드에서는 단측 스텐실 바이어스 — 메시리스 기울기의 표준 한계.

### B4. 가중 합 하한 `clamp_min(1.0)`: 허용 가능
고립 노드(차수 0)에서 안전한 폴백. 소수 이웃 노드에서 기울기 과소 추정 가능하나 실제로 드묾.

## C. 라플라시안

### C1. 2-패스 라플라시안: **과도한 수치 확산 유발**
```python
Laplacian(f) = d/dx(gx) + d/dy(gy)  # gx, gy = gradient(f)
```

첫 기울기 패스가 필드 평활화 → 두 번째 패스가 이미 평활화된 결과를 다시 평활화.
유효 스텐실이 직접 라플라시안보다 훨씬 넓어지며 연산자가 더 확산적.

`f = sin(kx)`에서 진정한 라플라시안은 `-k² sin(kx)`이나, 2-패스는 `|k²|`보다 작은 크기 반환.
→ 고주파 성분(벽면 근처)에서 특히 부정확.

**직접 라플라시안** (Brookshaw 1985, Moukalled et al. Ch. 9):
$$\Delta f_i \approx \frac{\sum_j w_{ij} (f_j - f_i) / r_{ij}^2}{\text{정규화}}$$

### C3. 가변 계수 확산 분해: 형태적으로 정확, 물리적으로 불완전
$$\nabla \cdot (\nu_{\text{eff}} \nabla u) = \nu_{\text{eff}} \Delta u + \nabla \nu_{\text{eff}} \cdot \nabla u$$
곱 법칙에 의한 분해는 대수적으로 정확. 그러나 RANS는 전체 변형률 텐서 필요 (D3 참조).

## D. 운동량 방정식

### D1. 비대칭 대류: **정확하고 잘 동기부여됨**
$$\mathbf{C}_{\text{skew}} = \frac{1}{2}\left[\mathbf{u} \cdot \nabla \mathbf{u} + \nabla \cdot (\mathbf{u} \otimes \mathbf{u})\right]$$
이산 수준에서 운동 에너지 보존 (Morinishi et al., JCP 1998; Kravchenko & Moin, JCP 1997).

### D2. 압력 기울기: **비차원화 하에서 정확**
`p* = p/(ρ·U_ref²)` → `∇p*` (1/ρ 불필요). AirfRANS 데이터가 동역학적 압력(p/ρ)이므로 일관.

### D3. 점성 전치항 누락: **상당한 생략**

RANS 점성 응력:
$$\nabla \cdot \left[\nu_{\text{eff}} (\nabla \mathbf{u} + \nabla \mathbf{u}^T)\right]$$

x-운동량 전체 점성항:
$$\nu_{\text{eff}} \Delta u + 2\frac{\partial \nu_t}{\partial x}\frac{\partial u}{\partial x} + \frac{\partial \nu_t}{\partial y}\left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\right)$$

코드:
$$\nu_{\text{eff}} \Delta u + \frac{\partial \nu_t}{\partial x}\frac{\partial u}{\partial x} + \frac{\partial \nu_t}{\partial y}\frac{\partial u}{\partial y}$$

**누락**: `∂ν_t/∂x · ∂u/∂x + ∂ν_t/∂y · ∂v/∂x`
∇ν_t가 크고 교차 기울기가 강한 영역(박리, 전단층)에서 오차 최대.
OpenFOAM simpleFoam은 `div(nuEff*dev2(T(grad(U))))`로 이 항 포함.

### D4. 부호 규약: **정확**
`res = conv + grad(p) - visc = 0` — 표준 형태, 부호 일관.

## E. 비차원화

| 항목 | 상태 |
|------|------|
| E1. p* = p/U_ref² | **정확** (AirfRANS p가 동역학적 압력 가정) |
| E2. ν_t* = ν_t/(U_ref·L_ref) | **정확** |
| E3. 노드별 U_ref 텐서 | **정확** — AirfRANS x[:,0:2] 자유류 특성이 그래프 내 상수이므로 배치에서 올바르게 작동 |

## F. 수치 안정성

| 항목 | 평가 |
|------|------|
| F1. 정밀도 축적 | scatter_add GPU 비결정성 → 재현성 영향 (정확도 아닌) |
| F2. softplus(ν_t) | **좋은 선택** — abs(), exp()보다 안전 |
| F3. NaN/Inf 경로 | eps 클램핑으로 적절히 보호 |
| F4. area_floor, min_degree | 정규화로서 물리적 정당화 가능 |

## 종합 평가표

| 항목 | 심각도 | 상태 |
|------|--------|------|
| A1. 접선 vs 수직 법선 | **높음** | 오류 |
| A3. 이중 면적 추정 | 중간 | 의문 |
| A4. 반감 + 부호 누적 | 낮음 | 정확 |
| B1. 단순화 최소제곱 | 중간 | 근사 |
| C1. 2-패스 라플라시안 | **높음** | 과도한 확산 |
| D1. 비대칭 대류 | 낮음 | 정확 |
| D2. 압력 기울기 | 낮음 | 정확 |
| D3. 전치항 누락 | **중-높음** | 생략 |
| E3. 노드별 U_ref | 낮음 | 정확 |
| F2. softplus | 낮음 | 적절 |

---

# T4 — 소프트웨어 엔지니어 (Code Structure & Performance)

## 1. 두 손실 클래스 — 어떤 것이 사용되는가?

**`NavierStokesPhysicsLoss`만 프로덕션에서 사용됨.**

- `src/pipeline.py:38`: `from src.physics_loss import NavierStokesPhysicsLoss`
- `src/pipeline.py:197-225`: `build_physics_loss()`가 `NavierStokesPhysicsLoss`만 인스턴스화
- `scripts/train.py:99`: `build_physics_loss()` 호출
- `UnifiedNavierStokesPhysicsLoss` (L1126)는 어디서도 임포트/인스턴스화되지 않음
- `forward()` 시그니처 불일치: `(pred, batch, model)` vs 호출부 `(predictions, targets, data=data, step=step)`

**두 클래스 간 핵심 충돌**:

| 항목 | NavierStokes (사용 중, 오류) | Unified (미사용, 정확) |
|------|----------------------------|----------------------|
| 발산 법선 | `nx=dx/r` (접선, **오류**) | `flux=u*dy-v*dx` (수직, **정확**) |
| 라플라시안 | 2-패스 grad(grad) (**과확산**) | 직접 `2*df/dist²` (**정확**) |
| 대류 | 비대칭 (정확) | 업윈드 (유효) |
| 가변 확산 | 3회 추가 기울기 호출 | 단일 패스 엣지 평균 |

## 2. 발산 법선 버그 확인

**파일**: `src/physics_loss.py:166-174`

```python
# 현재 (오류):
nx = dx / length       # 엣지 접선
ny = dy / length
flux = (u_face[:, 0] * nx + u_face[:, 1] * ny) * length

# 수정안:
nx = dy / length       # 수직 회전
ny = -dx / length
flux = (u_face[:, 0] * nx + u_face[:, 1] * ny) * length
# 단순화: flux = u_face[:, 0] * dy - u_face[:, 1] * dx
```

## 3. scatter_add 전체 카운트: 88회

**순전파 호출 체인 추적** (`use_skew=True` 기준):

| 구성 요소 | 호출 | scatter_add |
|-----------|------|-------------|
| 연속 (`conservative_divergence`) | 1 | 2-4 |
| 기울기 (u, v, p) | 3 × `weighted_gradient` | 18 |
| 비대칭 대류 (u², v², uv×2) | 4 × `weighted_gradient` | 24 |
| 라플라시안 (u, v) | 2 × `weighted_laplacian` (각 3 기울기) | 36 |
| ν_t 기울기 | 1 × `weighted_gradient` | 6 |
| **합계** | | **~88** |

## 4. 중복 `weighted_gradient(uv)` 호출

**파일**: `src/physics_loss.py:692-697`

```python
_, duvdy = weighted_gradient(uv, ...)   # L692: duvdx 계산 후 버림
duvdx, _ = weighted_gradient(uv, ...)   # L694: duvdy 계산 후 버림 → 동일 계산 반복!
```

**수정**: `duvdx, duvdy = weighted_gradient(uv, ...)` — 3줄 수정, scatter_add 6회 절약.

## 5. `_extract_dxdy_length` 스키마 감지 신뢰성

**파일**: `src/physics_loss.py:48-85`

```python
col1 = edge_attr[:, 1].abs().median()                    # O(E log E) 정렬!
is_default = (col1 <= 1.5) and torch.all(edge_attr[:, 0] > 0)
```

**신뢰성 문제**:
1. `[dx, dy, dist]` 스키마에서 `dx`가 모두 양수면 `[dist, dir_x, dir_y]`로 오분류 가능
2. `dy` median이 1.5 이하면 (정규화 좌표) 스키마 오분류 가능
3. `torch.median()` (전체 정렬) × **순전파 당 11회** = 상당한 오버헤드

**실제 데이터 흐름**:
- `edge_attr_dxdy`는 `[dx, dy, dist]`로 저장됨 (`edge_construction.py:382`)
- 물리 손실은 `getattr(data, 'edge_attr_dxdy', ...)` 사용 → `[dx, dy, dist]` 스키마

**수정**: 그래프당 1회 감지 후 캐싱, 또는 스키마를 데이터 속성으로 명시.

## 6. 반복되는 엣지 전처리

`weighted_gradient` 및 `conservative_divergence` 각 호출마다 독립적으로:
1. `_valid_edges(edge_index, N)` — 부울 마스크 생성
2. 마스크로 엣지 필터링
3. `_half_edges` — 또 다른 부울 마스크 (`row < col`)
4. 다시 필터링

`_momentum_loss`에서 **동일 입력에 10회** 반복.

**수정**: `_momentum_loss` 시작 시 1회 전처리:
```python
edge_index_h, edge_attr_h = _prepare_edges(edge_index, edge_attr, num_nodes)
```

## 7. 메모리 압력

`_momentum_loss`에서 동시에 유지되는 중간 텐서:
- 기울기: `dudx, dudy, dvdx, dvdy, dpdx, dpdy, dnutdx, dnutdy` (8 × N)
- 대류 중간값: `u², v², uv, du2dx, duvdy, duvdx, dv2dy` (7 × N)
- 라플라시안 내부: 각 3개 기울기 텐서 × 2 = 6 × N
- 엣지 텐서: 각 `weighted_gradient` 내부 8개 (E크기)

N=20K, E=200K, float32 기준:
- 88 × 200K × 4B = **~70MB** 엣지 텐서만

## 8. 수치 정밀도

| 항목 | 평가 |
|------|------|
| `eps=1e-12` vs float32 | 대부분 안전. `length²=1e-12`에서 `1/2e-12=5e11` 근접 |
| `clamp_min(1.0)` 분모 | 기울기 불연속이나 영향 노드 극소 |
| AMP bf16 + 물리 손실 | 모델 출력 bf16 가능 → 물리 손실 정밀도 혼합 문제 |
| 디버그 코드 | `debug=False` 시 오버헤드 없음 (`torch.no_grad()` 내부) |

## 9. Autograd 그래프 최적화 — 분리 가능 텐서

| 텐서 | 분리 필요? | 이유 |
|------|-----------|------|
| RBF 가중치 `w` | 불필요 | 이미 기하 텐서에서 유래 (비학습) |
| `Uref_local` | 권장 | `data.x` 역정규화에서 유래, 방어적 분리 |
| `mol_coeff` | 권장 | 물리 상수, 학습 대상 아님 |
| `node_area`, BC 마스크 | 불필요 | `@torch.no_grad()` 하에서 전처리됨 |

## 10. 우선순위별 수정 권장 (T4 종합)

| 우선순위 | 수정 | 파일:줄 | 노력 |
|----------|------|---------|------|
| P0 | 발산 법선 수정 | physics_loss.py:168-169 | 2줄 |
| P0 | 잔차 비차원화 | physics_loss.py:655,719 | 중간 |
| P1 | 2-패스 → 직접 라플라시안 | physics_loss.py:267-289 | 중간 |
| P1 | 중복 gradient(uv) 병합 | physics_loss.py:692-697 | 3줄 |
| P1 | 노드별 RBF 대역폭 | physics_loss.py:247 | 중간 |
| P2 | 전치항 추가 | physics_loss.py:712-713 | 작음 |
| P2 | 엣지 전처리 캐싱 | 전체 | 중간 |
| P2 | Huber delta 수정 | physics_loss.py:369, pipeline.py:223 | 사소 |
| P3 | RBF 가중치 사전계산 | physics_loss.py:247-248 | 중간 |
| P3 | 스키마 감지 캐싱 | physics_loss.py:64-65 | 작음 |
| P3 | `Uref_local`, `mol_coeff` 분리 | physics_loss.py:949,960 | 사소 |
| P4 | `UnifiedNavierStokesPhysicsLoss` 정리/통합 | physics_loss.py:1120-1919 | 정리 |

---

# T5 — ML 엔지니어 (Machine Learning Optimization)

## 1. 손실 경관 및 기울기 흐름

### 계산 그래프 깊이
운동량 손실의 기울기 흐름:
```
mom_loss → res_u → conv_u (u*dudx 곱) → weighted_gradient 출력
→ scatter_add → field[col]-field[row] → pred_scaled
→ 역정규화 → predictions
```
6-8 연산 깊이, 라플라시안은 두 배.

**기울기 소실 위험**: 모든 연산이 요소별 또는 scatter (역방향에서 선형) → 소실 가능성 낮음.
**기울기 폭발 위험**: 벽면 근처 `inv_r² = 1/(length² + eps)`가 매우 작은 엣지에서 매우 큼.

**현재 안전망**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` (`training.py:169`)

### 계산 가능성
~66-88 scatter 연산 = MSE 대비 **3-5배** 벽시계 시간 증가.
병목은 메모리: 모든 중간 텐서가 역전파를 위해 유지.

## 2. 손실 스케일 불일치 — 가장 심각한 문제

### 실제 스케일 추적

| 손실 항 | 전형적 크기 | 기본 가중치 | 가중 기여 |
|---------|------------|------------|----------|
| MSE (정규화 공간) | O(0.01-1.0) | 1.0 | O(0.01-1.0) |
| 연속 (div²) | O(0.1-10) | 0.0→0.15 | O(0.015-1.5) |
| **운동량** (res²) | **O(100-10,000)** | 0.0→0.20 | **O(20-2,000)** |
| BC | O(0.01-1.0) | 0.01 | O(0.0001-0.01) |

**운동량이 데이터 손실보다 2-4자릿수 큼.**
목표 가중치 0.20이면 `0.20 × O(10000) = O(2000)` → 데이터 피팅 완전 파괴.

### 해결 방안 비교

| 방법 | 구현 | 효과 |
|------|------|------|
| **잔차 비차원화** | `res /= (U²/L)` | 즉시 스케일 정렬 |
| **불확실성 가중치** | 3 학습 `nn.Parameter` | 자동 적응, 구현 간단 |
| **ReLoBRaLo** | EMA 기반 상대 균형 | 기울기 계산 불필요 |
| **GradNorm** | 과제별 기울기 노름 균형 | 이론적이나 비용 높음 |

### 불확실성 가중치 구현 예시
```python
# __init__:
self.log_var_cont = nn.Parameter(torch.zeros(1))
self.log_var_mom = nn.Parameter(torch.zeros(1))
self.log_var_bc = nn.Parameter(torch.zeros(1))

# forward:
total = (
    self.data_w * mse_loss
    + torch.exp(-self.log_var_cont) * cont_loss + self.log_var_cont
    + torch.exp(-self.log_var_mom) * mom_loss + self.log_var_mom
    + torch.exp(-self.log_var_bc) * bc_loss + self.log_var_bc
)
```

## 3. 역정규화 및 Autograd

| 연산 | 기울기 특성 |
|------|------------|
| `pred_phys = pred * scale + mean` | 아핀 변환, 잘 동작 |
| `softplus(ν_t_phys)` | sigmoid 기울기. mean_nut > 0이면 ~1, 매우 음수면 ~0 (사실상 dead) |
| `pred_scaled / U_ref` | U_ref는 입력 데이터 (비학습) → 분리됨, 문제 없음 |
| `clamp(min=0.05)` | 최대 20배 증폭 제한, 허용 |

## 4. 물리 손실의 정규화 효과

### 효과 조건
1. 물리 잔차가 데이터 손실에 **보완적** 기울기 신호 제공
2. 수치 이산화 오차의 노이즈보다 물리 신호가 강해야 함

### 현재 문제
벽면 근처(h~1e-3)에서 메시리스 RBF 라플라시안의 이산화 오차가 O(1).
→ 물리 손실이 **수치 아티팩트**를 학습하도록 훈련할 위험.

**권장**: 벽면 근처 물리 잔차에 **공간 하향 가중**. 또는 충분한 메시 정규성을 가진 내부 노드에서만 물리 손실 계산.

### 커리큘럼 불안정
epoch 50에서 물리 손실 도입 시:
- 모델이 이미 합리적 데이터 피팅 솔루션에 수렴
- 갑자기 O(10000)×0.05 = O(500) 운동량 손실 추가
- 데이터 손실 O(0.01) 완전 압도 → **재앙적 망각**

**권장**:
1. **epoch 0부터** 매우 작은 가중치로 물리 손실 시작
2. **손실 크기 인식 가중**: `w_phys = α / (phys_loss.detach() + ε)`
3. **대류항 stop-gradient**: `u.detach() * dudx` — 기울기 경쟁 방지

## 5. RBF 대역폭 단일 문제

```python
h2 = (length.mean() ** 2)  # 모든 엣지의 글로벌 평균
```

| 영역 | 엣지 길이 | w = exp(-r²/h²) | 문제 |
|------|-----------|-----------------|------|
| 벽면 | 0.001 | ~1.0 | 모두 동일 가중 → 경계층 과평활화 |
| 중간 | 0.05 | ~0.78 | 적절 |
| 원방 | 0.2 | ~0.018 | 거의 0 → 소수 이웃만 사용, 분산 증가 |

**배치 시 추가 문제**: `length.mean()`이 배치 내 모든 그래프 평균 → 다른 메시 스케일 간 간섭.

**권장**: 노드별 대역폭 또는 `weight_mode="inv_r2"` (자연적 스케일 적응, 연산 저렴)

## 6. 엣지 반감 대칭 누적 — 정확 확인

수학적 검증:
- 노드 i (row): `+w*(f_j-f_i)*(x_j-x_i)/r²` — 표준 RBF 기울기. **정확**.
- 노드 j (col): `w*(-(f_j-f_i))*(-(x_j-x_i))/r² = w*(f_j-f_i)*(x_j-x_i)/r²` — **동일 부호**.

**결론**: 대칭 누적은 수학적으로 올바름.

## 7. 커리큘럼 일정 대안

| 방법 | 핵심 | 적용성 |
|------|------|--------|
| **GradNorm** (Chen 2018) | 과제별 기울기 노름 균형 | 높음 — 스케일 불일치 해결 |
| **불확실성 가중치** (Kendall 2018) | 학습 로그-분산 | 높음 — 간단 |
| **ReLoBRaLo** (Bischof 2021) | 랜덤 룩백 EMA | **매우 높음** — 기울기 무비용 |
| **Causal PINN** (Wang 2022) | 시간적 인과 가중 | 정상 상태에 간접 적용 |
| **NTK 가중** (Wang 2022) | NTK 고유값 기반 | 이론적 최적이나 비용 높음 |

## 8. Huber 손실

`huber_delta=0.05`, 잔차 O(100) → **100%가 선형 구간**.
기울기 = `sign(r) × 0.05` (상수) → 잔차 크기 비례 정보 상실.

**권장**: 수렴 잔차 스케일에 맞춰 `delta=0.5-1.0`. 또는 적응형 `delta = percentile(|r|, 90)`.

## 9. 배치 물리 — 교차 그래프 오염

| 항목 | 상태 |
|------|------|
| 노드별 U_ref | 문제 없음 (그래프 내 상수) |
| scatter 연산 | 문제 없음 (PyG 배치에서 교차 엣지 없음) |
| mol_coeff 노드별 | 문제 없음 |
| **RBF 대역폭** | **문제**: `length.mean()` 모든 그래프 평균 |

## 10. 효율 개선 방안

| 방안 | 절약 | 난이도 |
|------|------|--------|
| N스텝마다 물리 손실 계산 | 2-4배 비용 분산 | 쉬움 |
| 물리 노드 서브샘플링 (30-50%) | 2-3배 물리 계산 속도 | 쉬움 |
| 라플라시안 `.detach()` | 역방향 그래프 깊이 절반 | 쉬움 |
| RBF 가중치 사전계산 | 11 exp() 제거 | 중간 |
| `weighted_gradient` 벡터화 | 4→1 호출 (scatter 12→3) | 중간 |

### 벡터화 예시
```python
# 개별 호출 4회 대신:
fields = torch.stack([u, v, p, nu_t], dim=1)  # [N, 4]
df = fields[col] - fields[row]                 # [E, 4]
gx_edge = w.unsqueeze(1) * df * (dx * inv_r2).unsqueeze(1)  # [E, 4]
# 단일 scatter로 4개 필드 동시 처리
```

---

# 팀 리더 — 통합 결론 및 실행 계획

## 핵심 발견 요약

### 모든 팀원이 확인한 3대 치명적 문제

1. **발산 법선 방향** — T3(물리), T4(코드), T2(Gen-FVGN 비교) 모두 확인.
   엣지 접선을 사용하여 연속 손실이 물리적으로 의미 없음.

2. **손실 스케일 불일치** — T5(ML), T4(코드) 확인.
   운동량 O(10000) vs 데이터 O(0.01). 비차원화 누락.
   HPO 실행 결과에서도 확인: `mom_weight: 4.07e-06` (최적화기가 스스로 극도로 낮은 가중치 선택).

3. **2-패스 라플라시안** — T3(물리), T4(코드) 확인.
   과도한 수치 확산, 88 scatter_add 중 36회 (41%) 차지.

### HPO 결과와의 상관관계

현재 실행 중인 HPO 최적 파라미터에서 주목할 점:
```
cont_weight: 2.57e-05     # 극도로 낮음
mom_weight: 4.07e-06      # 거의 0
cont_target: 2.74e-04     # 여전히 매우 낮음
mom_target: 1.81e-05      # 거의 0
```
→ **HPO가 물리 손실의 문제를 스스로 발견**: 스케일 불일치로 인해 물리 가중치를 거의 0으로 설정해야 최적 성능 달성. 이는 물리 손실이 현재 상태에서 **해롭다**는 것을 의미.

## 실행 계획

### Phase 1: 긴급 수정 (즉시)
1. 발산 법선 수정 (2줄)
2. 잔차 비차원화 추가
3. 중복 gradient(uv) 병합 (3줄)

### Phase 2: 정확도 개선
4. 직접 단일 패스 라플라시안
5. 점성 전치항 추가 (추가 기울기 호출 없음)
6. RBF 대역폭 수정 (inv_r2 또는 노드별)

### Phase 3: 훈련 안정성
7. ReLoBRaLo 또는 불확실성 가중치 구현
8. Huber delta 수정
9. 엣지 전처리 캐싱

### Phase 4: 성능 최적화
10. weighted_gradient 벡터화
11. RBF 가중치 사전계산
12. UnifiedNavierStokesPhysicsLoss 정리/통합

---

*보고서 작성: 2026-03-08*
*팀: T1(연구자), T2(레포 스카우터), T3(과학자), T4(소프트웨어 엔지니어), T5(ML 엔지니어)*
