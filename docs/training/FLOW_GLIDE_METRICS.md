# FLOW-GLIDE 성능 지표 정의 (`volume_rel_l2`, `surface_rel_l2`, `cd_relative_error`, `cl_relative_error`, `rho_D`, `rho_L`)

`scripts/train.py` 종료 시 `run_benchmark_and_log_experiment()`가 `scripts/score_benchmark.py`의
`score_test_set()`을 호출해 아래 6개 지표를 계산합니다. 저장 위치는 `docs/experiments/results/*.json`이며,
`Flow-Glide` 비교표(`docs/experiments/flow_glide_comparison_table.md`)에는 baseline과 함께 누적됩니다.

## 1) 볼륨 상대 L2 (`volume_rel_l2`)

- 한글명: Volume Rel. L₂ (작을수록 좋음, ↓)
- 계산 대상: 전체 노드의 4개 채널 동시 비교 (`u`, `v`, `p/ρ`, `ν_t`)
- 계산식:

```text
all_err = pred_phys - gt_phys
volume_rel_l2 = sqrt( sum(all_err^2) / sum(gt_phys^2) )
```

- 구현 위치: `scripts/score_benchmark.py`
  - `all_err = (pred_phys - targ_phys)`
  - `vol_sum_err2 += float((all_err ** 2).sum().item())`
  - `vol_sum_gt2 += float((targ_phys ** 2).sum().item())`
  - `sqrt(vol_sum_err2 / (vol_sum_gt2 + eps))`

## 2) 표면 상대 L2 (`surface_rel_l2`)

- 한글명: Surface Rel. L₂ (작을수록 좋음, ↓)
- 계산 대상: 표면 노드의 압력 채널(`p/ρ`, index=2)만
- 표면 노드 판정:
  - `surface_mask_from_x_phys(x_phys)` 사용
  - `wall_dist < 1e-6` 또는 wall normal(`nx, ny`)의 절댓값 합이 0보다 큰 경우를 표면으로 간주
- 계산식:

```text
surface_rel_l2 = sqrt( sum((p_pred_surf - p_gt_surf)^2) / sum(p_gt_surf^2) )
```

- 구현 위치: `scripts/score_benchmark.py`
  - `surf_mask = surface_mask_from_x_phys(x_phys)`
  - `p_pred_surf = pred_phys[surf_mask, 2]`
  - `p_gt_surf = targ_phys[surf_mask, 2]`

## 3) CD 상대 오차 (`cd_relative_error`)

- 한글명: CD Rel.Err (작을수록 좋음, ↓)
- 계산 방법:
  1. 그래프별로 drag 계수(`CD`)를 실제값/예측값 둘 다 계산
  2. 각 그래프의 상대오차 `|CD_pred - CD_gt| / |CD_gt|`
  3. 테스트 그래프 전체 평균

### Drag 계수 `CD` 계산(`compute_force_coefficients`)

- 구현 위치: `src/metrics.py`
- 핵심 단계:
  1. 표면 노드 추출: wall normal(`x[:,3:5]`)가 0이 아닌 노드
  2. 표면 노드 좌표를 중심 기준 각도로 정렬
  3. 각 패널 길이/벡터 계산 (`ds`, `seg_vec`)
  4. 압력력: `Fp = (-p/ρ * n_hat) * ds`
  5. 총력 `F = sum(Fp)`
  6. 정규화:
     - 동압(`q_ref = 0.5 * U_inf^2`, `U_inf = sqrt(u_inf^2 + v_inf^2)`)
     - 코드 길이(`chord = max(x)-min(x)`)
     - `CD = -F_x / (q_ref * chord)`

- 즉, `CD`는 양의 drag가 흐름 반대 방향이 되도록 부호를 반영합니다.

## 4) CL 상대 오차 (`cl_relative_error`)

- 한글명: CL Rel.Err (작을수록 좋음, ↓)
- 계산 구조는 `cd_relative_error`와 동일:

```text
|CL_pred - CL_gt| / |CL_gt| 를 그래프별로 계산 후 평균
```

- `CL` 역시 `compute_force_coefficients`에서
  - `CL = F_y / (q_ref * chord)`

## 5) Drag rank correlation (`rho_D`)

- 한글명: ρ_D (클수록 좋음, ↑)
- 정의: 그래프별 `CD_gt`와 `CD_pred`의 Spearman rank correlation
- 조건: 테스트 그래프 3개 미만이면 `nan`
- 구현: `scipy.stats.spearmanr(CD_gt, CD_pred)`

## 6) Lift rank correlation (`rho_L`)

- 한글명: ρ_L (클수록 좋음, ↑)
- 정의: 그래프별 `CL_gt`와 `CL_pred`의 Spearman rank correlation
- 조건/실행: `rho_D`와 동일

## 평가 파이프라인 요약

- `scripts/train.py` 종료 시:
  - `score_test_set()` → 6개 지표 계산
  - `train.py`의 `ExperimentTracker` 저장
  - `docs/experiments/flow_glide_comparison_table.md` 자동 갱신
- 사용 지표를 그대로 `EXP_*.json`(`_benchmark_metric` 키)과 테이블에 기록합니다.

## 참고 코드 위치

- 지표 계산 핵심: `scripts/score_benchmark.py`
- CD/CL 계산: `src/metrics.py`의 `surface_mask_from_x_phys`, `compute_force_coefficients`
- 학습 로그 연동: `scripts/train.py`

