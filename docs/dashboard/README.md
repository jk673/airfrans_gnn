# Integrated Training Dashboard

브라우저에서 학습 설정, 실행, 모니터링, 실험 기록 조회를 한번에 처리하는 통합 대시보드.

## Quick Start

```bash
python dashboard/app.py              # http://localhost:5000
python dashboard/app.py --port 8080  # 커스텀 포트
```

## Architecture

```
Browser (localhost:5000)
  ├── Tab 1: Config Panel → POST /api/start (config JSON)
  ├── Tab 2: Training Charts ← poll GET /api/status (3초 간격)
  └── Tab 3: Experiments ← GET /api/experiments

Flask (dashboard/app.py)
  └── TrainingSession (dashboard/runner.py)
        ├── load_airfrans_data()
        ├── convert_to_pyg()
        ├── build_model() + build_physics_loss()
        ├── Trainer.fit(on_epoch_end=callback)  ← 매 epoch 상태 업데이트
        └── run_benchmark_and_log_experiment()  ← 완료 후 자동 벤치마크
```

## File Structure

```
dashboard/
  __init__.py
  app.py              # Flask 서버 + API 라우트
  runner.py            # 백그라운드 학습 스레드 관리
  templates/
    index.html         # SPA (Config + Charts + Experiments 3탭)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | SPA 메인 페이지 |
| GET | `/api/config` | Config 기본값 JSON (타입/옵션 메타데이터 포함) |
| POST | `/api/start` | config JSON → 학습 시작. 이미 실행 중이면 409 |
| POST | `/api/stop` | 현재 epoch 완료 후 조기 종료 요청. 미실행 시 400 |
| GET | `/api/status` | 현재 학습 상태 + 메트릭 (프론트엔드 3초 폴링) |
| GET | `/api/experiments` | baselines + 전체 실험 목록 (FLOW-GLIDE 메트릭 포함) |
| GET | `/api/experiments/<id>` | 단일 실험 상세 JSON |

## Frontend (3 Tabs)

### Tab 1 — Config

모든 `scripts/main.py`의 `Config` dataclass 필드를 그룹별로 제공:

| Group | Fields |
|-------|--------|
| Data | task (select), seed, batch_size, num_workers |
| Model | hidden_dim, num_layers, num_global_tokens, dropout |
| Physics | continuity/momentum/bc 각각 weight, target, ramp_start_epoch, ramp_epochs |
| Optimizer | lr, weight_decay |
| Scheduler | scheduler_T_max |
| Training | num_epochs, device (select), amp (checkbox) |
| Experiment | notes (textarea) |

Start Training / Stop 버튼으로 학습 제어.

### Tab 2 — Training

- **Status bar**: 상태 뱃지 + epoch/best val/elapsed time 실시간 표시
- **6개 차트** (3x2 grid): Total Loss, MSE Loss, Continuity, Momentum, BC Loss, Learning Rate
- Train(blue, `#3b82f6`) / Val(orange, `#f97316`) / LR(purple, `#a855f7`)
- Chart.js + chartjs-plugin-zoom: 스크롤 줌, 드래그 팬, 더블클릭 리셋
- 학습 시작 시 자동으로 Training 탭 전환

### Tab 3 — Experiments

- FLOW-GLIDE 비교 테이블 (baseline + 내 실험)
- 6개 메트릭 컬럼: Volume Rel.L2, Surface Rel.L2, CD Rel.Err, CL Rel.Err, rho_D, rho_L
- 추가 정보: task, hidden dim, layers, parameters, notes
- 행 클릭 시 상세 패널 (config, duration, full metrics)
- 학습 완료 시 자동 새로고침

## Backend Design

### TrainingSession (runner.py)

단일 학습 세션을 daemon thread에서 관리:

```
TrainingState (dataclass)
  state: idle | loading | training | benchmarking | completed | failed | stopping
  session_id, current_epoch, total_epochs
  best_val, best_epoch, elapsed_sec
  metrics: {epochs: [], train: {loss_key: [...]}, val: {...}, lr: []}
  config, error_message, experiment_id
```

- **Thread safety**: `threading.Lock`으로 상태 접근, `threading.Event`로 조기 종료 시그널
- **학습 파이프라인**: `scripts/main.py`와 동일한 흐름을 `src/pipeline` 함수들로 재사용
- **콜백**: `Trainer.fit(on_epoch_end=callback)` — 매 epoch 끝에서 상태 업데이트 + 조기종료 체크

### Trainer.fit() Callback (src/pipeline.py 수정)

기존 `Trainer.fit()`에 `on_epoch_end` 파라미터 추가:

```python
def fit(self, train_loader, val_loader, num_epochs, routine, on_epoch_end=None):
    ...
    if on_epoch_end is not None:
        should_stop = on_epoch_end(epoch=epoch, train_logs=train_logs,
                                    val_logs=val_logs, lr=lr, is_best=is_best)
        if should_stop:
            break
```

기존 `scripts/main.py`와 완전 호환 (콜백 미전달 시 기존 동작).

## Edge Cases

| Scenario | Handling |
|----------|----------|
| 학습 중 Start 클릭 | 409 에러 + "이미 실행 중" 메시지 |
| 브라우저 새로고침 | daemon thread에서 학습 계속, 페이지 로드 시 pollStatus()로 복원 |
| 서버 재시작 | daemon thread 종료, 완료된 EXP_*.json은 보존 |
| CUDA OOM | try/except → state="failed", error_message 표시 |
| 데이터 미존재 | state="failed" + 에러 메시지 |

## Theme

기존 `docs/experiments/dashboard.html`과 동일한 다크 테마:

- Background: `#1a1a2e`, Card: `#16213e`
- Text: `#eee`, Accent: `#4ecca3`
- Font: `'Courier New', monospace`
