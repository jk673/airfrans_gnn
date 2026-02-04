# Quick Start Guide - Python Scripts

이제 모든 노트북이 실행 가능한 Python 스크립트로 변환되었습니다.

## 🚀 빠른 시작

### 1. 기본 학습 실행

```bash
# 기본 설정으로 학습 시작
python scripts/train.py

# 커스텀 설정으로 학습
python scripts/train.py \
    --batch-size 4 \
    --epochs 200 \
    --lr 3e-4 \
    --hidden 256 \
    --wandb-name my-experiment
```

### 2. Multi-Scale 모델 학습

```bash
python scripts/train_multiscale.py \
    --batch-size 4 \
    --epochs 200 \
    --num-scales 4 \
    --num-multiscale-layers 4
```

### 3. 하이퍼파라미터 최적화

```bash
python scripts/optuna_hpo.py \
    --study-name my-hpo \
    --storage sqlite:///hpo.db \
    --n-trials 50 \
    --viz-dir visualizations
```

### 4. 모델 평가

```bash
python scripts/eval_cp.py \
    --checkpoint checkpoints/best.pt \
    --split test \
    --surface-only \
    --verbose
```

## 📁 새로 생성된 파일들

### 핵심 모듈 (src/)
- **src/trainer.py** - 학습 오케스트레이션 (16K)
- **src/visualization.py** - 시각화 함수들 (10K)
- **src/metrics.py** - 평가 지표 (13K)
- **src/prediction.py** - 예측 헬퍼 함수들 (9K)

### 실행 스크립트 (scripts/)
- **scripts/train.py** - 메인 학습 스크립트 (34K, 40+ CLI args)
- **scripts/train_multiscale.py** - Multi-scale 모델 학습 (20K, 44+ CLI args)
- **scripts/eval_cp.py** - Cp 상대 L2 오차 평가 (11K, 15+ CLI args)
- **scripts/optuna_hpo.py** - 하이퍼파라미터 최적화 (27K, 30+ CLI args)

### 문서 (scripts/)
- **README.md** - 메인 학습 스크립트 문서
- **EXAMPLES.md** - 23개 학습 예제
- **README_MULTISCALE.md** - Multi-scale 변형 문서
- **README_OPTUNA_HPO.md** - HPO 종합 가이드
- **EXAMPLES_OPTUNA.md** - 12개 HPO 예제
- **OPTUNA_CHEATSHEET.md** - 빠른 참조

## 🎯 주요 개선사항

### ✅ 완전한 모듈화
- 모든 공통 기능이 `src/` 모듈로 추출됨
- 코드 중복 제로 (각 함수는 한 곳에만 존재)
- 재사용성 최대화

### ✅ CLI 인터페이스
- 모든 스크립트가 완전한 CLI 지원
- `--help`로 모든 옵션 확인 가능
- 하드코딩된 값 없음

### ✅ 가독성 최대화
- 명확한 함수/변수 이름
- 완전한 docstring
- Type hints (Python 3.10+)
- 논리적 구조

### ✅ 원본 개념 보존
- 모델 아키텍처 변경 없음
- 전처리 방법 변경 없음
- Physics loss 공식 변경 없음
- 학습 로직 변경 없음

## 📖 사용 예제

### 빠른 테스트
```bash
# 5 epoch 빠른 테스트
python scripts/train.py --epochs 5 --wandb-mode disabled
```

### 프로덕션 학습
```bash
# 완전한 학습 실행
python scripts/train.py \
    --epochs 200 \
    --batch-size 4 \
    --lr 3e-4 \
    --hidden 256 \
    --layers 16 \
    --amp \
    --wandb-name production-run \
    --ckpt-dir checkpoints/prod
```

### 물리 손실 튜닝
```bash
python scripts/train.py \
    --continuity-target-weight 0.3 \
    --momentum-target-weight 0.3 \
    --ramp-start-epoch 30 \
    --ramp-epochs 70
```

### HPO 실행 후 최적 모델 학습
```bash
# 1. HPO 실행
python scripts/optuna_hpo.py \
    --n-trials 50 \
    --save-best-config best_config.json

# 2. 최적 하이퍼파라미터로 학습 (수동으로 적용)
python scripts/train.py \
    --hidden 256 \
    --lr 2.5e-4 \
    --weight-decay 0.015 \
    # ... (best_config.json에서 확인한 값들)
```

## 🔍 도움말 보기

각 스크립트의 모든 옵션을 확인하려면:

```bash
python scripts/train.py --help
python scripts/train_multiscale.py --help
python scripts/eval_cp.py --help
python scripts/optuna_hpo.py --help
```

## 📊 노트북과의 비교

| 기능 | 노트북 | 스크립트 |
|------|--------|----------|
| 코드 중복 | 많음 (3-4개 노트북에 중복) | 없음 (모듈화) |
| CLI 지원 | ❌ | ✅ 40+ args |
| 재현성 | 어려움 | 쉬움 (CLI로 정확한 설정 공유) |
| 버전 관리 | 어려움 (출력 포함) | 쉬움 (깔끔한 diff) |
| CI/CD | 불가능 | 가능 |
| 디버깅 | 제한적 | 표준 Python 도구 사용 가능 |
| 성능 | Jupyter 오버헤드 | 오버헤드 없음 |
| 문서화 | 노트북 내부 | 47K 외부 문서 |

## 📚 더 많은 정보

- **전체 마이그레이션 내역**: `SCRIPTS_MIGRATION.md`
- **학습 예제**: `scripts/EXAMPLES.md`
- **HPO 예제**: `scripts/EXAMPLES_OPTUNA.md`
- **빠른 참조**: `scripts/OPTUNA_CHEATSHEET.md`

## 🎓 노트북은 여전히 유효

노트북은 대화형 탐색을 위해 유지됩니다:
- 결과 시각화 및 분석
- 프로토타이핑
- 디버깅
- 교육 목적

하지만 **재현 가능한 실험과 프로덕션 워크플로우**에는 스크립트를 사용하세요!
