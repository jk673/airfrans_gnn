"""
예시: AirfRANS 2D Airfoil 프로젝트에 ExperimentTracker 통합하기

이 파일은 기존 프로젝트에 tracker를 붙이는 실제 패턴을 보여줍니다.
프로젝트 구조에 맞게 import 경로만 수정하면 바로 사용 가능합니다.
"""

# ──────────────────────────────────────────────────
# 1. 기본 설정
# ──────────────────────────────────────────────────

from experiment_tracker import ExperimentTracker, track_experiment, track_preprocessing

# 프로젝트 루트에 tracker 인스턴스 생성 (보통 train.py 또는 main.py 상단)
tracker = ExperimentTracker(
    log_path="experiments/EXPERIMENT_LOG.md",  # MD 파일 경로
    project_name="AirfRANS 2D Airfoil - GNN Surrogate",
)


# ──────────────────────────────────────────────────
# 2. 전처리 함수에 데코레이터 적용
# ──────────────────────────────────────────────────

@track_preprocessing(tracker, config_param="config")
def preprocess_airfrans(raw_data_path, config):
    """
    AirfRANS 데이터 전처리 — config가 바뀌면 자동으로 MD에 기록됩니다.
    """
    preprocess_config = config  # tracker가 이 dict를 자동 기록

    # === 기존 전처리 코드 그대로 유지 ===
    # dataset = load_airfrans(raw_data_path)
    # normalized = normalize(dataset, config["normalization"])
    # graph_data = build_graphs(normalized, config["k_neighbors"])
    # ...

    # 예시 반환 — shape 메타정보를 같이 넘기면 자동 기록
    processed_data = {"nodes": "...", "edges": "..."}  # 실제 데이터
    meta = {
        "input_shape": "(1000, 5)",    # 예: 1000 nodes, 5 features
        "output_shape": "(1000, 4)",   # 예: pressure, velocity_x/y/z
    }
    return processed_data, meta


@track_preprocessing(tracker, config_param="config")
def augment_data(data, config):
    """데이터 증강 — 별도 전처리 단계로 기록."""
    # rotate, flip, noise injection 등
    return data


# ──────────────────────────────────────────────────
# 3. 학습 함수에 데코레이터 적용
# ──────────────────────────────────────────────────

@track_experiment(tracker, config_param="config", model_param="model")
def train(config, model=None):
    """
    학습 함수 — 실행 시마다 자동으로 MD에 기록됩니다.

    반환값이 dict이면 metrics로 자동 인식.
    반환값이 (result, metrics_dict) tuple이면 두 번째를 metrics로 사용.
    """
    # === 기존 학습 코드 그대로 유지 ===
    # optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # for epoch in range(config["epochs"]):
    #     loss = train_one_epoch(model, train_loader, optimizer)
    #     val_metrics = evaluate(model, val_loader)
    # ...

    # 학습 완료 후 최종 metrics를 dict로 반환하면 자동 기록
    return {
        "train_loss": 0.00234,
        "val_loss": 0.00312,
        "val_mae_pressure": 0.0045,
        "val_mae_velocity": 0.0023,
        "best_epoch": 142,
    }


# ──────────────────────────────────────────────────
# 4. 실행 예시
# ──────────────────────────────────────────────────

if __name__ == "__main__":

    # ─── 전처리 config ───
    preprocess_config = {
        "normalization": "standard",
        "k_neighbors": 15,
        "feature_columns": ["x", "y", "sdf", "nx", "ny"],
        "target_columns": ["p", "ux", "uy", "nut"],
        "train_ratio": 0.8,
        "remove_outliers": True,
    }

    # ─── 학습 config ───
    train_config = {
        "model": "GraphSAGE",
        "hidden_dim": 128,
        "num_layers": 6,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "epochs": 200,
        "batch_size": 4,
        "scheduler": "CosineAnnealingLR",
        "loss_fn": "MSELoss",
        "aggregator": "mean",
    }

    # 1) 전처리 실행 → 자동 기록
    data = preprocess_airfrans("data/airfrans/", config=preprocess_config)

    # 2) 모델 생성
    # model = GraphSAGE(**train_config)  # 실제 모델
    model = None  # 예시에서는 None

    # 3) 학습 실행 → 자동 기록 (config, metrics, model 구조, 소요시간 모두)
    metrics = train(config=train_config, model=model)

    # ─── 수동으로 노트 추가도 가능 ───
    tracker.add_note(
        "Reynolds number 범위: 2M-6M. "
        "High-Re 영역에서 wake 부근 prediction이 약함. "
        "다음 실험에서 attention mechanism 추가 예정.",
        tag="Observation",
    )

    # ─── 모델 아키텍처만 별도 기록 ───
    # tracker.log_model_change(model, "Added attention layer after message passing")

    print("\n✅ 모든 기록이 experiments/EXPERIMENT_LOG.md에 저장되었습니다.")