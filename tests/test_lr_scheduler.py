"""Tests for LR scheduler factory used by dashboard training."""

import torch
import pytest

from dashboard.runner import build_lr_scheduler


@pytest.fixture
def dummy_optimizer():
    model = torch.nn.Linear(1, 1)
    return torch.optim.SGD(model.parameters(), lr=0.01)


class TestBuildLrScheduler:
    def test_constant_returns_none(self, dummy_optimizer):
        sched = build_lr_scheduler(dummy_optimizer, {"scheduler_type": "Constant"})
        assert sched is None

    def test_cosine_annealing(self, dummy_optimizer):
        cfg = {"scheduler_type": "CosineAnnealingLR", "scheduler_T_max": 50, "scheduler_eta_min": 1e-6}
        sched = build_lr_scheduler(dummy_optimizer, cfg)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        # After T_max steps, LR should reach eta_min
        for _ in range(50):
            sched.step()
        assert dummy_optimizer.param_groups[0]["lr"] == pytest.approx(1e-6, abs=1e-8)

    def test_step_lr(self, dummy_optimizer):
        cfg = {"scheduler_type": "StepLR", "scheduler_step_size": 5, "scheduler_gamma": 0.5}
        sched = build_lr_scheduler(dummy_optimizer, cfg)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)
        for _ in range(5):
            sched.step()
        assert dummy_optimizer.param_groups[0]["lr"] == pytest.approx(0.005)

    def test_reduce_on_plateau(self, dummy_optimizer):
        cfg = {"scheduler_type": "ReduceLROnPlateau", "scheduler_factor": 0.5,
               "scheduler_patience": 2, "scheduler_min_lr": 1e-6}
        sched = build_lr_scheduler(dummy_optimizer, cfg)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_cosine_warm_restarts(self, dummy_optimizer):
        cfg = {"scheduler_type": "CosineAnnealingWarmRestarts",
               "scheduler_T_0": 10, "scheduler_T_mult": 2, "scheduler_eta_min": 0.0}
        sched = build_lr_scheduler(dummy_optimizer, cfg)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    def test_default_is_cosine(self, dummy_optimizer):
        sched = build_lr_scheduler(dummy_optimizer, {})
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_unknown_type_raises(self, dummy_optimizer):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_lr_scheduler(dummy_optimizer, {"scheduler_type": "FooBar"})


import json

from dashboard.app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


class TestLrPreviewEndpoint:
    def test_cosine_preview(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "CosineAnnealingLR",
            "scheduler_T_max": 20,
            "scheduler_eta_min": 0.0,
            "base_lr": 0.001,
            "num_epochs": 20,
        }), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["epochs"]) == 20
        assert len(data["lr"]) == 20
        assert data["lr"][0] == pytest.approx(0.001)
        assert data["lr"][-1] < data["lr"][0]  # LR decreased

    def test_constant_preview(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "Constant",
            "base_lr": 0.01,
            "num_epochs": 10,
        }), content_type="application/json")
        data = resp.get_json()
        assert all(lr == pytest.approx(0.01) for lr in data["lr"])

    def test_plateau_preview_worst_case(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "scheduler_min_lr": 1e-6,
            "base_lr": 0.01,
            "num_epochs": 20,
        }), content_type="application/json")
        data = resp.get_json()
        # LR should have decreased after patience epochs
        assert data["lr"][-1] < data["lr"][0]
