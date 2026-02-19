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
