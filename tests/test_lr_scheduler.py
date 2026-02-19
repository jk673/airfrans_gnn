"""Tests for LR scheduler factory used by dashboard training."""

import json

import torch
import pytest

from unittest.mock import patch, MagicMock

from dashboard.runner import (
    build_lr_scheduler,
    normalize_scheduler_type,
    validate_scheduler_config,
    simulate_lr_schedule,
    flatten_scheduler_config,
    build_scheduler_snapshot,
    ALLOWED_SCHEDULER_TYPES,
)


@pytest.fixture
def dummy_optimizer():
    model = torch.nn.Linear(1, 1)
    return torch.optim.SGD(model.parameters(), lr=0.01)


# ---------------------------------------------------------------------------
# normalize_scheduler_type
# ---------------------------------------------------------------------------

class TestNormalizeSchedulerType:
    @pytest.mark.parametrize("alias,expected", [
        ("cosine", "CosineAnnealingLR"),
        ("COSINE", "CosineAnnealingLR"),
        ("step", "StepLR"),
        ("plateau", "ReduceLROnPlateau"),
        ("reduce_on_plateau", "ReduceLROnPlateau"),
        ("cosine_warm_restarts", "CosineAnnealingWarmRestarts"),
        ("warm_restarts", "CosineAnnealingWarmRestarts"),
        ("constant", "Constant"),
        ("none", "Constant"),
        ("multistep", "MultiStepLR"),
        ("multi_step", "MultiStepLR"),
        ("exponential", "ExponentialLR"),
    ])
    def test_alias_normalizes(self, alias, expected):
        assert normalize_scheduler_type(alias) == expected

    @pytest.mark.parametrize("canonical", list(ALLOWED_SCHEDULER_TYPES))
    def test_canonical_passthrough(self, canonical):
        assert normalize_scheduler_type(canonical) == canonical

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            normalize_scheduler_type("FooBar")

    def test_whitespace_stripped(self):
        assert normalize_scheduler_type("  cosine  ") == "CosineAnnealingLR"


# ---------------------------------------------------------------------------
# validate_scheduler_config
# ---------------------------------------------------------------------------

class TestValidateSchedulerConfig:
    def test_fills_defaults(self):
        cfg = validate_scheduler_config({"scheduler_type": "CosineAnnealingLR"})
        assert cfg["scheduler_T_max"] == 100
        assert cfg["scheduler_eta_min"] == 0.0

    def test_normalizes_type(self):
        cfg = validate_scheduler_config({"scheduler_type": "cosine"})
        assert cfg["scheduler_type"] == "CosineAnnealingLR"

    def test_casts_string_to_int(self):
        cfg = validate_scheduler_config({
            "scheduler_type": "StepLR",
            "scheduler_step_size": "5",
            "scheduler_gamma": "0.5",
        })
        assert cfg["scheduler_step_size"] == 5
        assert isinstance(cfg["scheduler_step_size"], int)

    def test_negative_T_max_raises(self):
        with pytest.raises(ValueError, match="constraint 'positive'"):
            validate_scheduler_config({
                "scheduler_type": "CosineAnnealingLR",
                "scheduler_T_max": -1,
            })

    def test_zero_T_max_raises(self):
        with pytest.raises(ValueError, match="constraint 'positive'"):
            validate_scheduler_config({
                "scheduler_type": "CosineAnnealingLR",
                "scheduler_T_max": 0,
            })

    def test_negative_step_size_raises(self):
        with pytest.raises(ValueError, match="constraint 'positive'"):
            validate_scheduler_config({
                "scheduler_type": "StepLR",
                "scheduler_step_size": -10,
            })

    def test_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="constraint 'unit_range'"):
            validate_scheduler_config({
                "scheduler_type": "StepLR",
                "scheduler_gamma": 0.0,
            })

    def test_gamma_above_one_raises(self):
        with pytest.raises(ValueError, match="constraint 'unit_range'"):
            validate_scheduler_config({
                "scheduler_type": "StepLR",
                "scheduler_gamma": 1.5,
            })

    def test_negative_eta_min_raises(self):
        with pytest.raises(ValueError, match="constraint 'non_negative'"):
            validate_scheduler_config({
                "scheduler_type": "CosineAnnealingLR",
                "scheduler_eta_min": -0.001,
            })

    def test_invalid_type_value_raises(self):
        with pytest.raises(ValueError, match="must be int"):
            validate_scheduler_config({
                "scheduler_type": "CosineAnnealingLR",
                "scheduler_T_max": "not_a_number",
            })

    def test_plateau_defaults(self):
        cfg = validate_scheduler_config({"scheduler_type": "ReduceLROnPlateau"})
        assert cfg["scheduler_factor"] == 0.5
        assert cfg["scheduler_patience"] == 10
        assert cfg["scheduler_min_lr"] == 1e-6

    def test_multistep_milestones_from_string(self):
        cfg = validate_scheduler_config({
            "scheduler_type": "MultiStepLR",
            "scheduler_milestones": "[10, 20, 30]",
        })
        assert cfg["scheduler_milestones"] == [10, 20, 30]

    def test_multistep_milestones_from_list(self):
        cfg = validate_scheduler_config({
            "scheduler_type": "MultiStepLR",
            "scheduler_milestones": [15, 45],
        })
        assert cfg["scheduler_milestones"] == [15, 45]

    def test_exponential_defaults(self):
        cfg = validate_scheduler_config({"scheduler_type": "ExponentialLR"})
        assert cfg["scheduler_gamma"] == 0.95


# ---------------------------------------------------------------------------
# build_lr_scheduler (existing + new types)
# ---------------------------------------------------------------------------

class TestBuildLrScheduler:
    def test_constant_returns_none(self, dummy_optimizer):
        sched = build_lr_scheduler(dummy_optimizer, {"scheduler_type": "Constant"})
        assert sched is None

    def test_cosine_annealing(self, dummy_optimizer):
        cfg = {"scheduler_type": "CosineAnnealingLR", "scheduler_T_max": 50, "scheduler_eta_min": 1e-6}
        sched = build_lr_scheduler(dummy_optimizer, cfg)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
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

    def test_multistep_lr(self, dummy_optimizer):
        cfg = {"scheduler_type": "MultiStepLR", "scheduler_milestones": [5, 10], "scheduler_gamma": 0.5}
        sched = build_lr_scheduler(dummy_optimizer, cfg)
        assert isinstance(sched, torch.optim.lr_scheduler.MultiStepLR)
        # After milestone 5, LR should halve
        for _ in range(5):
            sched.step()
        assert dummy_optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
        # After milestone 10, LR should halve again
        for _ in range(5):
            sched.step()
        assert dummy_optimizer.param_groups[0]["lr"] == pytest.approx(0.0025)

    def test_exponential_lr(self, dummy_optimizer):
        cfg = {"scheduler_type": "ExponentialLR", "scheduler_gamma": 0.9}
        sched = build_lr_scheduler(dummy_optimizer, cfg)
        assert isinstance(sched, torch.optim.lr_scheduler.ExponentialLR)
        sched.step()
        assert dummy_optimizer.param_groups[0]["lr"] == pytest.approx(0.009)

    def test_default_is_cosine(self, dummy_optimizer):
        sched = build_lr_scheduler(dummy_optimizer, {})
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_unknown_type_raises(self, dummy_optimizer):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_lr_scheduler(dummy_optimizer, {"scheduler_type": "FooBar"})

    def test_alias_works(self, dummy_optimizer):
        sched = build_lr_scheduler(dummy_optimizer, {"scheduler_type": "cosine"})
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_invalid_param_raises(self, dummy_optimizer):
        with pytest.raises(ValueError):
            build_lr_scheduler(dummy_optimizer, {
                "scheduler_type": "CosineAnnealingLR",
                "scheduler_T_max": -5,
            })


# ---------------------------------------------------------------------------
# simulate_lr_schedule — parity and metadata
# ---------------------------------------------------------------------------

class TestSimulateLrSchedule:
    """simulate_lr_schedule must match Trainer.fit() step-then-read semantics."""

    def _training_loop_lrs(self, scheduler_cfg, num_epochs, base_lr, metrics=None):
        """Reproduce Trainer.fit() LR logging for comparison."""
        dummy = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(dummy.parameters(), lr=base_lr)
        sched = build_lr_scheduler(optimizer, scheduler_cfg)
        is_plateau = isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
        if metrics is None:
            metrics = [999.0] * num_epochs
        lrs = []
        for i in range(num_epochs):
            if sched is not None:
                if is_plateau:
                    sched.step(metrics[i])
                else:
                    sched.step()
            lrs.append(float(optimizer.param_groups[0]["lr"]))
        return lrs

    def test_step_lr_parity(self):
        cfg = {"scheduler_type": "StepLR", "scheduler_step_size": 3, "scheduler_gamma": 0.5}
        expected = self._training_loop_lrs(cfg, 10, 0.01)
        result = simulate_lr_schedule(cfg, 10, 0.01)
        assert result["lr"] == pytest.approx(expected)

    def test_cosine_parity(self):
        cfg = {"scheduler_type": "CosineAnnealingLR", "scheduler_T_max": 10, "scheduler_eta_min": 1e-5}
        expected = self._training_loop_lrs(cfg, 10, 0.01)
        result = simulate_lr_schedule(cfg, 10, 0.01)
        assert result["lr"] == pytest.approx(expected)

    def test_cosine_warm_restarts_parity(self):
        cfg = {"scheduler_type": "CosineAnnealingWarmRestarts", "scheduler_T_0": 5, "scheduler_T_mult": 1}
        expected = self._training_loop_lrs(cfg, 15, 0.01)
        result = simulate_lr_schedule(cfg, 15, 0.01)
        assert result["lr"] == pytest.approx(expected)

    def test_constant_parity(self):
        cfg = {"scheduler_type": "Constant"}
        expected = self._training_loop_lrs(cfg, 5, 0.01)
        result = simulate_lr_schedule(cfg, 5, 0.01)
        assert result["lr"] == pytest.approx(expected)

    def test_plateau_default_worst_case(self):
        cfg = {"scheduler_type": "ReduceLROnPlateau", "scheduler_patience": 3, "scheduler_factor": 0.5}
        result = simulate_lr_schedule(cfg, 20, 0.01)
        assert result["metadata"]["metric_mode"] == "flat_worst_case"
        assert result["lr"][-1] < result["lr"][0]

    def test_plateau_with_custom_metric_series(self):
        cfg = {"scheduler_type": "ReduceLROnPlateau", "scheduler_patience": 2, "scheduler_factor": 0.5}
        # Steadily improving: scheduler should never decay
        metrics = [1.0 - i * 0.05 for i in range(20)]
        result = simulate_lr_schedule(cfg, 20, 0.01, metric_series=metrics)
        assert result["metadata"]["metric_mode"] == "custom"
        # LR should stay at initial since loss always improves
        assert result["lr"][-1] == pytest.approx(result["lr"][0])

    def test_plateau_custom_parity(self):
        cfg = {"scheduler_type": "ReduceLROnPlateau", "scheduler_patience": 3, "scheduler_factor": 0.5}
        metrics = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        expected = self._training_loop_lrs(cfg, 10, 0.01, metrics=metrics)
        result = simulate_lr_schedule(cfg, 10, 0.01, metric_series=metrics)
        assert result["lr"] == pytest.approx(expected)

    def test_metric_series_wrong_length_raises(self):
        cfg = {"scheduler_type": "ReduceLROnPlateau"}
        with pytest.raises(ValueError, match="metric_series length"):
            simulate_lr_schedule(cfg, 10, 0.01, metric_series=[1.0] * 5)

    def test_metadata_keys_present(self):
        result = simulate_lr_schedule(
            {"scheduler_type": "StepLR", "scheduler_step_size": 5}, 10, 0.01
        )
        meta = result["metadata"]
        assert "scheduler_type" in meta
        assert "resolved" in meta
        assert "min_lr" in meta
        assert "max_lr" in meta
        assert meta["step_mode"] == "epoch"

    def test_metadata_min_max_lr(self):
        cfg = {"scheduler_type": "StepLR", "scheduler_step_size": 2, "scheduler_gamma": 0.5}
        result = simulate_lr_schedule(cfg, 10, 0.1)
        assert result["metadata"]["max_lr"] == pytest.approx(max(result["lr"]))
        assert result["metadata"]["min_lr"] == pytest.approx(min(result["lr"]))

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            simulate_lr_schedule({"scheduler_type": "CosineAnnealingLR", "scheduler_T_max": -1}, 10, 0.01)

    def test_alias_accepted(self):
        result = simulate_lr_schedule({"scheduler_type": "cosine"}, 10, 0.01)
        assert result["metadata"]["scheduler_type"] == "CosineAnnealingLR"

    def test_epochs_list_correct(self):
        result = simulate_lr_schedule({"scheduler_type": "Constant"}, 7, 0.01)
        assert result["epochs"] == list(range(7))


# ---------------------------------------------------------------------------
# /api/lr-preview endpoint
# ---------------------------------------------------------------------------

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
        assert data["lr"][-1] < data["lr"][0]  # LR decreased

    def test_response_includes_metadata(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "StepLR",
            "scheduler_step_size": 5,
            "base_lr": 0.01,
            "num_epochs": 10,
        }), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "metadata" in data
        meta = data["metadata"]
        assert meta["scheduler_type"] == "StepLR"
        assert "min_lr" in meta
        assert "max_lr" in meta
        assert meta["step_mode"] == "epoch"
        assert "resolved" in meta

    def test_plateau_with_metric_series(self, client):
        # Improving metrics — LR should stay constant
        metrics = [1.0 - i * 0.05 for i in range(20)]
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "ReduceLROnPlateau",
            "scheduler_patience": 3,
            "scheduler_factor": 0.5,
            "base_lr": 0.01,
            "num_epochs": 20,
            "metric_series": metrics,
        }), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["metadata"]["metric_mode"] == "custom"
        assert data["lr"][-1] == pytest.approx(data["lr"][0])

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
        assert data["lr"][-1] < data["lr"][0]

    def test_alias_in_preview(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "cosine",
            "scheduler_T_max": 10,
            "base_lr": 0.01,
            "num_epochs": 10,
        }), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["lr"]) == 10

    def test_invalid_param_returns_400(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "CosineAnnealingLR",
            "scheduler_T_max": -1,
            "base_lr": 0.001,
            "num_epochs": 10,
        }), content_type="application/json")
        assert resp.status_code == 400

    def test_unknown_type_returns_400(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "NotAScheduler",
            "base_lr": 0.001,
            "num_epochs": 10,
        }), content_type="application/json")
        assert resp.status_code == 400

    def test_multistep_preview(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "MultiStepLR",
            "scheduler_milestones": [5, 10],
            "scheduler_gamma": 0.5,
            "base_lr": 0.01,
            "num_epochs": 15,
        }), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["lr"]) == 15
        # Step-then-read: milestone 5 fires on 5th scheduler.step() call (i=4)
        assert data["lr"][4] < data["lr"][3]

    def test_exponential_preview(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "ExponentialLR",
            "scheduler_gamma": 0.9,
            "base_lr": 0.01,
            "num_epochs": 10,
        }), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["lr"][1] < data["lr"][0]


# ---------------------------------------------------------------------------
# Phase 3: flatten_scheduler_config
# ---------------------------------------------------------------------------

class TestFlattenSchedulerConfig:
    def test_flat_passthrough(self):
        raw = {"scheduler_type": "cosine", "scheduler_T_max": 50, "base_lr": 0.01}
        out = flatten_scheduler_config(raw)
        assert out == raw

    def test_nested_scheduler_merged(self):
        raw = {
            "scheduler": {"scheduler_type": "StepLR", "scheduler_step_size": 5},
            "base_lr": 0.01,
        }
        out = flatten_scheduler_config(raw)
        assert out["scheduler_type"] == "StepLR"
        assert out["scheduler_step_size"] == 5
        assert out["base_lr"] == 0.01
        assert "scheduler" not in out

    def test_outer_keys_take_precedence_over_nested(self):
        raw = {
            "scheduler": {"scheduler_type": "StepLR"},
            "scheduler_type": "cosine",  # outer wins
        }
        out = flatten_scheduler_config(raw)
        assert out["scheduler_type"] == "cosine"

    def test_no_scheduler_key_unchanged(self):
        raw = {"scheduler_type": "ExponentialLR", "scheduler_gamma": 0.9}
        out = flatten_scheduler_config(raw)
        assert out == raw

    def test_non_dict_scheduler_value_ignored(self):
        # If scheduler is not a dict (e.g. a string), don't attempt to merge
        raw = {"scheduler": "cosine", "base_lr": 0.01}
        out = flatten_scheduler_config(raw)
        assert out == raw

    def test_empty_nested_scheduler(self):
        raw = {"scheduler": {}, "base_lr": 0.01}
        out = flatten_scheduler_config(raw)
        assert "scheduler" not in out
        assert out["base_lr"] == 0.01


# ---------------------------------------------------------------------------
# Phase 3: /api/config schema contract
# ---------------------------------------------------------------------------

class TestConfigSchema:
    def test_scheduler_type_options_complete(self, client):
        resp = client.get("/api/config")
        data = resp.get_json()
        options = data["scheduler"]["scheduler_type"]["options"]
        expected = {
            "Constant", "CosineAnnealingLR", "StepLR",
            "ReduceLROnPlateau", "CosineAnnealingWarmRestarts",
            "MultiStepLR", "ExponentialLR",
        }
        assert set(options) == expected

    def test_scheduler_params_have_constraints(self, client):
        resp = client.get("/api/config")
        data = resp.get_json()
        sched = data["scheduler"]
        # Positive-integer params must have min=1
        for key in ("scheduler_T_max", "scheduler_step_size", "scheduler_T_0", "scheduler_T_mult"):
            assert sched[key].get("min") == 1, f"{key} should have min=1"
        # Non-negative float params must have min=0
        for key in ("scheduler_eta_min", "scheduler_min_lr"):
            assert sched[key].get("min") == 0.0, f"{key} should have min=0.0"
        # Unit-range params must have min and max
        for key in ("scheduler_gamma", "scheduler_factor"):
            assert sched[key].get("min") is not None
            assert sched[key].get("max") is not None

    def test_scheduler_params_have_help_text(self, client):
        resp = client.get("/api/config")
        data = resp.get_json()
        sched = data["scheduler"]
        for key in ("scheduler_T_max", "scheduler_step_size", "scheduler_gamma",
                    "scheduler_factor", "scheduler_patience", "scheduler_milestones"):
            assert "help" in sched[key], f"{key} should have help text"

    def test_scheduler_step_mode_present(self, client):
        resp = client.get("/api/config")
        data = resp.get_json()
        assert "scheduler_step_mode" in data["scheduler"]
        step_mode = data["scheduler"]["scheduler_step_mode"]
        assert step_mode["value"] == "epoch"
        assert "epoch" in step_mode["options"]

    def test_nested_payload_works_in_preview(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler": {
                "scheduler_type": "StepLR",
                "scheduler_step_size": 5,
                "scheduler_gamma": 0.5,
            },
            "base_lr": 0.01,
            "num_epochs": 10,
        }), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["lr"]) == 10
        assert data["metadata"]["scheduler_type"] == "StepLR"

    def test_nested_payload_parity_with_flat(self, client):
        flat_resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "CosineAnnealingLR",
            "scheduler_T_max": 10,
            "base_lr": 0.01,
            "num_epochs": 10,
        }), content_type="application/json")
        nested_resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler": {"scheduler_type": "CosineAnnealingLR", "scheduler_T_max": 10},
            "base_lr": 0.01,
            "num_epochs": 10,
        }), content_type="application/json")
        assert flat_resp.status_code == 200
        assert nested_resp.status_code == 200
        assert flat_resp.get_json()["lr"] == pytest.approx(nested_resp.get_json()["lr"])


# ---------------------------------------------------------------------------
# Phase 4: Warmup (LinearLR + SequentialLR)
# ---------------------------------------------------------------------------

class TestWarmupValidation:
    def test_warmup_steps_defaults_to_zero(self, dummy_optimizer):
        cfg = validate_scheduler_config({"scheduler_type": "CosineAnnealingLR"})
        assert cfg["scheduler_warmup_steps"] == 0

    def test_warmup_start_factor_defaults(self, dummy_optimizer):
        cfg = validate_scheduler_config({"scheduler_type": "StepLR"})
        assert cfg["scheduler_warmup_start_factor"] == pytest.approx(0.1)

    def test_warmup_end_factor_defaults(self, dummy_optimizer):
        cfg = validate_scheduler_config({"scheduler_type": "StepLR"})
        assert cfg["scheduler_warmup_end_factor"] == pytest.approx(1.0)

    def test_negative_warmup_steps_raises(self):
        with pytest.raises(ValueError, match="constraint 'non_negative'"):
            validate_scheduler_config({
                "scheduler_type": "CosineAnnealingLR",
                "scheduler_warmup_steps": -1,
            })

    def test_zero_start_factor_raises(self):
        with pytest.raises(ValueError, match="constraint 'unit_range'"):
            validate_scheduler_config({
                "scheduler_type": "StepLR",
                "scheduler_warmup_start_factor": 0.0,
            })

    def test_zero_end_factor_raises(self):
        with pytest.raises(ValueError, match="constraint 'positive'"):
            validate_scheduler_config({
                "scheduler_type": "StepLR",
                "scheduler_warmup_end_factor": 0.0,
            })

    def test_plateau_with_warmup_raises(self):
        with pytest.raises(ValueError, match="not supported with 'ReduceLROnPlateau'"):
            validate_scheduler_config({
                "scheduler_type": "ReduceLROnPlateau",
                "scheduler_warmup_steps": 5,
            })


class TestWarmupBuilder:
    def test_no_warmup_returns_base(self, dummy_optimizer):
        sched = build_lr_scheduler(dummy_optimizer, {
            "scheduler_type": "StepLR",
            "scheduler_warmup_steps": 0,
        })
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_warmup_returns_sequential_lr(self, dummy_optimizer):
        sched = build_lr_scheduler(dummy_optimizer, {
            "scheduler_type": "CosineAnnealingLR",
            "scheduler_T_max": 20,
            "scheduler_warmup_steps": 5,
        })
        assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)

    def test_warmup_lr_ramps_up(self, dummy_optimizer):
        """LR should increase during warmup phase."""
        sched = build_lr_scheduler(dummy_optimizer, {
            "scheduler_type": "CosineAnnealingLR",
            "scheduler_T_max": 20,
            "scheduler_warmup_steps": 5,
            "scheduler_warmup_start_factor": 0.1,
            "scheduler_warmup_end_factor": 1.0,
        })
        lrs = []
        for _ in range(5):
            sched.step()
            lrs.append(dummy_optimizer.param_groups[0]["lr"])
        # LR should be monotonically increasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1]

    def test_warmup_then_cosine_decays(self, dummy_optimizer):
        """After warmup the cosine schedule should decay LR."""
        sched = build_lr_scheduler(dummy_optimizer, {
            "scheduler_type": "CosineAnnealingLR",
            "scheduler_T_max": 10,
            "scheduler_warmup_steps": 3,
        })
        # Run past warmup
        for _ in range(3):
            sched.step()
        lr_at_warmup_end = dummy_optimizer.param_groups[0]["lr"]
        # Run into cosine phase
        for _ in range(10):
            sched.step()
        lr_after_cosine = dummy_optimizer.param_groups[0]["lr"]
        assert lr_after_cosine < lr_at_warmup_end

    def test_warmup_with_multistep(self, dummy_optimizer):
        sched = build_lr_scheduler(dummy_optimizer, {
            "scheduler_type": "MultiStepLR",
            "scheduler_milestones": [10, 20],
            "scheduler_warmup_steps": 3,
        })
        assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)

    def test_warmup_with_exponential(self, dummy_optimizer):
        sched = build_lr_scheduler(dummy_optimizer, {
            "scheduler_type": "ExponentialLR",
            "scheduler_gamma": 0.9,
            "scheduler_warmup_steps": 2,
        })
        assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)


class TestWarmupPreviewEndpoint:
    def test_warmup_cosine_preview(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "CosineAnnealingLR",
            "scheduler_T_max": 20,
            "scheduler_warmup_steps": 5,
            "scheduler_warmup_start_factor": 0.1,
            "base_lr": 0.01,
            "num_epochs": 25,
        }), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["lr"]) == 25
        # Warmup phase: LR should rise
        assert data["lr"][4] > data["lr"][0]
        # After warmup: LR should be higher than start
        assert data["lr"][5] >= data["lr"][0]

    def test_warmup_plateau_returns_400(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "ReduceLROnPlateau",
            "scheduler_warmup_steps": 5,
            "base_lr": 0.01,
            "num_epochs": 20,
        }), content_type="application/json")
        assert resp.status_code == 400
        assert "not supported" in resp.get_json()["error"]

    def test_warmup_metadata_reflected(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "StepLR",
            "scheduler_step_size": 5,
            "scheduler_warmup_steps": 3,
            "base_lr": 0.01,
            "num_epochs": 15,
        }), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        resolved = data["metadata"]["resolved"]
        assert resolved["scheduler_warmup_steps"] == 3

    def test_config_has_warmup_fields(self, client):
        resp = client.get("/api/config")
        sched = resp.get_json()["scheduler"]
        assert "scheduler_warmup_steps" in sched
        assert "scheduler_warmup_start_factor" in sched
        assert "scheduler_warmup_end_factor" in sched
        assert sched["scheduler_warmup_steps"]["min"] == 0


# ---------------------------------------------------------------------------
# Phase 5: Error contract for UI client-side guard reliance
# ---------------------------------------------------------------------------

class TestPhase5ErrorContract:
    """Verify that /api/lr-preview 400 responses carry a machine-readable
    'error' key so the UI inline error display can show meaningful messages."""

    def test_400_has_error_key(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "UnknownSched",
            "base_lr": 0.01,
            "num_epochs": 10,
        }), content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert isinstance(data["error"], str)
        assert len(data["error"]) > 0

    def test_constraint_violation_error_is_descriptive(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "CosineAnnealingLR",
            "scheduler_T_max": 0,
            "base_lr": 0.01,
            "num_epochs": 10,
        }), content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "scheduler_T_max" in data["error"]

    def test_warmup_incompatibility_error_is_descriptive(self, client):
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "ReduceLROnPlateau",
            "scheduler_warmup_steps": 5,
            "base_lr": 0.01,
            "num_epochs": 20,
        }), content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "ReduceLROnPlateau" in data["error"]

    def test_preview_metadata_summary_fields_present(self, client):
        """Ensure metadata has all fields needed by the summary card."""
        resp = client.post("/api/lr-preview", data=json.dumps({
            "scheduler_type": "CosineAnnealingLR",
            "scheduler_T_max": 20,
            "scheduler_warmup_steps": 3,
            "base_lr": 0.01,
            "num_epochs": 20,
        }), content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        meta = data["metadata"]
        # All fields the JS summary card reads
        assert "scheduler_type" in meta
        assert "min_lr" in meta
        assert "max_lr" in meta
        assert "resolved" in meta
        assert "scheduler_warmup_steps" in meta["resolved"]
        # Sanity: max_lr >= min_lr
        assert meta["max_lr"] >= meta["min_lr"]

    def test_preview_lr_list_length_matches_epochs(self, client):
        for n in [1, 10, 50]:
            resp = client.post("/api/lr-preview", data=json.dumps({
                "scheduler_type": "StepLR",
                "scheduler_step_size": 5,
                "base_lr": 0.01,
                "num_epochs": n,
            }), content_type="application/json")
            assert resp.status_code == 200
            data = resp.get_json()
            assert len(data["lr"]) == n
            assert len(data["epochs"]) == n


# ---------------------------------------------------------------------------
# Phase 6: build_scheduler_snapshot + experiment metadata capture
# ---------------------------------------------------------------------------

class TestBuildSchedulerSnapshot:
    def test_canonical_type_in_snapshot(self):
        snap = build_scheduler_snapshot(
            {"scheduler_type": "cosine", "scheduler_T_max": 50},
            num_epochs=50, base_lr=0.001,
        )
        assert snap["type"] == "CosineAnnealingLR"

    def test_params_are_resolved_defaults(self):
        snap = build_scheduler_snapshot(
            {"scheduler_type": "CosineAnnealingLR"},
            num_epochs=100, base_lr=0.001,
        )
        assert "scheduler_T_max" in snap["params"]
        assert "scheduler_eta_min" in snap["params"]
        # No raw-invalid values: T_max should be positive int
        assert snap["params"]["scheduler_T_max"] > 0

    def test_first_and_final_lr_present(self):
        snap = build_scheduler_snapshot(
            {"scheduler_type": "StepLR", "scheduler_step_size": 5, "scheduler_gamma": 0.5},
            num_epochs=20, base_lr=0.01,
        )
        assert "first_lr" in snap
        assert "final_lr" in snap
        assert snap["final_lr"] < snap["first_lr"]  # LR should have decayed

    def test_step_mode_is_epoch(self):
        snap = build_scheduler_snapshot(
            {"scheduler_type": "Constant"},
            num_epochs=10, base_lr=0.001,
        )
        assert snap["step_mode"] == "epoch"

    def test_warmup_steps_in_snapshot(self):
        snap = build_scheduler_snapshot(
            {"scheduler_type": "CosineAnnealingLR", "scheduler_warmup_steps": 5},
            num_epochs=30, base_lr=0.001,
        )
        assert snap["warmup_steps"] == 5

    def test_no_raw_invalid_values_leak(self):
        """Snapshot params must all be validated — no string/None/negative values."""
        snap = build_scheduler_snapshot(
            {"scheduler_type": "ReduceLROnPlateau"},
            num_epochs=20, base_lr=0.001,
        )
        for k, v in snap["params"].items():
            assert v is not None, f"{k} should not be None"
            if isinstance(v, (int, float)):
                assert not (v != v), f"{k} should not be NaN"  # NaN check

    def test_alias_normalised_in_snapshot(self):
        for alias, expected in [("step", "StepLR"), ("exponential", "ExponentialLR"),
                                 ("plateau", "ReduceLROnPlateau")]:
            snap = build_scheduler_snapshot(
                {"scheduler_type": alias},
                num_epochs=10, base_lr=0.001,
            )
            assert snap["type"] == expected, f"alias {alias!r} should map to {expected}"

    def test_snapshot_is_serialisable(self):
        """The snapshot dict must be JSON-serialisable (no torch tensors etc.)."""
        import json
        snap = build_scheduler_snapshot(
            {"scheduler_type": "CosineAnnealingWarmRestarts", "scheduler_T_0": 10},
            num_epochs=30, base_lr=0.001,
        )
        # Should not raise
        serialised = json.dumps(snap)
        assert len(serialised) > 0


class TestSchedulerSnapshotInBenchmarkCall:
    """Verify the snapshot is passed through to run_benchmark_and_log_experiment."""

    def _make_cfg(self, scheduler_type="StepLR", **kwargs):
        return {
            "scheduler_type": scheduler_type,
            "scheduler_step_size": 5,
            "scheduler_gamma": 0.5,
            "lr": 0.001,
            "num_epochs": 10,
            "task": "scarce",
            "hidden_dim": 16,
            "num_layers": 4,
            **kwargs,
        }

    def test_scfg_contains_scheduler_key(self):
        """build_scheduler_snapshot returns a dict that can be used as scfg["scheduler"]."""
        cfg = self._make_cfg()
        snap = build_scheduler_snapshot(cfg, cfg["num_epochs"], cfg["lr"])

        # Simulate what _run does
        scfg = {
            "task": cfg.get("task"),
            "hidden": cfg.get("hidden_dim"),
            "layers": cfg.get("num_layers"),
            "scheduler": snap,
        }

        assert "scheduler" in scfg
        assert scfg["scheduler"]["type"] == "StepLR"
        assert "params" in scfg["scheduler"]

    def test_scfg_scheduler_type_is_canonical(self):
        cfg = self._make_cfg(scheduler_type="cosine")
        snap = build_scheduler_snapshot(cfg, 10, 0.001)
        assert snap["type"] == "CosineAnnealingLR"

    def test_scfg_scheduler_no_raw_alias_leaks(self):
        """The 'type' field in the snapshot must be the canonical class name."""
        for alias in ("cosine", "step", "plateau", "warm_restarts", "multistep", "exponential"):
            snap = build_scheduler_snapshot({"scheduler_type": alias}, 10, 0.001)
            assert snap["type"] in ALLOWED_SCHEDULER_TYPES, (
                f"alias {alias!r} produced non-canonical type {snap['type']!r}"
            )
