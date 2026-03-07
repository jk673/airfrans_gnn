from types import SimpleNamespace
import sys

import pytest
import torch

from dashboard import hpo


def test_is_out_of_memory_error_matches_torch_and_runtime_errors():
    assert hpo._is_out_of_memory_error(torch.OutOfMemoryError("CUDA out of memory"))
    assert hpo._is_out_of_memory_error(RuntimeError("CUDA error: out of memory"))
    assert hpo._is_out_of_memory_error(MemoryError("host memory exhausted"))
    assert not hpo._is_out_of_memory_error(RuntimeError("shape mismatch"))


def test_hpo_oom_prunes_only_the_current_trial(monkeypatch):
    class FakeTrialPruned(Exception):
        pass

    class FakeTrial:
        def __init__(self, number):
            self.number = number
            self.user_attrs = {}

        def report(self, value, step):
            return None

        def should_prune(self):
            return False

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    class FakeStudy:
        def __init__(self):
            self.enqueued = []

        def enqueue_trial(self, params):
            self.enqueued.append(params)

        def optimize(self, objective, n_trials, callbacks, gc_after_trial):
            for trial_num in range(n_trials):
                trial = FakeTrial(trial_num)
                try:
                    objective(trial)
                except FakeTrialPruned:
                    continue

    fake_optuna = SimpleNamespace(
        create_study=lambda **kwargs: FakeStudy(),
        samplers=SimpleNamespace(TPESampler=lambda **kwargs: object()),
        pruners=SimpleNamespace(MedianPruner=lambda **kwargs: object()),
        exceptions=SimpleNamespace(TrialPruned=FakeTrialPruned),
    )

    fit_calls = {"count": 0}

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, train_loader, val_loader, num_epochs, routine, on_epoch_end):
            fit_calls["count"] += 1
            if fit_calls["count"] == 1:
                raise torch.OutOfMemoryError("CUDA out of memory while allocating tensor")

            on_epoch_end(
                epoch=0,
                train_logs={"total_loss": 0.5},
                val_logs={"total_loss": 0.25},
                lr=1e-3,
                is_best=True,
            )

    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setattr(hpo, "build_model", lambda *args, **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(hpo, "build_physics_loss", lambda *args, **kwargs: object())
    monkeypatch.setattr(hpo, "build_lr_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr(hpo, "Trainer", FakeTrainer)
    monkeypatch.setattr(hpo, "train_one_epoch", lambda *args, **kwargs: None)
    monkeypatch.setattr(hpo, "validate_one_epoch", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    session = hpo.HpoSession()
    dummy_bundle = SimpleNamespace(edge_dim=2, train_loader=[object()], val_loader=[object()])
    monkeypatch.setattr(session, "_get_bundle", lambda batch_size, task, seed: dummy_bundle)

    session._run(
        specs=[],
        settings={
            "task": "scarce",
            "seed": 42,
            "n_lhs": 1,
            "total_trials": 2,
            "epochs_per_trial": 1,
            "pruning_patience": 1,
            "device": "cpu",
        },
    )

    status = session.get_status()
    assert status["state"] == "completed"
    assert len(status["trials"]) == 2
    assert status["trials"][0]["state"] == "pruned"
    assert status["trials"][0]["reason"] == "oom"
    assert status["trials"][1]["state"] == "complete"
    assert status["trials"][1]["value"] == pytest.approx(0.25)
    assert status["best_value"] == pytest.approx(0.25)
    assert status["current_params"] == {}
    assert status["completed_trials"] == 1
    assert status["pruned_trials"] == 1
    assert status["finished_trials"] == 2
    assert status["progress_pct"] == pytest.approx(100.0)
    assert status["error_message"] == ""


def test_hpo_status_exposes_current_params_while_trial_is_running(monkeypatch):
    class FakeTrialPruned(Exception):
        pass

    class FakeTrial:
        def __init__(self, number):
            self.number = number

        def report(self, value, step):
            return None

        def should_prune(self):
            return False

        def set_user_attr(self, key, value):
            return None

    class FakeStudy:
        def enqueue_trial(self, params):
            return None

        def optimize(self, objective, n_trials, callbacks, gc_after_trial):
            objective(FakeTrial(0))

    fake_optuna = SimpleNamespace(
        create_study=lambda **kwargs: FakeStudy(),
        samplers=SimpleNamespace(TPESampler=lambda **kwargs: object()),
        pruners=SimpleNamespace(MedianPruner=lambda **kwargs: object()),
        exceptions=SimpleNamespace(TrialPruned=FakeTrialPruned),
    )

    captured = {}

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, train_loader, val_loader, num_epochs, routine, on_epoch_end):
            status = session.get_status()
            captured["current_params"] = status["current_params"]
            captured["current_trial"] = status["current_trial"]
            captured["progress_pct"] = status["progress_pct"]
            on_epoch_end(
                epoch=0,
                train_logs={"total_loss": 0.4},
                val_logs={"total_loss": 0.2},
                lr=1e-3,
                is_best=True,
            )

    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setattr(hpo, "_suggest_params", lambda trial, specs: {
        "hidden_dim": 128,
        "lr": 1e-3,
        "amp": True,
        "batch_size": 2,
    })
    monkeypatch.setattr(hpo, "build_model", lambda *args, **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(hpo, "build_physics_loss", lambda *args, **kwargs: object())
    monkeypatch.setattr(hpo, "build_lr_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr(hpo, "Trainer", FakeTrainer)
    monkeypatch.setattr(hpo, "train_one_epoch", lambda *args, **kwargs: None)
    monkeypatch.setattr(hpo, "validate_one_epoch", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    session = hpo.HpoSession()
    dummy_bundle = SimpleNamespace(edge_dim=2, train_loader=[object()], val_loader=[object()])
    monkeypatch.setattr(session, "_get_bundle", lambda batch_size, task, seed: dummy_bundle)

    specs = [
        hpo.HpoParamSpec("hidden_dim", "model", "int"),
        hpo.HpoParamSpec("lr", "optimizer", "float"),
        hpo.HpoParamSpec("amp", "training", "bool"),
        hpo.HpoParamSpec("batch_size", "training", "int"),
    ]
    session._run(
        specs=specs,
        settings={
            "task": "scarce",
            "seed": 42,
            "n_lhs": 1,
            "total_trials": 1,
            "epochs_per_trial": 1,
            "pruning_patience": 1,
            "device": "cpu",
        },
    )

    assert captured["current_params"] == {
        "hidden_dim": 128,
        "lr": 1e-3,
        "amp": True,
        "batch_size": 2,
    }
    assert captured["current_trial"] == 1
    assert captured["progress_pct"] == pytest.approx(100.0)
