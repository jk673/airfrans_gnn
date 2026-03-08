from dashboard import app as dashboard_app_module


def _reset_lazy_globals():
    dashboard_app_module._runner_module = None
    dashboard_app_module._training_session = None
    dashboard_app_module._hpo_module = None
    dashboard_app_module._hpo_session = None
    dashboard_app_module._benchmark_module = None


def test_status_is_idle_without_loading_training_modules():
    _reset_lazy_globals()
    client = dashboard_app_module.app.test_client()

    resp = client.get("/api/status")

    assert resp.status_code == 200
    assert resp.json["state"] == "idle"
    assert resp.json["best_val"] is None
    assert resp.json["metrics"] == {"epochs": [], "train": {}, "val": {}, "lr": []}
    assert dashboard_app_module._runner_module is None
    assert dashboard_app_module._training_session is None


def test_hpo_status_is_idle_without_loading_hpo_modules():
    _reset_lazy_globals()
    client = dashboard_app_module.app.test_client()

    resp = client.get("/api/hpo/status")

    assert resp.status_code == 200
    assert resp.json["state"] == "idle"
    assert resp.json["trials"] == []
    assert resp.json["progress_pct"] == 0.0
    assert dashboard_app_module._hpo_module is None
    assert dashboard_app_module._hpo_session is None


def test_config_endpoint_keeps_lazy_modules_unloaded():
    _reset_lazy_globals()
    client = dashboard_app_module.app.test_client()

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert "data" in resp.json
    assert "training" in resp.json
    assert dashboard_app_module._runner_module is None
    assert dashboard_app_module._hpo_module is None
