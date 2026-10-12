"""
End-to-end tests of every endpoint against the real artefacts in models/ and
outputs/. Nothing is mocked except where a failure mode is being simulated
(missing models, warm-up in progress).
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import prediction
from api.core import AppState
from api.main import create_app

from .conftest import wait_for

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ["cooler", "valve", "pump", "accumulator"]
TEST_RESULTS = pd.read_csv(ROOT / "outputs" / "metrics" / "test_results.csv").set_index("component")
RUN_META = json.loads((ROOT / "models" / "run_metadata.json").read_text())


def csv_bytes(client: TestClient, cycle_id: int) -> bytes:
    r = client.get(f"/api/cycles/{cycle_id}/csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    return r.content


# --------------------------------------------------------------------------
# Health and descriptive endpoints
# --------------------------------------------------------------------------
def test_health_ready(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ready"
    assert all(body["checks"]["models"].values())
    assert body["checks"]["feature_table"] and body["checks"]["raw_data"] and body["checks"]["metrics"]
    assert all(step["done"] for step in body["steps"])
    assert body["versions"]["scikit-learn"] == "1.8.0"


def test_overview_numbers_come_from_artefacts(client: TestClient) -> None:
    body = client.get("/api/overview").json()
    summary = RUN_META["dataset_summary"]
    assert body["dataset"]["n_cycles"] == summary["n_cycles"] == 2205
    assert body["dataset"]["raw_values_per_cycle"] == summary["total_raw_dimensions"]
    assert body["feature_counts"]["engineered"] == sum(RUN_META["feature_blocks"].values())
    assert body["feature_counts"]["cleaned"] == RUN_META["cleaning"]["n_output"]
    assert body["run"]["n_configurations"] == 48
    for h in body["headline"]:
        row = TEST_RESULTS.loc[h["component"]]
        assert h["test_f1_macro"] == pytest.approx(row.f1_macro, abs=0)
        assert h["block_cv_f1_macro"] == pytest.approx(row.block_cv_f1_macro, abs=0)
    assert [h["component"] for h in body["headline"]] == COMPONENTS


def test_sensors(client: TestClient) -> None:
    body = client.get("/api/sensors").json()
    assert len(body["sensors"]) == 17
    grouped = [s for g in body["groups"] for s in g["sensors"]]
    assert sorted(grouped) == sorted(s["name"] for s in body["sensors"])


def test_labels(client: TestClient) -> None:
    body = client.get("/api/labels").json()
    assert body["n_cycles"] == 2205
    for comp in body["components"]:
        assert sum(c["count"] for c in comp["counts"]) == 2205
    assert sum(body["stable_flag"].values()) == 2205


# --------------------------------------------------------------------------
# Cycles
# --------------------------------------------------------------------------
def test_cycle_valid_downsamples_but_keeps_extremes(client: TestClient) -> None:
    body = client.get("/api/cycles/0").json()
    assert body["cycle_id"] == 0 and len(body["signals"]) == 17
    raw = prediction.load_cycle_row(0)
    for sig in body["signals"]:
        assert sig["n_points"] == len(sig["t"]) == len(sig["v"])
        if sig["sampling_rate_hz"] == 100:
            assert sig["downsampled"] and sig["n_points"] <= 600
            # min/max envelope: the plotted extremes are the true extremes
            assert max(sig["v"]) == pytest.approx(raw[sig["sensor"]].max(), rel=1e-5)
            assert min(sig["v"]) == pytest.approx(raw[sig["sensor"]].min(), rel=1e-5)
        else:
            assert not sig["downsampled"] and sig["n_points"] == sig["n_samples"]
    assert set(body["labels"]) == set(COMPONENTS)


@pytest.mark.parametrize("cycle_id", [2205, 99999, -1])
def test_cycle_unknown_is_404(client: TestClient, cycle_id: int) -> None:
    r = client.get(f"/api/cycles/{cycle_id}")
    assert r.status_code == 404
    assert r.json()["error"] == "cycle_not_found"


def test_cycle_non_integer_is_422(client: TestClient) -> None:
    r = client.get("/api/cycles/abc")
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def test_predict_dataset_cycle(client: TestClient) -> None:
    r = client.post("/api/predict/cycle/2100")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == {"kind": "dataset", "cycle_id": 2100, "filename": None}
    assert [c["component"] for c in body["components"]] == COMPONENTS
    assert body["system_status"] in {"Normal", "Degraded", "Critical"}
    for comp in body["components"]:
        assert sum(p["p"] for p in comp["probabilities"]) == pytest.approx(1.0, abs=1e-6)
        assert comp["confidence"] == pytest.approx(max(p["p"] for p in comp["probabilities"]))
        assert len(comp["explanation"]) == 10
        assert len(comp["neighbours"]) == 5
        assert all(n["cycle"] != 2100 for n in comp["neighbours"])
        assert comp["true_value"] is not None and comp["in_test_set"] is not None
        assert comp["correct"] == (comp["true_value"] == comp["predicted_value"])


def test_raw_path_agrees_with_feature_table(client: TestClient) -> None:
    """Prediction from raw files must equal the pipeline applied to the saved feature row."""
    table = pd.read_csv(ROOT / "outputs" / "features" / "engineered_features.csv", skiprows=range(1, 51), nrows=1)
    assert int(table["cycle_id"].iloc[0]) == 50
    body = client.post("/api/predict/cycle/50").json()
    for comp in body["components"]:
        bundle = prediction.load_component_bundle(comp["component"])
        cols = bundle["metadata"]["input_feature_columns"]
        proba = bundle["model"].predict_proba(table[cols].to_numpy(dtype=np.float64))[0]
        api_proba = [p["p"] for p in comp["probabilities"]]
        np.testing.assert_allclose(api_proba, proba, atol=1e-6)


def test_predict_unknown_cycle_is_404(client: TestClient) -> None:
    r = client.post("/api/predict/cycle/5000")
    assert r.status_code == 404 and r.json()["error"] == "cycle_not_found"


def test_predict_upload_good_csv_matches_dataset_cycle(client: TestClient) -> None:
    content = csv_bytes(client, 777)
    r = client.post("/api/predict/upload", files={"file": ("cycle_777.csv", content, "text/csv")})
    assert r.status_code == 200, r.text
    up = r.json()
    ds = client.post("/api/predict/cycle/777").json()
    assert up["source"]["kind"] == "upload" and up["source"]["filename"] == "cycle_777.csv"
    for a, b in zip(up["components"], ds["components"]):
        assert a["predicted_value"] == b["predicted_value"]
        np.testing.assert_allclose([p["p"] for p in a["probabilities"]], [p["p"] for p in b["probabilities"]], atol=1e-9)
        assert a["true_value"] is None and a["in_test_set"] is None


def _malformed(client: TestClient) -> "dict[str, bytes]":
    good = pd.read_csv(io.BytesIO(csv_bytes(client, 3)))
    missing = good.drop(columns=["PS1"]).to_csv(index=False).encode()
    short = good.copy()
    short.loc[5990:, "PS1"] = np.nan  # PS1 now has 5990 samples instead of 6000
    text = good.copy().astype(object)
    text["TS1"] = "hot"
    return {
        "empty": b"",
        "missing_column": missing,
        "wrong_sample_count": short.to_csv(index=False).encode(),
        "non_numeric": text.to_csv(index=False).encode(),
        "binary_garbage": bytes(range(256)) * 10,
        "header_only": (",".join(good.columns) + "\n").encode(),
    }


@pytest.mark.parametrize(
    "case", ["empty", "missing_column", "wrong_sample_count", "non_numeric", "binary_garbage", "header_only"]
)
def test_predict_upload_malformed_csv_is_400(client: TestClient, case: str) -> None:
    content = _malformed(client)[case]
    r = client.post("/api/predict/upload", files={"file": (f"{case}.csv", content, "text/csv")})
    assert r.status_code == 400, (case, r.text)
    body = r.json()
    assert body["error"] == "invalid_csv" and body["detail"]


def test_predict_upload_requires_file(client: TestClient) -> None:
    r = client.post("/api/predict/upload")
    assert r.status_code == 422


# --------------------------------------------------------------------------
# Metrics, features, models
# --------------------------------------------------------------------------
def test_metrics(client: TestClient) -> None:
    body = client.get("/api/metrics").json()
    assert len(body["cv"]) == 48 and len(body["test"]) == 4 and len(body["experiment_effect"]) == 4
    assert body["feature_set_sizes"] == {"baseline": 17, "engineered": 553, "selected": 40}


def test_features(client: TestClient) -> None:
    body = client.get("/api/features").json()
    assert {b["key"]: b["count"] for b in body["blocks"]} == RUN_META["feature_blocks"]
    assert body["cleaning"]["n_output"] == 553
    assert [s["n_selected"] for s in body["selection"]] == [17, 40, 40, 553]


def test_importance(client: TestClient) -> None:
    body = client.get("/api/features/importance", params={"component": "valve", "top": 5}).json()
    assert len(body["impurity"]) == 5 and len(body["permutation"]) == 5
    imps = [i["importance"] for i in body["impurity"]]
    assert imps == sorted(imps, reverse=True)
    assert client.get("/api/features/importance", params={"component": "boiler"}).status_code == 422


@pytest.mark.parametrize("component", COMPONENTS)
def test_evaluation_parity_with_saved_results(client: TestClient, component: str) -> None:
    """Live test-set scores must equal outputs/metrics/test_results.csv exactly."""
    body = client.get(f"/api/evaluation/{component}").json()
    row = TEST_RESULTS.loc[component]
    assert body["n_test"] == row.n_test_cycles
    assert body["accuracy"] == pytest.approx(row.accuracy, abs=1e-12)
    assert body["f1_macro"] == pytest.approx(row.f1_macro, abs=1e-12)
    assert body["roc_auc"] == pytest.approx(row.roc_auc_ovr_macro, abs=1e-12)
    assert body["parity"]["matches"] is True
    assert sum(map(sum, body["confusion"])) == row.n_test_cycles
    assert len(body["misclassified"]) == body["n_errors"]
    assert body["saved_report_text"]


def test_pipeline(client: TestClient) -> None:
    body = client.get("/api/models/valve/pipeline").json()
    assert body["steps"][-1]["name"] == "model"
    assert body["funnel"][0]["features"] == 553 and body["funnel"][-1]["features"] == 40
    assert len(body["selected_features"]) == 40


def test_generalization_and_timeline(client: TestClient) -> None:
    body = client.get("/api/generalization").json()
    assert body["block_size"] == RUN_META["block_size"] and len(body["rows"]) == 4
    tl = client.get("/api/generalization/pump/timeline", params={"start": 10, "end": 60}).json()
    assert len(tl["points"]) == 51 and {p["split"] for p in tl["points"]} <= {"train", "test"}
    bad = client.get("/api/generalization/pump/timeline", params={"start": 60, "end": 10})
    assert bad.status_code == 400 and bad.json()["error"] == "invalid_window"


# --------------------------------------------------------------------------
# Errors, what-if, robustness
# --------------------------------------------------------------------------
def test_error_cases_and_profile(client: TestClient) -> None:
    acc = client.get("/api/errors/accumulator").json()
    assert acc["mode"] == "errors" and acc["n_errors"] == len(acc["cases"]) > 0
    cooler = client.get("/api/errors/cooler").json()
    assert cooler["mode"] == "least_confident" and len(cooler["cases"]) == 10
    case = acc["cases"][0]
    prof = client.get(f"/api/errors/accumulator/{case['cycle']}").json()
    assert prof["true_value"] == case["true_value"] and prof["predicted_value"] == case["predicted_value"]
    assert len(prof["features"]) == 10 and prof["features"][0]["z_pred_mean"] is not None
    assert client.get("/api/errors/accumulator/99999").status_code == 404


def test_whatif(client: TestClient) -> None:
    opts = client.get("/api/whatif/valve").json()
    feat = opts["features"][0]
    same = client.post("/api/whatif/valve", json={"cycle": opts["cycle"], "feature": feat["name"], "value": feat["current"]}).json()
    assert same["changed"] is False
    assert [p["p"] for p in same["original"]] == [p["p"] for p in same["modified"]]
    bad = client.post("/api/whatif/valve", json={"cycle": opts["cycle"], "feature": "nope", "value": 1})
    assert bad.status_code == 400 and bad.json()["error"] == "unknown_feature"
    assert client.post("/api/whatif/valve", json={"cycle": 1}).status_code == 422


def test_noise_and_selective(client: TestClient) -> None:
    noise = client.get("/api/robustness/accumulator/noise").json()
    assert [p["level"] for p in noise["points"]] == [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    assert noise["points"][0]["f1_macro"] == pytest.approx(TEST_RESULTS.loc["accumulator"].f1_macro, abs=1e-12)
    assert client.get("/api/robustness/accumulator/noise").json() == noise  # cached and deterministic
    sel = client.get("/api/robustness/accumulator/selective").json()
    assert sel["grid"][0]["threshold"] == 0.3 and sel["grid"][-1]["threshold"] == 1.0
    coverages = [g["coverage"] for g in sel["grid"]]
    assert coverages == sorted(coverages, reverse=True)


def test_figures(client: TestClient) -> None:
    figs = client.get("/api/figures").json()["figures"]
    keys = {f["key"] for f in figs}
    assert "correlation_heatmap" in keys and "pca_valve" in keys
    png = client.get("/api/figures/correlation_heatmap")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert client.get("/api/figures/model_comparison").status_code == 404  # redrawn by the frontend


def test_cors_only_for_vite_dev_server(client: TestClient) -> None:
    ok = client.options("/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    evil = client.options("/api/health", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in evil.headers


# --------------------------------------------------------------------------
# Failure modes of the service itself
# --------------------------------------------------------------------------
def test_missing_models_returns_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prediction, "MODELS_DIR", tmp_path)
    with TestClient(create_app()) as c:
        health = wait_for(c)
        assert health["status"] == "unavailable"
        assert not any(health["checks"]["models"].values())
        for method, url in [("post", "/api/predict/cycle/0"), ("get", "/api/evaluation/valve"), ("get", "/api/labels")]:
            r = getattr(c, method)(url)
            assert r.status_code == 503, url
            assert r.json()["error"] == "models_unavailable"
        up = c.post("/api/predict/upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
        assert up.status_code == 503


def test_warming_up_returns_503_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    original = AppState._load_models

    def slow_load(self: AppState) -> None:
        release.wait(timeout=60)
        original(self)

    monkeypatch.setattr(AppState, "_load_models", slow_load)
    with TestClient(create_app()) as c:
        assert c.get("/api/health").json()["status"] == "warming"
        r = c.post("/api/predict/cycle/0")
        assert r.status_code == 503 and r.json()["error"] == "warming_up"
        assert r.headers["retry-after"] == "2"
        release.set()
        assert wait_for(c)["status"] == "ready"
