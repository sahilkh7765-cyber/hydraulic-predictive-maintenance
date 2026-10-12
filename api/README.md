# Hydraulic PdM API

A thin FastAPI layer over `src/`. It loads the four saved pipelines once,
exposes the saved metrics, and scores cycles through `src/prediction.py`.
It never trains, re-fits or writes to `models/` or `outputs/`.

```powershell
# from the project root, inside the venv
uvicorn api.main:app --port 8000
python -m pytest api/tests -q        # 36 tests, ~10 s
```

Interactive docs: <http://127.0.0.1:8000/docs> (generated from `api/schemas.py`).

## Lifecycle

On startup a background thread loads every artefact once (about 3 s on a laptop):
metrics CSVs and `run_metadata.json` → the four pipelines → the engineered
feature table → test-set scoring → training-cycle embeddings for neighbour
search. Until it finishes, every endpoint except `/api/health` and
`/api/sensors` answers **503 `warming_up`** with `Retry-After: 2`, and the
frontend shows a "loading models" screen.

## Errors

Every error body is `{"error": "<code>", "detail": "<human message>"}`.

| Status | `error` | When |
|---|---|---|
| 400 | `invalid_csv` | upload is empty, unparseable, missing a sensor column, or has the wrong sample count (message from `load_cycle_csv`) |
| 400 | `invalid_window`, `unknown_feature`, `invalid_value`, `invalid_cycle` | bad query/body values |
| 404 | `cycle_not_found` | cycle id outside 0 … 2204 |
| 404 | `figure_not_found`, `not_found` | unknown figure / route |
| 413 | `file_too_large` | upload above 25 MB |
| 422 | `validation_error` | wrong types, unknown component key, missing upload file |
| 503 | `warming_up` | artefacts still loading |
| 503 | `models_unavailable` | `models/<component>/` incomplete (run `python src/train.py`) |
| 503 | `raw_data_unavailable`, `feature_table_unavailable`, `metrics_unavailable` | the corresponding files are missing |

CORS allows only the Vite dev server (`http://localhost:5173`, `http://127.0.0.1:5173`).

## Endpoints

`{component}` is one of `cooler | valve | pump | accumulator`. Condition classes
are always identified by their documented value (valve `73/80/90/100`, …).

| Method | Path | Source of the numbers |
|---|---|---|
| GET | `/api/health` | file checks + warm-up progress + package versions |
| GET | `/api/overview` | `models/run_metadata.json`, `outputs/metrics/test_results.csv`, `model_results.csv` |
| GET | `/api/sensors` | `SENSOR_SPECS` in `src/data_loader.py` |
| GET | `/api/labels` | `data/raw/profile.txt` |
| GET | `/api/cycles/{id}` | `load_cycle_row` (100 Hz channels min/max-decimated to ≤600 points) |
| GET | `/api/cycles/{id}/csv` | `export_cycle_csv`: a valid upload template |
| GET | `/api/metrics` | `model_results.csv`, `test_results.csv`, `feature_experiment_effect.csv` |
| GET | `/api/features` | `run_metadata.json` + each model's `metadata.json` |
| GET | `/api/features/importance?component=&top=20` | `feature_importance.csv` |
| POST | `/api/predict/cycle/{id}` | `load_cycle_row` → `predict_system_condition` → `explain_prediction` |
| POST | `/api/predict/upload` (multipart `file`) | `load_cycle_csv` → same as above |
| GET | `/api/models/{component}/pipeline` | the fitted pipeline object + `metadata.json` |
| GET | `/api/evaluation/{component}` | saved pipeline scored live on `split_indices.npz` test cycles |
| GET | `/api/generalization` | `test_results.csv` |
| GET | `/api/generalization/{component}/timeline?start=0&end=300` | labels + saved split |
| GET | `/api/errors/{component}` and `/api/errors/{component}/{cycle}` | live test-set scores, `feature_importance.csv` |
| GET / POST | `/api/whatif/{component}` | saved pipeline on a modified feature row |
| GET | `/api/robustness/{component}/noise` | live, seed 42, cached per process |
| GET | `/api/robustness/{component}/selective` | live test-set scores |
| GET | `/api/figures`, `/api/figures/{key}` | PCA and correlation-heatmap PNGs only |

The evaluation, error, what-if and robustness computations replicate `MLapp.py`
(same bins, thresholds, noise levels, seed). A test asserts that the live
test-set accuracy, macro-F1 and ROC-AUC equal `test_results.csv` exactly.

### Example: `POST /api/predict/cycle/2100` (abridged)

```json
{
  "source": {"kind": "dataset", "cycle_id": 2100, "filename": null},
  "system_status": "Critical",
  "driving_component": "Internal pump leakage",
  "mean_confidence": 0.98811,
  "n_engineered_features": 626,
  "components": [
    {
      "component": "valve", "display_name": "Valve condition", "unit": "%",
      "predicted_value": 90, "predicted_label": "small lag", "severity": "Degraded",
      "confidence": 0.95667, "margin": 0.93667, "entropy": 0.15716,
      "probabilities": [{"value": 73, "label": "close to total failure", "severity": "Critical", "p": 0.00333}, "…"],
      "model": "random_forest", "feature_set": "selected", "n_features": 40,
      "explanation": [{"rank": 1, "feature": "PS2_cv", "value": 0.43198, "importance": 0.22153, "block": "statistical"}, "…"],
      "neighbours": [{"cycle": 2098, "distance": 0.60096, "true_value": 90, "agrees": true}, "…"],
      "reliability": {"lo": 0.95, "hi": 1.0, "n": 321, "accuracy": 1.0},
      "true_value": 90, "true_label": "small lag", "correct": true, "in_test_set": false
    }
  ]
}
```

An upload returns the same shape with `source.kind = "upload"` and the
`true_*`, `correct` and `in_test_set` fields set to `null`.

### Example: `GET /api/evaluation/accumulator` (abridged)

```json
{
  "component": "accumulator", "n_test": 441,
  "accuracy": 0.98866, "f1_macro": 0.98811, "roc_auc": 0.99988, "ece": 0.05555, "n_errors": 5,
  "classes": [{"value": 90, "label": "close to total failure", "severity": "Critical"}, "…"],
  "confusion": [[161, 0, 0, 0], "…"],
  "report": [{"value": 90, "label": "close to total failure", "precision": 0.96988, "recall": 1.0, "f1": 0.98471, "support": 161}, "…"],
  "roc": [{"value": 90, "label": "close to total failure", "score": 0.99996, "points": [{"x": 0, "y": 0}, "…"]}, "…"],
  "pr":  ["…same shape, score = average precision…"],
  "calibration": [{"mean_confidence": 0.4, "accuracy": 0.0, "count": 1}, "…"],
  "confidence_hist": [{"lo": 0.0, "hi": 0.05, "correct": 0, "wrong": 0}, "…"],
  "misclassified": [{"cycle": 499, "true_value": 100, "predicted_value": 90, "confidence": 0.49333, "p_true": 0.32333, "stable_flag": 0}, "…"],
  "saved_report_text": "Hydraulic accumulator - held-out test set (441 cycles) …",
  "parity": {"saved_accuracy": 0.98866, "saved_f1_macro": 0.98811, "matches": true}
}
```

All other shapes are defined in `api/schemas.py` and mirrored as zod schemas in
`frontend/src/lib/schemas.ts`.
