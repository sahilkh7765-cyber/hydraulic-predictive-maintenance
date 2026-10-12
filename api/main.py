"""
main.py
=======
FastAPI entry point. Run from the project root:

    uvicorn api.main:app --port 8000

Read-only apart from prediction: nothing here trains, re-fits or writes to
models/ or outputs/. See api/README.md for every endpoint's response shape.
"""

from __future__ import annotations

import platform
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import AsyncIterator, Literal, Optional

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import analysis as A
from . import schemas as S
from .core import AppState
from .errors import ApiError

Component = Literal["cooler", "valve", "pump", "accumulator"]
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def create_app() -> FastAPI:
    """Build an app with its own AppState (the tests create several)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState()
        app.state.store = state
        state.start()  # warm-up runs in a background thread; /api/health reports progress
        yield

    app = FastAPI(
        title="Hydraulic predictive maintenance API",
        version="1.0.0",
        lifespan=lifespan,
        responses={503: {"model": S.ApiErrorBody}},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Typed errors
    # ------------------------------------------------------------------
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse({"error": exc.code, "detail": exc.detail}, status_code=exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        parts = [f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
        return JSONResponse({"error": "validation_error", "detail": "; ".join(parts)}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "http_error"
        return JSONResponse({"error": code, "detail": str(exc.detail)}, status_code=exc.status_code)

    def store(request: Request) -> AppState:
        return request.app.state.store  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Status and descriptive data
    # ------------------------------------------------------------------
    @app.get("/api/health", response_model=S.Health)
    def health(request: Request) -> S.Health:
        s = store(request)
        return S.Health(
            status=s.status,  # type: ignore[arg-type]
            message=s.message,
            steps=[S.WarmupStep(name=k, done=v) for k, v in s.steps.items()],
            checks=S.HealthChecks(**s.checks),  # type: ignore[arg-type]
            versions={
                "python": platform.python_version(),
                "scikit-learn": _pkg("scikit-learn"),
                "xgboost": _pkg("xgboost"),
                "pandas": _pkg("pandas"),
                "numpy": _pkg("numpy"),
            },
        )

    @app.get("/api/overview", response_model=S.Overview)
    def overview(request: Request) -> S.Overview:
        return A.overview(store(request))

    @app.get("/api/sensors", response_model=S.SensorsResponse)
    def sensors() -> S.SensorsResponse:
        return A.sensors()

    @app.get("/api/labels", response_model=S.LabelsResponse)
    def labels(request: Request) -> S.LabelsResponse:
        return A.labels(store(request))

    @app.get("/api/cycles/{cycle_id}", response_model=S.CycleResponse, responses={404: {"model": S.ApiErrorBody}})
    def cycle(request: Request, cycle_id: int) -> S.CycleResponse:
        return A.cycle(store(request), cycle_id)

    @app.get("/api/cycles/{cycle_id}/csv", response_class=Response, responses={404: {"model": S.ApiErrorBody}})
    def cycle_csv(request: Request, cycle_id: int) -> Response:
        body = A.cycle_csv(store(request), cycle_id)
        return Response(
            body,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="cycle_{cycle_id}.csv"'},
        )

    # ------------------------------------------------------------------
    # Metrics and features
    # ------------------------------------------------------------------
    @app.get("/api/metrics", response_model=S.MetricsResponse)
    def metrics(request: Request) -> S.MetricsResponse:
        return A.metrics(store(request))

    @app.get("/api/features", response_model=S.FeaturesResponse)
    def features(request: Request) -> S.FeaturesResponse:
        return A.features(store(request))

    @app.get("/api/features/importance", response_model=S.ImportanceResponse)
    def feature_importance(
        request: Request, component: Component, top: int = Query(20, ge=1, le=100)
    ) -> S.ImportanceResponse:
        return A.importance(store(request), component, top)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @app.post(
        "/api/predict/cycle/{cycle_id}",
        response_model=S.PredictionResponse,
        responses={404: {"model": S.ApiErrorBody}},
    )
    def predict_cycle(request: Request, cycle_id: int) -> S.PredictionResponse:
        return A.predict_dataset_cycle(store(request), cycle_id)

    @app.post("/api/predict/upload", response_model=S.PredictionResponse, responses={400: {"model": S.ApiErrorBody}})
    def predict_upload(request: Request, file: UploadFile = File(...)) -> S.PredictionResponse:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise ApiError(413, "file_too_large", "The CSV is larger than 25 MB; one cycle is about 1 MB.")
        return A.predict_upload(store(request), content, file.filename)

    # ------------------------------------------------------------------
    # Models and evaluation
    # ------------------------------------------------------------------
    @app.get("/api/models/{component}/pipeline", response_model=S.PipelineResponse)
    def pipeline(request: Request, component: Component) -> S.PipelineResponse:
        return A.pipeline(store(request), component)

    @app.get("/api/evaluation/{component}", response_model=S.EvaluationResponse)
    def evaluation(request: Request, component: Component) -> S.EvaluationResponse:
        return A.evaluation(store(request), component)

    @app.get("/api/generalization", response_model=S.GeneralizationResponse)
    def generalization(request: Request) -> S.GeneralizationResponse:
        return A.generalization(store(request))

    @app.get("/api/generalization/{component}/timeline", response_model=S.TimelineResponse)
    def timeline(
        request: Request, component: Component, start: int = Query(0, ge=0), end: int = Query(300, ge=0)
    ) -> S.TimelineResponse:
        return A.timeline(store(request), component, start, end)

    # ------------------------------------------------------------------
    # Robustness and errors
    # ------------------------------------------------------------------
    @app.get("/api/errors/{component}", response_model=S.ErrorCasesResponse)
    def error_cases(request: Request, component: Component) -> S.ErrorCasesResponse:
        return A.error_cases(store(request), component)

    @app.get("/api/errors/{component}/{cycle_id}", response_model=S.ErrorProfileResponse)
    def error_profile(request: Request, component: Component, cycle_id: int) -> S.ErrorProfileResponse:
        return A.error_profile(store(request), component, cycle_id)

    @app.get("/api/whatif/{component}", response_model=S.WhatIfOptions)
    def whatif_options(request: Request, component: Component, cycle: Optional[int] = None) -> S.WhatIfOptions:
        return A.whatif_options(store(request), component, cycle)

    @app.post("/api/whatif/{component}", response_model=S.WhatIfResponse)
    def whatif(request: Request, component: Component, body: S.WhatIfRequest) -> S.WhatIfResponse:
        return A.whatif(store(request), component, body)

    @app.get("/api/robustness/{component}/noise", response_model=S.NoiseResponse)
    def noise(request: Request, component: Component) -> S.NoiseResponse:
        return A.noise(store(request), component)

    @app.get("/api/robustness/{component}/selective", response_model=S.SelectiveResponse)
    def selective(request: Request, component: Component) -> S.SelectiveResponse:
        return A.selective(store(request), component)

    # ------------------------------------------------------------------
    # Static figures the frontend does not redraw (PCA, correlation heatmap)
    # ------------------------------------------------------------------
    @app.get("/api/figures", response_model=S.FiguresResponse)
    def figures(request: Request) -> S.FiguresResponse:
        return A.figures(store(request))

    @app.get("/api/figures/{key}", response_class=FileResponse)
    def figure(request: Request, key: str) -> FileResponse:
        entry = A.figure_whitelist(store(request)).get(key)
        if entry is None or not entry[1].exists():
            raise ApiError(404, "figure_not_found", f"No figure named '{key}'.")
        return FileResponse(entry[1], media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    return app


app = create_app()
