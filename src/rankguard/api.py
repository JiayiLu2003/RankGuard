from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rankguard.schemas import (
    ExperimentResponse,
    ModelResult,
    RecommendationResponse,
    ReleaseGate,
    UserSummary,
)


def artifact_root() -> Path:
    return Path(os.getenv("RANKGUARD_ARTIFACT_DIR", "artifacts/latest"))


@lru_cache(maxsize=4)
def _load_json(path: str) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


def load_experiment() -> dict[str, Any]:
    return _load_json(str(artifact_root() / "experiment.json"))


def load_recommendations() -> dict[str, Any]:
    return _load_json(str(artifact_root() / "recommendations.json"))


app = FastAPI(
    title="RankGuard API",
    version="0.1.0",
    description="Evidence-first recommendation evaluation and serving API",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(FileNotFoundError)
def missing_artifact(_request: Request, _error: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Run rankguard-train first"})


@app.get("/api/health")
def health() -> dict[str, str]:
    try:
        load_experiment()
    except FileNotFoundError:
        return {"status": "degraded", "artifacts": "missing"}
    return {"status": "ok", "artifacts": "loaded"}


@app.get("/api/experiment", response_model=ExperimentResponse)
def experiment() -> dict[str, Any]:
    try:
        return load_experiment()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Run rankguard-train first") from error


@app.get("/api/models", response_model=list[ModelResult])
def models() -> list[dict[str, Any]]:
    return load_experiment()["models"]


@app.get("/api/release", response_model=ReleaseGate)
def release() -> dict[str, Any]:
    return load_experiment()["release_gate"]


@app.get("/api/users", response_model=list[UserSummary])
def users(limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
    payload = load_recommendations()["users"]
    return list(payload.values())[:limit]


@app.get("/api/users/{user_id}/recommendations", response_model=RecommendationResponse)
def recommend(
    user_id: int,
    model: str = Query(default="lightgcn"),
) -> dict[str, Any]:
    payload = load_recommendations()
    user_key = str(user_id)
    if user_key not in payload["users"]:
        raise HTTPException(status_code=404, detail="User is not in the evaluated cohort")
    if model not in payload["models"]:
        raise HTTPException(status_code=404, detail="Unknown model")
    return {
        "model": model,
        "user": payload["users"][user_key],
        "recommendations": payload["models"][model][user_key],
    }


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def _safe_frontend_file(root: Path, requested_path: str) -> Path | None:
    resolved_root = root.resolve()
    candidate = (resolved_root / requested_path).resolve()
    if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
        return None
    return candidate


if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        candidate = _safe_frontend_file(frontend_dist, full_path)
        if full_path and candidate is not None:
            return FileResponse(candidate)
        return FileResponse(frontend_dist / "index.html")


def run() -> None:
    uvicorn.run("rankguard.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
