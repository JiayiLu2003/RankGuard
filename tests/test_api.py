from __future__ import annotations

import json

from fastapi.testclient import TestClient

from rankguard import api


def test_health_and_recommendation_endpoints(tmp_path, monkeypatch) -> None:
    experiment = {
        "models": [{"id": "demo"}],
        "release_gate": {"candidate": "demo"},
    }
    recommendations = {
        "users": {"1": {"raw_user_id": 1, "history_length": 2, "held_out_title": "Target"}},
        "models": {
            "demo": {
                "1": [
                    {
                        "rank": 1,
                        "item_id": 3,
                        "title": "Movie",
                        "genres": ["Drama"],
                        "training_popularity": 8,
                    }
                ]
            }
        },
    }
    (tmp_path / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
    (tmp_path / "recommendations.json").write_text(json.dumps(recommendations), encoding="utf-8")
    monkeypatch.setenv("RANKGUARD_ARTIFACT_DIR", str(tmp_path))
    api._load_json.cache_clear()
    client = TestClient(api.app)

    assert client.get("/api/health").json()["status"] == "ok"
    response = client.get("/api/users/1/recommendations?model=demo")
    assert response.status_code == 200
    assert response.json()["recommendations"][0]["title"] == "Movie"
    assert client.get("/api/users/999/recommendations?model=demo").status_code == 404


def test_missing_artifacts_return_diagnostic_503(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RANKGUARD_ARTIFACT_DIR", str(tmp_path / "missing"))
    api._load_json.cache_clear()
    client = TestClient(api.app)

    response = client.get("/api/models")

    assert response.status_code == 503
    assert response.json() == {"detail": "Run rankguard-train first"}


def test_frontend_file_resolution_cannot_escape_root(tmp_path) -> None:
    root = tmp_path / "dist"
    asset = root / "assets" / "app.js"
    asset.parent.mkdir(parents=True)
    asset.write_text("console.log('safe')", encoding="utf-8")
    outside = tmp_path / "private.txt"
    outside.write_text("not public", encoding="utf-8")

    assert api._safe_frontend_file(root, "assets/app.js") == asset.resolve()
    assert api._safe_frontend_file(root, "../private.txt") is None


def test_openapi_exposes_typed_response_contracts() -> None:
    schema = api.app.openapi()

    experiment_response = schema["paths"]["/api/experiment"]["get"]["responses"]["200"]
    recommendation_response = schema["paths"][
        "/api/users/{user_id}/recommendations"
    ]["get"]["responses"]["200"]

    assert experiment_response["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ExperimentResponse"
    )
    assert recommendation_response["content"]["application/json"]["schema"]["$ref"].endswith(
        "/RecommendationResponse"
    )
