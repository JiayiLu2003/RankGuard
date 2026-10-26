from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rankguard.config import ExperimentConfig
from rankguard.data import InteractionDataset
from rankguard.evaluation import EvaluationResult

MODEL_LABELS = {
    "popularity": "Popularity",
    "item_knn": "Item-kNN",
    "bpr": "BPR-MF",
    "lightgcn": "LightGCN",
}


def _round_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def build_experiment_artifact(
    *,
    config: ExperimentConfig,
    data: InteractionDataset,
    results: dict[str, EvaluationResult],
    training_seconds: dict[str, float],
    loss_history: dict[str, list[float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_key = f"ndcg@{config.top_k}"
    winner = max(results, key=lambda name: results[name].metrics[metric_key])
    baseline_value = results["popularity"].metrics[metric_key]
    winner_value = results[winner].metrics[metric_key]
    relative_lift = (winner_value - baseline_value) / max(baseline_value, 1e-12)
    winner_cohorts = results[winner].cohorts
    baseline_cohorts = results["popularity"].cohorts
    cohort_regressions = []
    for cohort, values in winner_cohorts.items():
        baseline_ndcg = float(baseline_cohorts[cohort][metric_key])
        winner_ndcg = float(values[metric_key])
        if winner_ndcg < baseline_ndcg:
            cohort_regressions.append(cohort)

    decision = "release" if relative_lift >= 0.05 and not cohort_regressions else "review"
    if decision == "release":
        decision_reason = (
            f"{MODEL_LABELS[winner]} improves {metric_key.upper()} by "
            f"{relative_lift:.1%} over popularity with no measured cohort regression."
        )
    elif cohort_regressions:
        readable_cohorts = ", ".join(name.replace("_", " ") for name in cohort_regressions)
        decision_reason = (
            f"{MODEL_LABELS[winner]} leads overall, but regresses for: {readable_cohorts}."
        )
    else:
        decision_reason = (
            f"{MODEL_LABELS[winner]} leads overall, but its {metric_key.upper()} lift of "
            f"{relative_lift:.1%} is below the 5% release threshold."
        )

    models = []
    for name, result in results.items():
        models.append(
            {
                "id": name,
                "label": MODEL_LABELS[name],
                "metrics": result.metrics,
                "intervals": result.intervals,
                "cohorts": result.cohorts,
                "training_seconds": training_seconds[name],
                "loss_tail": loss_history.get(name, [])[-8:],
            }
        )

    history_lengths = np.asarray([len(items) for items in data.train_by_user])
    experiment = {
        "project": "RankGuard",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": "MovieLens 100K",
            "source": "GroupLens Research",
            "source_url": "https://grouplens.org/datasets/movielens/100k/",
            "license_note": "Dataset is downloaded at runtime and is not redistributed.",
            "users": data.num_users,
            "items": data.num_items,
            "train_interactions": data.num_train_interactions,
            "test_interactions": data.num_users,
            "positive_rating_threshold": config.positive_rating,
            "split": "Per-user chronological leave-last-one-out",
            "candidate_protocol": "Full catalog, excluding training interactions",
            "history_p50": float(np.median(history_lengths)),
        },
        "config": config.serializable(),
        "models": models,
        "release_gate": {
            "decision": decision,
            "candidate": winner,
            "candidate_label": MODEL_LABELS[winner],
            "relative_ndcg_lift_vs_popularity": relative_lift,
            "cohort_regressions": cohort_regressions,
            "reason": decision_reason,
            "rule": (
                "Release when NDCG@K lift >= 5% and no history-length cohort "
                "regresses vs popularity."
            ),
        },
        "limitations": [
            "Offline relevance does not establish online click-through rate or business lift.",
            "MovieLens is a compact research dataset and does not model production traffic scale.",
            "The release gate protects three history-length cohorts; it is not a fairness audit.",
            "Bootstrap intervals quantify user-sampling uncertainty, not distribution shift.",
        ],
    }

    recommendation_payload: dict[str, Any] = {"models": {}, "users": {}}
    for user_index, raw_user_id in enumerate(data.raw_user_ids):
        user_key = str(int(raw_user_id))
        recommendation_payload["users"][user_key] = {
            "raw_user_id": int(raw_user_id),
            "history_length": len(data.train_by_user[user_index]),
            "held_out_title": data.item_titles[int(data.test_item_by_user[user_index])],
        }
    for name, result in results.items():
        model_recommendations: dict[str, Any] = {}
        for user_index, raw_user_id in enumerate(data.raw_user_ids):
            rows = []
            for rank, item_index in enumerate(result.recommendations[user_index], start=1):
                item_index = int(item_index)
                rows.append(
                    {
                        "rank": rank,
                        "item_id": int(data.raw_item_ids[item_index]),
                        "title": data.item_titles[item_index],
                        "genres": data.item_genres[item_index],
                        "training_popularity": int(data.item_popularity[item_index]),
                    }
                )
            model_recommendations[str(int(raw_user_id))] = rows
        recommendation_payload["models"][name] = model_recommendations
    return _round_nested(experiment), recommendation_payload


def write_artifacts(
    target: Path,
    experiment: dict[str, Any],
    recommendations: dict[str, Any],
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "experiment.json").write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (target / "recommendations.json").write_text(
        json.dumps(recommendations, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
