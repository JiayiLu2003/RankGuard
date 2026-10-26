from __future__ import annotations

import numpy as np

from rankguard.artifacts import build_experiment_artifact
from rankguard.config import ExperimentConfig
from rankguard.data import InteractionDataset
from rankguard.evaluation import EvaluationResult


def _result(ndcg: float, cohort_ndcg: float) -> EvaluationResult:
    return EvaluationResult(
        metrics={
            "recall@10": ndcg,
            "ndcg@10": ndcg,
            "mrr@10": ndcg,
            "catalog_coverage@10": 0.5,
            "novelty@10": 4.0,
            "long_tail_share@10": 0.4,
            "ranking_latency_ms_per_user": 0.01,
        },
        intervals={"recall@10": [ndcg, ndcg], "ndcg@10": [ndcg, ndcg], "mrr@10": [ndcg, ndcg]},
        cohorts={
            name: {"users": 1, "recall@10": cohort_ndcg, "ndcg@10": cohort_ndcg}
            for name in ["short_history", "medium_history", "deep_history"]
        },
        per_user={"hit": np.ones(1), "ndcg": np.ones(1), "mrr": np.ones(1)},
        recommendations=np.asarray([[1]]),
    )


def test_release_gate_uses_lift_and_cohort_guard() -> None:
    data = InteractionDataset(
        train_by_user=[[0]],
        test_item_by_user=np.asarray([1]),
        raw_user_ids=np.asarray([7]),
        raw_item_ids=np.asarray([1, 2]),
        item_titles=["A", "B"],
        item_genres=[[], []],
        item_popularity=np.asarray([1, 0]),
        train_pairs=np.asarray([[0, 0]]),
    )
    results = {
        "popularity": _result(0.1, 0.1),
        "item_knn": _result(0.11, 0.11),
        "bpr": _result(0.15, 0.15),
        "lightgcn": _result(0.13, 0.13),
    }
    artifact, recommendations = build_experiment_artifact(
        config=ExperimentConfig(),
        data=data,
        results=results,
        training_seconds={name: 1.0 for name in results},
        loss_history={},
    )

    assert artifact["release_gate"]["decision"] == "release"
    assert artifact["release_gate"]["candidate"] == "bpr"
    assert recommendations["models"]["bpr"]["7"][0]["title"] == "B"
