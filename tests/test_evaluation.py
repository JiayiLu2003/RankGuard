from __future__ import annotations

import numpy as np

from rankguard.data import InteractionDataset
from rankguard.evaluation import evaluate


def dataset() -> InteractionDataset:
    return InteractionDataset(
        train_by_user=[[0, 1], [1, 2], [2, 3]],
        test_item_by_user=np.asarray([2, 3, 4]),
        raw_user_ids=np.asarray([10, 20, 30]),
        raw_item_ids=np.asarray([100, 101, 102, 103, 104]),
        item_titles=["A", "B", "C", "D", "E"],
        item_genres=[[] for _ in range(5)],
        item_popularity=np.asarray([1, 2, 2, 1, 0]),
        train_pairs=np.asarray([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3]]),
    )


def test_evaluation_masks_seen_items_and_finds_all_targets() -> None:
    data = dataset()
    scores = np.asarray(
        [
            [99.0, 98.0, 4.0, 3.0, 2.0],
            [1.0, 99.0, 98.0, 5.0, 0.0],
            [1.0, 2.0, 99.0, 98.0, 5.0],
        ]
    )
    result = evaluate(scores, data, k=2, bootstrap_samples=50, seed=7)

    assert result.metrics["recall@2"] == 1.0
    assert result.metrics["ndcg@2"] == 1.0
    assert not any(
        item in data.train_by_user[user]
        for user, row in enumerate(result.recommendations)
        for item in row
    )


def test_bootstrap_interval_contains_mean() -> None:
    data = dataset()
    scores = np.zeros((3, 5), dtype=np.float32)
    scores[0, 2] = 2
    scores[1, 3] = 2
    result = evaluate(scores, data, k=1, bootstrap_samples=100, seed=11)
    low, high = result.intervals["recall@1"]

    assert low <= result.metrics["recall@1"] <= high
