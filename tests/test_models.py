from __future__ import annotations

import numpy as np
import pytest

from rankguard.data import InteractionDataset
from rankguard.models import (
    ItemKNNRecommender,
    PopularityRecommender,
    TorchBPRRecommender,
)


def tiny_dataset() -> InteractionDataset:
    train_by_user = [[0, 1], [1, 2], [2, 3], [3, 4]]
    train_pairs = np.asarray(
        [(user, item) for user, items in enumerate(train_by_user) for item in items],
        dtype=np.int64,
    )
    return InteractionDataset(
        train_by_user=train_by_user,
        test_item_by_user=np.asarray([2, 3, 4, 5]),
        raw_user_ids=np.asarray([10, 20, 30, 40]),
        raw_item_ids=np.asarray([100, 101, 102, 103, 104, 105]),
        item_titles=["A", "B", "C", "D", "E", "F"],
        item_genres=[[] for _ in range(6)],
        item_popularity=np.bincount(train_pairs[:, 1], minlength=6),
        train_pairs=train_pairs,
    )


def test_classical_recommenders_produce_finite_score_matrices() -> None:
    data = tiny_dataset()

    popularity = PopularityRecommender().fit(data).score_matrix()
    item_knn = ItemKNNRecommender(neighbors=2).fit(data).score_matrix()

    assert popularity.shape == (data.num_users, data.num_items)
    assert item_knn.shape == (data.num_users, data.num_items)
    assert np.isfinite(popularity).all()
    assert np.isfinite(item_knn).all()
    assert popularity[0, 1] > popularity[0, 5]


@pytest.mark.parametrize(
    ("name", "layers"),
    [("bpr", None), ("lightgcn", 1)],
)
def test_torch_recommenders_train_on_cpu(name: str, layers: int | None) -> None:
    data = tiny_dataset()
    model = TorchBPRRecommender(
        name=name,
        embedding_dim=4,
        epochs=2,
        learning_rate=0.01,
        weight_decay=1e-5,
        seed=7,
        device="cpu",
        lightgcn_layers=layers,
    ).fit(data)

    scores = model.score_matrix()

    assert scores.shape == (data.num_users, data.num_items)
    assert np.isfinite(scores).all()
    assert len(model.loss_history) == 2
    assert all(np.isfinite(model.loss_history))


def test_negative_sampler_never_returns_seen_items() -> None:
    data = tiny_dataset()
    users = data.train_pairs[:, 0]
    train_sets = [set(items) for items in data.train_by_user]

    negatives = TorchBPRRecommender._sample_negatives(
        np.random.default_rng(17),
        users,
        train_sets,
        data.num_items,
    )

    assert all(
        item not in train_sets[user]
        for user, item in zip(users, negatives, strict=True)
    )
