from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rankguard.data import InteractionDataset


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    intervals: dict[str, list[float]]
    cohorts: dict[str, dict[str, float | int]]
    per_user: dict[str, np.ndarray]
    recommendations: np.ndarray


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    candidate = np.argpartition(scores, -k, axis=1)[:, -k:]
    candidate_scores = np.take_along_axis(scores, candidate, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidate, order, axis=1)


def _bootstrap_interval(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    if len(values) == 0:
        return [0.0, 0.0]
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(values), size=len(values))
        means[index] = values[draw].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def evaluate(
    scores: np.ndarray,
    data: InteractionDataset,
    *,
    k: int,
    bootstrap_samples: int,
    seed: int,
) -> EvaluationResult:
    if scores.shape != (data.num_users, data.num_items):
        raise ValueError(
            f"score matrix has shape {scores.shape}; expected {(data.num_users, data.num_items)}"
        )
    masked_scores = scores.copy()
    for user, seen_items in enumerate(data.train_by_user):
        masked_scores[user, seen_items] = -np.inf

    started = time.perf_counter()
    recommendations = _top_k(masked_scores, k)
    ranking_latency_ms = (time.perf_counter() - started) * 1000 / data.num_users
    matches = recommendations == data.test_item_by_user[:, None]
    hits = matches.any(axis=1).astype(np.float64)
    positions = np.argmax(matches, axis=1)
    ndcg = np.where(hits > 0, 1.0 / np.log2(positions + 2), 0.0)
    reciprocal_rank = np.where(hits > 0, 1.0 / (positions + 1), 0.0)

    recommended_items = recommendations.ravel()
    coverage = len(np.unique(recommended_items)) / data.num_items
    popularity_probability = (data.item_popularity + 1) / (
        data.num_train_interactions + data.num_items
    )
    novelty = float((-np.log2(popularity_probability[recommended_items])).mean())
    long_tail_cutoff = float(np.quantile(data.item_popularity, 0.8))
    long_tail_share = float(
        (data.item_popularity[recommended_items] <= long_tail_cutoff).mean()
    )

    metrics = {
        f"recall@{k}": float(hits.mean()),
        f"ndcg@{k}": float(ndcg.mean()),
        f"mrr@{k}": float(reciprocal_rank.mean()),
        f"catalog_coverage@{k}": float(coverage),
        f"novelty@{k}": novelty,
        f"long_tail_share@{k}": long_tail_share,
        "ranking_latency_ms_per_user": float(ranking_latency_ms),
    }

    rng = np.random.default_rng(seed)
    intervals = {
        f"recall@{k}": _bootstrap_interval(hits, bootstrap_samples, rng),
        f"ndcg@{k}": _bootstrap_interval(ndcg, bootstrap_samples, rng),
        f"mrr@{k}": _bootstrap_interval(reciprocal_rank, bootstrap_samples, rng),
    }
    history_lengths = np.asarray([len(items) for items in data.train_by_user])
    cohort_masks = {
        "short_history": history_lengths <= 20,
        "medium_history": (history_lengths > 20) & (history_lengths <= 50),
        "deep_history": history_lengths > 50,
    }
    cohorts: dict[str, dict[str, float | int]] = {}
    for name, mask in cohort_masks.items():
        cohorts[name] = {
            "users": int(mask.sum()),
            f"recall@{k}": float(hits[mask].mean()) if mask.any() else 0.0,
            f"ndcg@{k}": float(ndcg[mask].mean()) if mask.any() else 0.0,
        }

    return EvaluationResult(
        metrics=metrics,
        intervals=intervals,
        cohorts=cohorts,
        per_user={"hit": hits, "ndcg": ndcg, "mrr": reciprocal_rank},
        recommendations=recommendations,
    )
