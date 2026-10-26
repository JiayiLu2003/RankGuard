from __future__ import annotations

import argparse
import time
from pathlib import Path

from rankguard.artifacts import build_experiment_artifact, write_artifacts
from rankguard.config import ExperimentConfig
from rankguard.data import load_movielens
from rankguard.evaluation import EvaluationResult, evaluate
from rankguard.models import (
    ItemKNNRecommender,
    PopularityRecommender,
    Recommender,
    TorchBPRRecommender,
)


def run(config: ExperimentConfig) -> dict[str, object]:
    data = load_movielens(
        config.data_dir,
        positive_rating=config.positive_rating,
        min_user_positives=config.min_user_positives,
    )
    models: list[Recommender] = [
        PopularityRecommender(),
        ItemKNNRecommender(neighbors=80),
        TorchBPRRecommender(
            name="bpr",
            embedding_dim=config.embedding_dim,
            epochs=config.bpr_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            seed=config.seed,
            device=config.device,
        ),
        TorchBPRRecommender(
            name="lightgcn",
            embedding_dim=config.embedding_dim,
            epochs=config.lightgcn_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            seed=config.seed,
            device=config.device,
            lightgcn_layers=config.lightgcn_layers,
        ),
    ]
    results: dict[str, EvaluationResult] = {}
    training_seconds: dict[str, float] = {}
    loss_history: dict[str, list[float]] = {}

    for model in models:
        started = time.perf_counter()
        model.fit(data)
        training_seconds[model.name] = time.perf_counter() - started
        results[model.name] = evaluate(
            model.score_matrix(),
            data,
            k=config.top_k,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed,
        )
        loss_history[model.name] = getattr(model, "loss_history", [])
        metric = results[model.name].metrics[f"ndcg@{config.top_k}"]
        print(f"{model.name:>10} | NDCG@{config.top_k}: {metric:.4f}")

    experiment, recommendations = build_experiment_artifact(
        config=config,
        data=data,
        results=results,
        training_seconds=training_seconds,
        loss_history=loss_history,
    )
    write_artifacts(config.artifact_dir, experiment, recommendations)
    return experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate RankGuard models")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/latest"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs for a smoke run")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = ExperimentConfig(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        device=args.device,
        bpr_epochs=5 if args.quick else 35,
        lightgcn_epochs=6 if args.quick else 45,
        bootstrap_samples=100 if args.quick else 500,
    )
    run(config)


if __name__ == "__main__":
    main()
