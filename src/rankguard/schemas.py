from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    name: str
    source: str
    source_url: str
    license_note: str
    users: int
    items: int
    train_interactions: int
    test_interactions: int
    positive_rating_threshold: float
    split: str
    candidate_protocol: str
    history_p50: float


class ExperimentSettings(BaseModel):
    data_dir: str
    artifact_dir: str
    positive_rating: float
    min_user_positives: int
    top_k: int
    embedding_dim: int
    bpr_epochs: int
    lightgcn_epochs: int
    lightgcn_layers: int
    learning_rate: float
    weight_decay: float
    bootstrap_samples: int
    seed: int
    device: str


class ModelResult(BaseModel):
    id: str
    label: str
    metrics: dict[str, float]
    intervals: dict[str, list[float]]
    cohorts: dict[str, dict[str, int | float]]
    training_seconds: float
    loss_tail: list[float]


class ReleaseGate(BaseModel):
    decision: str
    candidate: str
    candidate_label: str
    relative_ndcg_lift_vs_popularity: float
    cohort_regressions: list[str]
    reason: str
    rule: str


class ExperimentResponse(BaseModel):
    project: str
    generated_at: datetime
    dataset: DatasetSummary
    config: ExperimentSettings
    models: list[ModelResult]
    release_gate: ReleaseGate
    limitations: list[str]


class UserSummary(BaseModel):
    raw_user_id: int
    history_length: int
    held_out_title: str


class RecommendationItem(BaseModel):
    rank: int
    item_id: int
    title: str
    genres: list[str]
    training_popularity: int


class RecommendationResponse(BaseModel):
    model: str
    user: UserSummary
    recommendations: list[RecommendationItem]
