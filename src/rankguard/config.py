from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts/latest")
    positive_rating: float = 4.0
    min_user_positives: int = 5
    top_k: int = 10
    embedding_dim: int = 48
    bpr_epochs: int = 35
    lightgcn_epochs: int = 45
    lightgcn_layers: int = 2
    learning_rate: float = 0.03
    weight_decay: float = 1e-5
    bootstrap_samples: int = 500
    seed: int = 2026
    device: str = "auto"

    def serializable(self) -> dict[str, object]:
        values = asdict(self)
        values["data_dir"] = str(self.data_dir)
        values["artifact_dir"] = str(self.artifact_dir)
        return values
