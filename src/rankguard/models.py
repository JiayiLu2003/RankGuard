from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from rankguard.data import InteractionDataset


class Recommender(ABC):
    name: str

    @abstractmethod
    def fit(self, data: InteractionDataset) -> Recommender:
        raise NotImplementedError

    @abstractmethod
    def score_matrix(self) -> np.ndarray:
        raise NotImplementedError


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self) -> None:
        self._scores: np.ndarray | None = None

    def fit(self, data: InteractionDataset) -> PopularityRecommender:
        item_scores = np.log1p(data.item_popularity).astype(np.float32)
        self._scores = np.broadcast_to(item_scores, (data.num_users, data.num_items)).copy()
        return self

    def score_matrix(self) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("fit must be called before score_matrix")
        return self._scores


class ItemKNNRecommender(Recommender):
    name = "item_knn"

    def __init__(self, neighbors: int = 80) -> None:
        self.neighbors = neighbors
        self._scores: np.ndarray | None = None

    def fit(self, data: InteractionDataset) -> ItemKNNRecommender:
        rows = data.train_pairs[:, 0]
        cols = data.train_pairs[:, 1]
        values = np.ones(len(rows), dtype=np.float32)
        matrix = sp.csr_matrix((values, (rows, cols)), shape=(data.num_users, data.num_items))
        norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
        inverse_norms = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        normalized = matrix @ sp.diags(inverse_norms)
        similarity = (normalized.T @ normalized).tolil()
        similarity.setdiag(0.0)
        similarity = similarity.tocsr()

        if self.neighbors < data.num_items:
            trimmed = similarity.tolil()
            for item in range(data.num_items):
                row_values = similarity.getrow(item).toarray().ravel()
                if np.count_nonzero(row_values) > self.neighbors:
                    keep = np.argpartition(row_values, -self.neighbors)[-self.neighbors :]
                    mask = np.ones(data.num_items, dtype=bool)
                    mask[keep] = False
                    trimmed[item, np.flatnonzero(mask)] = 0.0
            similarity = trimmed.tocsr()
            similarity.eliminate_zeros()
        self._scores = (matrix @ similarity).toarray().astype(np.float32)
        return self

    def score_matrix(self) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("fit must be called before score_matrix")
        return self._scores


class BPRModule(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, dim)
        self.item_embedding = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_embedding.weight, self.item_embedding.weight


class LightGCNModule(BPRModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        dim: int,
        layers: int,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__(num_users, num_items, dim)
        self.layers = layers
        self.register_buffer("adjacency", adjacency)
        self.num_users = num_users

    def embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        initial = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_embeddings = [initial]
        current = initial
        for _ in range(self.layers):
            current = torch.sparse.mm(self.adjacency, current)
            layer_embeddings.append(current)
        combined = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return combined[: self.num_users], combined[self.num_users :]


class TorchBPRRecommender(Recommender):
    def __init__(
        self,
        *,
        name: str,
        embedding_dim: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        seed: int,
        device: str,
        lightgcn_layers: int | None = None,
    ) -> None:
        self.name = name
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self.lightgcn_layers = lightgcn_layers
        self.loss_history: list[float] = []
        self._scores: np.ndarray | None = None

    @staticmethod
    def _normalized_adjacency(data: InteractionDataset, device: torch.device) -> torch.Tensor:
        user_nodes = data.train_pairs[:, 0]
        item_nodes = data.train_pairs[:, 1] + data.num_users
        src = np.concatenate([user_nodes, item_nodes])
        dst = np.concatenate([item_nodes, user_nodes])
        degrees = np.bincount(src, minlength=data.num_users + data.num_items).astype(np.float32)
        weights = 1.0 / np.sqrt(np.maximum(degrees[src] * degrees[dst], 1.0))
        indices = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
        values = torch.tensor(weights, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=(data.num_users + data.num_items, data.num_users + data.num_items),
            device=device,
            check_invariants=False,
        ).coalesce()

    @staticmethod
    def _sample_negatives(
        rng: np.random.Generator,
        users: np.ndarray,
        train_sets: list[set[int]],
        num_items: int,
    ) -> np.ndarray:
        negatives = rng.integers(0, num_items, size=len(users), dtype=np.int64)
        invalid = np.fromiter(
            (item in train_sets[user] for user, item in zip(users, negatives, strict=True)),
            dtype=bool,
            count=len(users),
        )
        while invalid.any():
            negatives[invalid] = rng.integers(0, num_items, size=int(invalid.sum()), dtype=np.int64)
            invalid = np.fromiter(
                (item in train_sets[user] for user, item in zip(users, negatives, strict=True)),
                dtype=bool,
                count=len(users),
            )
        return negatives

    def fit(self, data: InteractionDataset) -> TorchBPRRecommender:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        resolved_device = (
            "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        )
        if resolved_device == "auto":
            resolved_device = "cpu"
        device = torch.device(resolved_device)

        if self.lightgcn_layers is None:
            model: BPRModule = BPRModule(data.num_users, data.num_items, self.embedding_dim)
        else:
            adjacency = self._normalized_adjacency(data, device)
            model = LightGCNModule(
                data.num_users,
                data.num_items,
                self.embedding_dim,
                self.lightgcn_layers,
                adjacency,
            )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        users_np = data.train_pairs[:, 0]
        positives_np = data.train_pairs[:, 1]
        users = torch.tensor(users_np, dtype=torch.long, device=device)
        positives = torch.tensor(positives_np, dtype=torch.long, device=device)
        train_sets = [set(items) for items in data.train_by_user]
        rng = np.random.default_rng(self.seed)

        for _ in range(self.epochs):
            negatives_np = self._sample_negatives(rng, users_np, train_sets, data.num_items)
            negatives = torch.tensor(negatives_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            user_embeddings, item_embeddings = model.embeddings()
            user_vectors = user_embeddings[users]
            positive_vectors = item_embeddings[positives]
            negative_vectors = item_embeddings[negatives]
            positive_scores = (user_vectors * positive_vectors).sum(dim=1)
            negative_scores = (user_vectors * negative_vectors).sum(dim=1)
            ranking_loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()
            regularization = (
                user_vectors.square().mean()
                + positive_vectors.square().mean()
                + negative_vectors.square().mean()
            )
            loss = ranking_loss + self.weight_decay * regularization
            loss.backward()
            optimizer.step()
            self.loss_history.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings = model.embeddings()
            self._scores = (user_embeddings @ item_embeddings.T).detach().cpu().numpy()
        return self

    def score_matrix(self) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("fit must be called before score_matrix")
        return self._scores
