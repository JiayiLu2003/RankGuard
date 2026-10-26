from __future__ import annotations

import io
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


@dataclass
class InteractionDataset:
    train_by_user: list[list[int]]
    test_item_by_user: np.ndarray
    raw_user_ids: np.ndarray
    raw_item_ids: np.ndarray
    item_titles: list[str]
    item_genres: list[list[str]]
    item_popularity: np.ndarray
    train_pairs: np.ndarray

    @property
    def num_users(self) -> int:
        return len(self.train_by_user)

    @property
    def num_items(self) -> int:
        return len(self.raw_item_ids)

    @property
    def num_train_interactions(self) -> int:
        return int(self.train_pairs.shape[0])


def _tls_context() -> ssl.SSLContext:
    try:
        import truststore
    except ImportError:
        return ssl.create_default_context(cafile=certifi.where())
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def _download_and_extract(root: Path) -> Path:
    target = root / "ml-100k"
    if (target / "u.data").exists() and (target / "u.item").exists():
        return target

    root.mkdir(parents=True, exist_ok=True)
    tls_context = _tls_context()
    with urllib.request.urlopen(  # noqa: S310
        MOVIELENS_URL,
        timeout=60,
        context=tls_context,
    ) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        allowed = {
            "ml-100k/u.data",
            "ml-100k/u.item",
            "ml-100k/u.genre",
            "ml-100k/README",
        }
        for member in allowed.intersection(archive.namelist()):
            archive.extract(member, root)
    if not (target / "u.data").exists():
        raise RuntimeError("MovieLens download completed without u.data")
    return target


def load_movielens(
    root: Path,
    positive_rating: float = 4.0,
    min_user_positives: int = 5,
) -> InteractionDataset:
    dataset_dir = _download_and_extract(root)
    ratings = pd.read_csv(
        dataset_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": "int64", "item_id": "int64", "rating": "float32", "timestamp": "int64"},
    )
    positives = ratings.loc[ratings["rating"] >= positive_rating].copy()
    counts = positives.groupby("user_id").size()
    eligible_users = counts[counts >= min_user_positives].index
    positives = positives[positives["user_id"].isin(eligible_users)]
    positives = positives.sort_values(["user_id", "timestamp", "item_id"])

    raw_user_ids = np.sort(positives["user_id"].unique())
    raw_item_ids = np.sort(positives["item_id"].unique())
    user_map = {raw_id: idx for idx, raw_id in enumerate(raw_user_ids)}
    item_map = {raw_id: idx for idx, raw_id in enumerate(raw_item_ids)}
    positives["user"] = positives["user_id"].map(user_map)
    positives["item"] = positives["item_id"].map(item_map)

    train_by_user: list[list[int]] = [[] for _ in raw_user_ids]
    test_item_by_user = np.empty(len(raw_user_ids), dtype=np.int64)
    train_pairs: list[tuple[int, int]] = []
    for user, group in positives.groupby("user", sort=True):
        items = group["item"].astype(int).tolist()
        train_items, test_item = items[:-1], items[-1]
        train_by_user[user] = train_items
        test_item_by_user[user] = test_item
        train_pairs.extend((user, item) for item in train_items)

    item_popularity = np.zeros(len(raw_item_ids), dtype=np.int64)
    for _, item in train_pairs:
        item_popularity[item] += 1

    metadata_columns = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        *GENRE_COLUMNS,
    ]
    items = pd.read_csv(
        dataset_dir / "u.item",
        sep="|",
        names=metadata_columns,
        encoding="latin-1",
        dtype={"item_id": "int64"},
    ).set_index("item_id")
    titles: list[str] = []
    genres: list[list[str]] = []
    for raw_item_id in raw_item_ids:
        if raw_item_id not in items.index:
            titles.append(f"Movie {raw_item_id}")
            genres.append([])
            continue
        row = items.loc[raw_item_id]
        titles.append(str(row["title"]))
        genres.append(
            [
                genre
                for genre in GENRE_COLUMNS
                if int(row[genre]) == 1 and genre != "unknown"
            ]
        )

    return InteractionDataset(
        train_by_user=train_by_user,
        test_item_by_user=test_item_by_user,
        raw_user_ids=raw_user_ids,
        raw_item_ids=raw_item_ids,
        item_titles=titles,
        item_genres=genres,
        item_popularity=item_popularity,
        train_pairs=np.asarray(train_pairs, dtype=np.int64),
    )
