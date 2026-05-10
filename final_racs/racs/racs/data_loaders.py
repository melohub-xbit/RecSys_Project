"""
Dataset loaders for RACS.

Supports MovieLens ml-1m (DAT format), ml-20m / ml-25m (CSV format), and
Semantic Scholar JSONL dumps. Each loader produces a common `Dataset` shape
(items + interactions + per-user sequences) consumed by the offline phase.
"""

import csv
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common output types (mirror of data_loaders.Dataset / Item)
# ---------------------------------------------------------------------------

@dataclass
class Item:
    item_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    items: dict[str, Item]
    interactions: list[tuple[str, str, int, float]]
    user_sequences: dict[str, list[tuple[str, int]]] = field(default_factory=dict)

    def build_user_sequences(self):
        by_user = defaultdict(list)
        for uid, iid, reward, ts in self.interactions:
            by_user[uid].append((iid, reward, ts))
        self.user_sequences = {}
        for uid, seq in by_user.items():
            seq.sort(key=lambda x: x[2])
            self.user_sequences[uid] = [(iid, r) for iid, r, _ in seq]
        return self

    def summary(self) -> str:
        n_users = len(set(uid for uid, *_ in self.interactions))
        n_pos = sum(1 for _, _, r, _ in self.interactions if r == 1)
        n_neg = sum(1 for _, _, r, _ in self.interactions if r == 0)
        return (
            f"Dataset({self.name}): {len(self.items)} items, "
            f"{n_users} users, {len(self.interactions)} interactions "
            f"(pos={n_pos}, neg={n_neg})"
        )

    def save(self, path: str):
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)

        items_data = {}
        for iid, item in self.items.items():
            meta = {k: v for k, v in item.metadata.items()
                    if k != "precomputed_embedding"}
            items_data[iid] = {"title": item.title, "text": item.text, "metadata": meta}
        with open(d / "items.json", "w", encoding="utf-8") as f:
            json.dump(items_data, f, ensure_ascii=False)

        uids, iids, rewards, timestamps = [], [], [], []
        for uid, iid, reward, ts in self.interactions:
            uids.append(uid)
            iids.append(iid)
            rewards.append(reward)
            timestamps.append(ts)
        np.savez_compressed(
            d / "interactions.npz",
            uids=np.array(uids, dtype=object),
            iids=np.array(iids, dtype=object),
            rewards=np.array(rewards, dtype=np.int8),
            timestamps=np.array(timestamps, dtype=np.float64),
        )

        with open(d / "manifest.json", "w") as f:
            json.dump({"name": self.name, "n_items": len(self.items),
                        "n_interactions": len(self.interactions)}, f)
        logger.info("Dataset saved to %s", d)

    @classmethod
    def load(cls, path: str) -> "Dataset":
        d = Path(path)
        with open(d / "manifest.json") as f:
            manifest = json.load(f)
        with open(d / "items.json", encoding="utf-8") as f:
            items_data = json.load(f)
        items = {
            iid: Item(item_id=iid, title=v["title"], text=v["text"],
                       metadata=v.get("metadata", {}))
            for iid, v in items_data.items()
        }
        data = np.load(d / "interactions.npz", allow_pickle=True)
        interactions = list(zip(
            data["uids"].tolist(),
            data["iids"].tolist(),
            data["rewards"].tolist(),
            data["timestamps"].tolist(),
        ))
        ds = cls(name=manifest["name"], items=items, interactions=interactions)
        ds.build_user_sequences()
        logger.info("Dataset loaded from %s: %s", d, ds.summary())
        return ds


# ---------------------------------------------------------------------------
# 1. MovieLens (ml-1m DAT + ml-20m / ml-25m CSV)
# ---------------------------------------------------------------------------

def load_movielens(data_dir: str,
                   positive_threshold: float = 4.0,
                   name: str | None = None) -> Dataset:
    """
    Load a MovieLens dataset (ml-1m, ml-20m, or ml-25m).

    ml-1m uses "::" delimited .dat files.
    ml-20m / ml-25m use CSV files with headers.

    Ratings >= positive_threshold become reward=1, else reward=0.
    """
    data_dir = Path(data_dir)

    if (data_dir / "movies.dat").exists():
        movies_file = data_dir / "movies.dat"
        ratings_file = data_dir / "ratings.dat"
        fmt = "dat"
    elif (data_dir / "movies.csv").exists():
        movies_file = data_dir / "movies.csv"
        ratings_file = data_dir / "ratings.csv"
        fmt = "csv"
    else:
        raise FileNotFoundError(f"No movies.dat or movies.csv found in {data_dir}")

    items = {}
    if fmt == "dat":
        with open(movies_file, encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                mid, title, genres = parts[0], parts[1], parts[2]
                text = f"{title} | {genres.replace('|', ', ')}"
                items[mid] = Item(item_id=mid, title=title, text=text,
                                  metadata={"genres": genres.split("|")})
    else:
        with open(movies_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid, title, genres = row["movieId"], row["title"], row["genres"]
                text = f"{title} | {genres.replace('|', ', ')}"
                items[mid] = Item(item_id=mid, title=title, text=text,
                                  metadata={"genres": genres.split("|")})

    interactions = []
    if fmt == "dat":
        with open(ratings_file, encoding="latin-1") as f:
            for line in f:
                uid, mid, rating, ts = line.strip().split("::")
                reward = 1 if float(rating) >= positive_threshold else 0
                interactions.append((uid, mid, reward, float(ts)))
    else:
        with open(ratings_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row["userId"]
                mid = row["movieId"]
                rating = float(row["rating"])
                ts = float(row["timestamp"])
                reward = 1 if rating >= positive_threshold else 0
                interactions.append((uid, mid, reward, ts))

    if name is None:
        suffix = data_dir.name.replace("ml-", "").upper()
        name = f"MovieLens-{suffix}" if suffix else "MovieLens"

    logger.info("%s: %d items, %d interactions", name, len(items), len(interactions))
    ds = Dataset(name=name, items=items, interactions=interactions)
    ds.build_user_sequences()
    return ds


# ---------------------------------------------------------------------------
# 2. Semantic Scholar (JSONL, small or bigger)
# ---------------------------------------------------------------------------

def load_semantic_scholar(data_path: str,
                          min_author_papers: int = 3,
                          neg_sample_ratio: int = 3,
                          seed: int = 42,
                          name: str = "SemanticScholar") -> Dataset:
    """
    Load a Semantic Scholar JSONL dump. Author = user; authored papers = positive,
    random non-authored papers = negatives (at neg_sample_ratio:1).
    """
    rng = random.Random(seed)
    data_path = Path(data_path)

    papers = {}
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["paperId"]
            abstract = rec.get("abstract") or ""
            title = rec.get("title") or ""
            text = f"{title}. {abstract}".strip()
            if not text or text == ".":
                continue
            papers[pid] = Item(
                item_id=pid,
                title=title,
                text=text,
                metadata={
                    "year": rec.get("year"),
                    "authors": [a["name"] for a in rec.get("authors", [])],
                    "author_ids": [a["authorId"] for a in rec.get("authors", [])
                                   if a.get("authorId")],
                },
            )
            emb = rec.get("embedding")
            if emb and emb.get("vector"):
                papers[pid].metadata["precomputed_embedding"] = emb["vector"]

    logger.info("S2 (%s): loaded %d papers with text", name, len(papers))

    author_papers = defaultdict(list)
    for pid, item in papers.items():
        year = item.metadata.get("year") or 2020
        for aid in item.metadata.get("author_ids", []):
            author_papers[aid].append((pid, year))

    author_papers = {
        aid: sorted(plist, key=lambda x: x[1])
        for aid, plist in author_papers.items()
        if len(plist) >= min_author_papers
    }
    logger.info("S2 (%s): %d authors with >= %d papers",
                name, len(author_papers), min_author_papers)

    all_pids = list(papers.keys())
    interactions = []
    for aid, plist in author_papers.items():
        authored_set = {pid for pid, _ in plist}
        for pid, year in plist:
            ts = float(year)
            interactions.append((aid, pid, 1, ts))
            neg_count, attempts = 0, 0
            while neg_count < neg_sample_ratio and attempts < neg_sample_ratio * 10:
                neg_pid = rng.choice(all_pids)
                if neg_pid not in authored_set:
                    interactions.append((aid, neg_pid, 0, ts + 0.5))
                    neg_count += 1
                attempts += 1

    ds = Dataset(name=name, items=papers, interactions=interactions)
    ds.build_user_sequences()
    logger.info("S2 (%s): %s", name, ds.summary())
    return ds


