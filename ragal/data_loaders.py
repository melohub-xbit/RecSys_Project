import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Common output types
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
            items_data[iid] = {
                "title": item.title,
                "text": item.text,
                "metadata": meta,
            }
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

# MovieLens-1M Loaders
def load_movielens(data_dir: str = "data/movielens/ml-1m",
                   positive_threshold: int = 4) -> Dataset:
    data_dir = Path(data_dir)

    items = {}
    with open(data_dir / "movies.dat", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            mid, title, genres = parts[0], parts[1], parts[2]
            text = f"{title} | {genres.replace('|', ', ')}"
            items[mid] = Item(
                item_id=mid,
                title=title,
                text=text,
                metadata={"genres": genres.split("|")},
            )

    interactions = []
    with open(data_dir / "ratings.dat", encoding="latin-1") as f:
        for line in f:
            uid, mid, rating, ts = line.strip().split("::")
            reward = 1 if int(rating) >= positive_threshold else 0
            interactions.append((uid, mid, reward, float(ts)))

    logger.info("MovieLens-1M: %d items, %d interactions", len(items), len(interactions))
    ds = Dataset(name="MovieLens-1M", items=items, interactions=interactions)
    ds.build_user_sequences()
    return ds

# Semantic Scholar Loaders
def load_semantic_scholar(data_path: str = "data/semantic_scholar/recsys_complete_dataset.jsonl",
                          min_author_papers: int = 3,
                          neg_sample_ratio: int = 3,
                          seed: int = 42) -> Dataset:
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

    logger.info("S2: loaded %d papers with text", len(papers))

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
    logger.info("S2: %d authors with >= %d papers", len(author_papers), min_author_papers)

    all_pids = list(papers.keys())
    interactions = []

    for aid, plist in author_papers.items():
        authored_set = {pid for pid, _ in plist}

        for pid, year in plist:
            ts = float(year)
            interactions.append((aid, pid, 1, ts))

            neg_count = 0
            attempts = 0
            while neg_count < neg_sample_ratio and attempts < neg_sample_ratio * 10:
                neg_pid = rng.choice(all_pids)
                if neg_pid not in authored_set:
                    interactions.append((aid, neg_pid, 0, ts + 0.5))
                    neg_count += 1
                attempts += 1

    ds = Dataset(name="SemanticScholar", items=papers, interactions=interactions)
    ds.build_user_sequences()
    logger.info("S2: %s", ds.summary())
    return ds

# Citation Network Loaders
def _reconstruct_abstract(indexed_abstract: dict) -> str:
    if not indexed_abstract:
        return ""
    length = indexed_abstract.get("IndexLength", 0)
    inv = indexed_abstract.get("InvertedIndex", {})
    if not inv or length == 0:
        return ""
    words = [""] * length
    for word, positions in inv.items():
        for pos in positions:
            if pos < length:
                words[pos] = word
    return " ".join(words)

def load_citation_network(data_path: str = "data/citation_network/dblp.v12.json",
                          fos_filter: list[str] | None = None,
                          max_papers: int = 100_000,
                          min_refs: int = 3,
                          min_author_papers: int = 3,
                          neg_sample_ratio: int = 3,
                          seed: int = 42) -> Dataset:
    rng = random.Random(seed)
    data_path = Path(data_path)

    if fos_filter:
        fos_filter_lower = {f.lower() for f in fos_filter}
    else:
        fos_filter_lower = None

    papers = {}
    logger.info("CitationNet: streaming %s...", data_path)

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line in ("", "[", "]"):
                continue
            if line.startswith(","):
                line = line[1:]
            if line.endswith(","):
                line = line[:-1]
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not rec.get("title"):
                continue
            refs = rec.get("references", [])
            if len(refs) < min_refs:
                continue
            abstract = _reconstruct_abstract(rec.get("indexed_abstract"))
            if len(abstract) < 20:
                continue

            if fos_filter_lower:
                fos_names = {f["name"].lower() for f in rec.get("fos", []) if f.get("name")}
                if not fos_names & fos_filter_lower:
                    continue

            pid = str(rec["id"])
            title = rec["title"]
            text = f"{title}. {abstract}"

            authors = rec.get("authors", [])
            author_names = [a.get("name", "") for a in authors]
            author_ids = [str(a["id"]) for a in authors if a.get("id")]

            venue = rec.get("venue", {})
            fos_list = [f["name"] for f in rec.get("fos", []) if f.get("name")]

            papers[pid] = Item(
                item_id=pid,
                title=title,
                text=text,
                metadata={
                    "year": rec.get("year"),
                    "authors": author_names,
                    "author_ids": author_ids,
                    "references": [str(r) for r in refs],
                    "venue": venue.get("raw", ""),
                    "fos": fos_list,
                    "n_citation": rec.get("n_citation", 0),
                },
            )

            if len(papers) >= max_papers:
                break

    logger.info("CitationNet: loaded %d papers after filtering", len(papers))

    author_papers = defaultdict(list)
    for pid, item in papers.items():
        year = item.metadata.get("year") or 2000
        for aid in item.metadata.get("author_ids", []):
            author_papers[aid].append((pid, year))

    author_papers = {
        aid: sorted(plist, key=lambda x: x[1])
        for aid, plist in author_papers.items()
        if len(plist) >= min_author_papers
    }
    logger.info("CitationNet: %d authors with >= %d papers", len(author_papers), min_author_papers)

    all_pids = list(papers.keys())
    interactions = []

    for aid, plist in author_papers.items():
        authored_set = {pid for pid, _ in plist}
        for pid, year in plist:
            ts = float(year)
            interactions.append((aid, pid, 1, ts))

            neg_count = 0
            attempts = 0
            while neg_count < neg_sample_ratio and attempts < neg_sample_ratio * 10:
                neg_pid = rng.choice(all_pids)
                if neg_pid not in authored_set:
                    interactions.append((aid, neg_pid, 0, ts + 0.5))
                    neg_count += 1
                attempts += 1

    ds = Dataset(name="CitationNetwork", items=papers, interactions=interactions)
    ds.build_user_sequences()
    logger.info("CitationNet: %s", ds.summary())
    return ds

# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 60)
    print("Loading MovieLens-1M...")
    ml = load_movielens()
    print(ml.summary())
    sample_user = list(ml.user_sequences.keys())[0]
    print(f"  Sample user {sample_user}: {len(ml.user_sequences[sample_user])} interactions")
    print(f"  First 3: {ml.user_sequences[sample_user][:3]}")
    print()

    print("=" * 60)
    print("Loading Semantic Scholar...")
    s2 = load_semantic_scholar()
    print(s2.summary())
    if s2.user_sequences:
        sample_user = list(s2.user_sequences.keys())[0]
        print(f"  Sample author {sample_user}: {len(s2.user_sequences[sample_user])} interactions")
    print()

    print("=" * 60)
    print("Loading Citation Network...")
    cn = load_citation_network(
        max_papers=50_000,
        fos_filter=["Computer science"],
    )
    print(cn.summary())
    if cn.user_sequences:
        sample_user = list(cn.user_sequences.keys())[0]
        print(f"  Sample author {sample_user}: {len(cn.user_sequences[sample_user])} interactions")
