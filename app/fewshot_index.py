"""Few-shot example retrieval index with cross-domain database exclusion."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info("Loading BAAI/bge-small-en-v1.5 on CPU...")
        _EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    return _EMBEDDING_MODEL


class FewShotIndex:
    """Vector index for training question-SQL pairs with strict same-database exclusion."""

    def __init__(self, chroma_dir: Optional[Path] = None) -> None:
        self.chroma_dir = Path(chroma_dir or settings.chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="fewshot_examples",
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def build_from_train(
        self,
        train_path: Optional[Path] = None,
        max_samples: Optional[int] = 1000,
    ) -> int:
        """Embed training questions into Chroma. Defaults to 1,000 diverse training examples."""
        target_path = train_path or Path("./data/spider/train_spider.json")
        if not target_path.exists():
            logger.warning("Train file not found at %s. Few-shot index will be empty.", target_path)
            return 0

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if max_samples and max_samples < len(data):
            # Take evenly spaced samples across databases
            step = max(1, len(data) // max_samples)
            data = data[::step][:max_samples]

        existing_count = self.collection.count()
        if existing_count >= len(data):
            logger.info("Few-shot index already populated with %d examples.", existing_count)
            return existing_count

        model = get_embedding_model()
        questions = [item["question"] for item in data]
        embeddings = model.encode(questions, batch_size=64, show_progress_bar=True).tolist()

        ids = [f"fs_{i}" for i in range(len(data))]
        metadatas = [
            {"db_id": item["db_id"], "sql": item["query"], "qid": str(i)}
            for i, item in enumerate(data)
        ]

        # Batch upsert
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=questions[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        logger.info("Indexed %d few-shot examples into %s", len(ids), self.chroma_dir)
        return len(ids)

    def retrieve(
        self,
        question: str,
        db_id: str,
        k: int = 3,
    ) -> List[Dict[str, str]]:
        """Retrieve top-k similar question-SQL pairs excluding identical db_id.

        Args:
            question: User query.
            db_id: Database ID of the target question (must be excluded).
            k: Number of examples to return.

        Returns:
            List of dicts with 'question', 'sql', and 'db_id'.
        """
        if self.collection.count() == 0:
            return []

        model = get_embedding_model()
        query_vec = model.encode(question).tolist()

        # Fetch extra candidates to filter out matching db_id
        n_fetch = min(self.collection.count(), max(k * 5, 25))
        res = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_fetch,
            include=["documents", "metadatas", "distances"],
        )

        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []

        candidates = []
        for doc, meta in zip(docs, metas):
            cand_db = meta.get("db_id", "")
            # Strict exclusion: NEVER return examples from the same db_id
            if cand_db.lower() == db_id.lower():
                continue

            candidates.append({
                "question": doc,
                "sql": meta.get("sql", ""),
                "db_id": cand_db,
            })
            if len(candidates) >= k:
                break

        return candidates


# Singleton index instance
_FEWSHOT_INDEX: Optional[FewShotIndex] = None


def get_fewshot_index(chroma_dir: Optional[Path] = None) -> FewShotIndex:
    global _FEWSHOT_INDEX
    if _FEWSHOT_INDEX is None:
        _FEWSHOT_INDEX = FewShotIndex(chroma_dir=chroma_dir)
    return _FEWSHOT_INDEX


def retrieve_fewshots(
    question: str,
    db_id: str,
    k: int = 3,
    chroma_dir: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Retrieve few-shot examples using the global index."""
    index = get_fewshot_index(chroma_dir=chroma_dir)
    return index.retrieve(question=question, db_id=db_id, k=k)

