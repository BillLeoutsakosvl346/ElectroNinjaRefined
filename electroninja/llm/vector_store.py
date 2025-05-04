# electroninja/llm/vector_store.py
"""FAISS‑backed vector store using OpenAI embeddings (SDK ≥ 1.75)."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from openai import OpenAI

from electroninja.config.settings import Config

logger = logging.getLogger("electroninja")


class VectorStore:
    """Simple semantic search over example circuits."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)

        self.embedding_model = "text-embedding-3-small"
        self.vector_size = 1536

        # FAISS index + metadata
        self.faiss = None
        self.index = None
        self.metadata_list: List[Dict[str, Any]] = []

        try:
            import faiss  # type: ignore
            self.faiss = faiss
            self.index = faiss.IndexFlatL2(self.vector_size)
            logger.info("FAISS index initialised")
        except ImportError:
            logger.error("FAISS not installed – vector search disabled.")

    # ────────────────────────────────────────────────────────────────
    # embeddings
    # ────────────────────────────────────────────────────────────────
    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Return a float32 embedding, or None on failure."""
        if not text:
            return None
        try:
            resp = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text],
                encoding_format="float",
            )
            vec = resp.data[0].embedding
            return np.asarray(vec, dtype=np.float32)
        except Exception as exc:        # pragma: no cover
            logger.error("Embedding error: %s", exc, exc_info=True)
            return None

    # ────────────────────────────────────────────────────────────────
    # CRUD
    # ────────────────────────────────────────────────────────────────
    def add_document(self, asc_code: str, metadata: Dict[str, Any]) -> bool:
        vec = self._embed(asc_code)
        if vec is None or self.faiss is None:
            return False
        self.index.add(vec.reshape(1, -1))
        self.metadata_list.append({"asc_code": asc_code, "metadata": metadata})
        return True

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return *sorted* nearest neighbours with an explicit score field."""
        if self.faiss is None or self.index is None or self.index.ntotal == 0:
            return []

        q_vec = self._embed(query)
        if q_vec is None:
            return []

        dists, idxs = self.index.search(q_vec.reshape(1, -1), top_k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx == -1:
                continue
            payload = self.metadata_list[idx]
            results.append({**payload, "score": float(dist)})

        # ✨ ensure ascending distance (most similar first)
        results.sort(key=lambda r: r["score"])
        return results[:top_k]


    # ────────────────────────────────────────────────────────────────
    # persistence
    # ────────────────────────────────────────────────────────────────
    def save(self) -> bool:
        if self.faiss is None or self.index is None:
            return False
        try:
            os.makedirs(Path(self.config.VECTOR_DB_INDEX).parent, exist_ok=True)
            self.faiss.write_index(self.index, self.config.VECTOR_DB_INDEX)
            with open(self.config.VECTOR_DB_METADATA, "wb") as fh:
                pickle.dump(self.metadata_list, fh)
            return True
        except Exception as exc:        # pragma: no cover
            logger.error("VectorStore save error: %s", exc, exc_info=True)
            return False

    def load(self) -> bool:
        try:
            import faiss  # type: ignore
            self.faiss = faiss
            if Path(self.config.VECTOR_DB_INDEX).is_file():
                self.index = faiss.read_index(self.config.VECTOR_DB_INDEX)
                with open(self.config.VECTOR_DB_METADATA, "rb") as fh:
                    self.metadata_list = pickle.load(fh)
            else:
                self.index = faiss.IndexFlatL2(self.vector_size)
            return True
        except Exception as exc:        # pragma: no cover
            logger.error("VectorStore load error: %s", exc, exc_info=True)
            return False
