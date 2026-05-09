"""FAISS evidence store backed by SentenceTransformer embeddings."""

from __future__ import annotations
import json
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEXES_DIR = os.path.join(os.path.dirname(__file__), "indexes")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_encoder: SentenceTransformer | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(EMBED_MODEL)
    return _encoder


class EvidenceStore:
    def __init__(self, condition: str):
        self.condition = condition
        os.makedirs(INDEXES_DIR, exist_ok=True)
        self._index_path = os.path.join(INDEXES_DIR, f"{condition}.faiss")
        self._docs_path = os.path.join(INDEXES_DIR, f"{condition}_docs.json")
        self._index: faiss.IndexFlatL2 | None = None
        self._docs: List[str] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._index_path) and os.path.exists(self._docs_path):
            self._index = faiss.read_index(self._index_path)
            with open(self._docs_path) as f:
                self._docs = json.load(f)

    def build(self, docs: List[dict]) -> None:
        texts = [f"{d['title']}. {d['abstract']}" for d in docs if d.get("abstract")]
        if not texts:
            return
        encoder = _get_encoder()
        embeddings = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings.astype(np.float32))
        self._docs = texts
        faiss.write_index(self._index, self._index_path)
        with open(self._docs_path, "w") as f:
            json.dump(self._docs, f)

    def query(self, text: str, k: int = 3) -> List[str]:
        if self._index is None or not self._docs:
            return []
        encoder = _get_encoder()
        vec = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        k = min(k, len(self._docs))
        _, indices = self._index.search(vec, k)
        return [self._docs[i] for i in indices[0] if i < len(self._docs)]

    def is_built(self) -> bool:
        return self._index is not None and len(self._docs) > 0
