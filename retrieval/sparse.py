"""Sparse (BM25) and hybrid (BM25 + dense embedding) retrieval over SQLite."""

from __future__ import annotations

import logging
import re
import sqlite3
from array import array
from pathlib import Path
from typing import Any, Callable, Protocol

from data.loaders import connect_database

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "better", "but", "by",
        "can", "could", "did", "do", "does", "for", "from", "had", "has",
        "have", "how", "i", "if", "in", "is", "it", "of", "on", "or",
        "our", "should", "than", "that", "the", "their", "there", "this",
        "to", "was", "we", "were", "what", "when", "where", "which",
        "who", "why", "will", "with", "would", "you", "your",
    }
)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_embedding_model: Any = None


class RetrievalBackend(Protocol):
    """Protocol for retrieval functions used by the variants."""

    def __call__(
        self,
        db_path: str | Path,
        query: str,
        *,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return top-k evidence items for the query."""


def search_index(
    db_path: str | Path,
    query: str,
    *,
    top_k: int = 3,
) -> list[dict[str, str | int | float]]:
    """Return the top-k BM25-only sparse retrieval matches for a query."""

    normalized_query = _normalize_query(query)
    if not normalized_query:
        return []

    connection = connect_database(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT
                d.document_id,
                d.source_type,
                d.source_record_id,
                d.title,
                d.content,
                bm25(retrieval_fts) AS score
            FROM retrieval_fts
            JOIN retrieval_documents AS d
                ON d.document_id = retrieval_fts.document_id
            WHERE retrieval_fts MATCH ?
            ORDER BY score ASC, d.document_id ASC
            LIMIT ?
            """,
            (normalized_query, top_k),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def search_hybrid(
    db_path: str | Path,
    query: str,
    *,
    top_k: int = 3,
    alpha: float = 0.5,
    embedder: Callable[[list[str]], list[Any]] | None = None,
) -> list[dict[str, str | int | float]]:
    """Return top-k evidence blending BM25 (sparse) and embedding (dense) similarity."""

    bm25_scores = _bm25_scores(db_path, query)
    cosine_scores = _cosine_scores(db_path, query, embedder)

    bm25_normalized = _min_max_normalize(bm25_scores, invert=True)
    cosine_normalized = _min_max_normalize(cosine_scores)

    candidate_ids = set(bm25_normalized) | set(cosine_normalized)
    if not candidate_ids:
        return []

    blended = {
        document_id: alpha * cosine_normalized.get(document_id, 0.0)
        + (1 - alpha) * bm25_normalized.get(document_id, 0.0)
        for document_id in candidate_ids
    }
    top_ids = sorted(blended, key=lambda doc_id: (-blended[doc_id], doc_id))[:top_k]
    return _fetch_documents(db_path, top_ids, blended)


def embed_texts(
    texts: list[str],
    embedder: Callable[[list[str]], list[Any]] | None = None,
) -> list[list[float]]:
    """Embed a batch of texts into dense vectors for hybrid retrieval."""

    if embedder is not None:
        return [[float(value) for value in vector] for vector in embedder(texts)]

    global _embedding_model  # noqa: PLW0603
    if _embedding_model is None:
        from fastembed import TextEmbedding  # noqa: PLC0415

        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return [vector.tolist() for vector in _embedding_model.embed(texts)]


def pack_vector(vector: list[float]) -> bytes:
    """Serialize a float vector to bytes for SQLite BLOB storage."""

    return array("f", vector).tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    """Deserialize a float vector previously stored by `pack_vector`."""

    values = array("f")
    values.frombytes(blob)
    return list(values)


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two equal-length vectors (cosine similarity, since the embedding model's outputs are already L2-normalized)."""

    return sum(x * y for x, y in zip(a, b, strict=True))


def _min_max_normalize(scores: dict[str, float], *, invert: bool = False) -> dict[str, float]:
    """Scale a score dict to [0, 1]."""

    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return dict.fromkeys(scores, 1.0)

    def _scale(value: float) -> float:
        normalized = (value - lo) / (hi - lo)
        return 1.0 - normalized if invert else normalized

    return {key: _scale(value) for key, value in scores.items()}


def _bm25_scores(db_path: str | Path, query: str) -> dict[str, float]:
    """Return raw BM25 scores (lower is better) for every FTS-matching document."""

    normalized_query = _normalize_query(query)
    if not normalized_query:
        return {}

    connection = connect_database(db_path)
    try:
        rows = connection.execute(
            """
            SELECT document_id, bm25(retrieval_fts) AS score
            FROM retrieval_fts
            WHERE retrieval_fts MATCH ?
            """,
            (normalized_query,),
        ).fetchall()
        return {str(document_id): float(score) for document_id, score in rows}
    finally:
        connection.close()


def _cosine_scores(
    db_path: str | Path,
    query: str,
    embedder: Callable[[list[str]], list[Any]] | None,
) -> dict[str, float]:
    """Return cosine similarity between the query and every embedded document."""

    connection = connect_database(db_path)
    try:
        rows = connection.execute("SELECT document_id, vector FROM retrieval_embeddings").fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        connection.close()

    if not rows:
        return {}

    try:
        query_vector = embed_texts([query], embedder)[0]
    except Exception as exc:  # noqa: BLE001 - embedding must never block retrieval
        logger.warning("Dense embedding unavailable, falling back to BM25-only ranking: %s", exc)
        return {}

    return {str(document_id): _dot(query_vector, unpack_vector(blob)) for document_id, blob in rows}


def _fetch_documents(
    db_path: str | Path,
    document_ids: list[str],
    scores: dict[str, float],
) -> list[dict[str, str | int | float]]:
    """Fetch full document rows for `document_ids`, preserving their input order."""

    if not document_ids:
        return []

    connection = connect_database(db_path)
    connection.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in document_ids)
        rows = connection.execute(
            f"""
            SELECT document_id, source_type, source_record_id, title, content
            FROM retrieval_documents
            WHERE document_id IN ({placeholders})
            """,
            document_ids,
        ).fetchall()
        by_id = {
            row["document_id"]: {**dict(row), "score": scores[row["document_id"]]}
            for row in rows
        }
        return [by_id[document_id] for document_id in document_ids if document_id in by_id]
    finally:
        connection.close()


def _normalize_query(query: str) -> str:
    """Normalize free text into an FTS5 MATCH expression."""

    terms = re.findall(r"[A-Za-z0-9]+", query.lower())
    meaningful_terms = [term for term in terms if term not in _STOPWORDS]
    selected_terms = meaningful_terms or terms
    return " OR ".join(selected_terms)
