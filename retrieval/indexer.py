"""Utilities for building a hybrid (sparse + dense) retrieval index in SQLite."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any, Callable

from data.loaders import connect_database
from retrieval.sparse import embed_texts, pack_vector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexedDocument:
    """Structured representation of a retrieval document."""

    document_id: str
    source_type: str
    source_record_id: int
    title: str
    content: str


def initialize_index(connection: sqlite3.Connection) -> None:
    """Create the tables required for hybrid (sparse + dense) retrieval."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS retrieval_documents (
            document_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_record_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_fts
        USING fts5(
            document_id UNINDEXED,
            title,
            content
        );

        CREATE TABLE IF NOT EXISTS retrieval_embeddings (
            document_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL
        );
        """
    )


def rebuild_index(
    db_path: str | Path,
    *,
    embedder: Callable[[list[str]], list[Any]] | None = None,
) -> dict[str, int]:
    """Rebuild the hybrid (sparse + dense) retrieval index from product/policy tables."""

    connection = connect_database(db_path)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            initialize_index(connection)
            connection.execute("DELETE FROM retrieval_documents")
            connection.execute("DELETE FROM retrieval_fts")
            connection.execute("DELETE FROM retrieval_embeddings")

            documents = _collect_documents(connection)
            connection.executemany(
                """
                INSERT INTO retrieval_documents (
                    document_id, source_type, source_record_id, title, content
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [astuple(document) for document in documents],
            )
            connection.executemany(
                "INSERT INTO retrieval_fts (document_id, title, content) VALUES (?, ?, ?)",
                [(document.document_id, document.title, document.content) for document in documents],
            )
            _populate_embeddings(connection, documents, embedder)
        return {"documents_indexed": len(documents)}
    finally:
        connection.close()


def _populate_embeddings(
    connection: sqlite3.Connection,
    documents: list[IndexedDocument],
    embedder: Callable[[list[str]], list[Any]] | None,
) -> None:
    """Compute and store one dense embedding per document, for hybrid retrieval."""

    if not documents:
        return

    try:
        texts = [f"{document.title}. {document.content}" for document in documents]
        vectors = embed_texts(texts, embedder)
    except Exception as exc:  # noqa: BLE001 - embeddings must never block indexing
        logger.warning("Skipping dense embeddings for this index rebuild: %s", exc)
        return

    connection.executemany(
        "INSERT INTO retrieval_embeddings (document_id, vector) VALUES (?, ?)",
        [
            (document.document_id, pack_vector(vector))
            for document, vector in zip(documents, vectors, strict=True)
        ],
    )


def _collect_documents(connection: sqlite3.Connection) -> list[IndexedDocument]:
    """Collect retrieval documents from the products and policies tables."""

    product_rows = connection.execute(
        """
        SELECT id, sku, name, description, price_cents, category, weight_g,
               material, color, brand, rating_stars, rating_count, rating_grade
        FROM products
        ORDER BY id
        """
    ).fetchall()
    policy_rows = connection.execute("SELECT id, title, body FROM policies ORDER BY id").fetchall()

    documents: list[IndexedDocument] = []
    for row in product_rows:
        content_parts = [
            f"SKU {row['sku']}.",
            f"Category {row['category'] or 'unknown'}.",
            row["description"] or "",
            f"Price {row['price_cents']} cents.",
            f"Weight {row['weight_g'] or 'unknown'} grams.",
            f"Material {row['material'] or 'unknown'}.",
            f"Color {row['color'] or 'unknown'}.",
        ]
        if row["brand"]:
            content_parts.append(f"Brand {row['brand']}.")
        rating_stars, rating_count = row["rating_stars"], row["rating_count"]
        if rating_stars is not None or rating_count is not None:
            stars_text = f"{rating_stars} stars" if rating_stars is not None else "an unrated score"
            count_text = f" from {rating_count} ratings" if rating_count is not None else ""
            grade_text = f" ({row['rating_grade']})" if row["rating_grade"] else ""
            content_parts.append(f"Rated {stars_text}{count_text}{grade_text}.")

        reviews = connection.execute(
            "SELECT title, text FROM product_reviews WHERE product_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
        for review in reviews:
            if review["title"] or review["text"]:
                content_parts.append(f"Review: {review['title'] or ''} {review['text'] or ''}".strip())

        documents.append(
            IndexedDocument(
                document_id=f"product:{row['id']}",
                source_type="product",
                source_record_id=row["id"],
                title=row["name"],
                content=" ".join(part for part in content_parts if part).strip(),
            )
        )

    documents.extend(
        IndexedDocument(
            document_id=f"policy:{row['id']}",
            source_type="policy",
            source_record_id=row["id"],
            title=row["title"],
            content=row["body"],
        )
        for row in policy_rows
    )
    return documents
