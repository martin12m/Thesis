"""Utilities for initializing and seeding the local SQLite database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a list of JSON objects from disk."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        msg = f"Expected a list of records in {file_path}"
        raise ValueError(msg)

    if not all(isinstance(item, dict) for item in payload):
        msg = f"Expected object records in {file_path}"
        raise ValueError(msg)

    return payload


def connect_database(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled."""

    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the minimal products and policies schema."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            description TEXT,
            price_cents INTEGER NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            category TEXT,
            weight_g INTEGER,
            material TEXT,
            color TEXT,
            brand TEXT,
            rating_stars REAL,
            rating_count INTEGER,
            rating_grade TEXT
        );

        CREATE TABLE IF NOT EXISTS policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            body TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS product_policies (
            product_id INTEGER NOT NULL,
            policy_id INTEGER NOT NULL,
            PRIMARY KEY (product_id, policy_id),
            FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
            FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS product_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            external_review_id TEXT,
            stars REAL,
            title TEXT,
            text TEXT,
            network TEXT,
            author TEXT,
            created_at TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
        );
        """
    )
    _ensure_product_columns(
        connection,
        {
            "category": "TEXT",
            "weight_g": "INTEGER",
            "material": "TEXT",
            "color": "TEXT",
            "brand": "TEXT",
            "rating_stars": "REAL",
            "rating_count": "INTEGER",
            "rating_grade": "TEXT",
        },
    )


def _ensure_product_columns(
    connection: sqlite3.Connection,
    columns: dict[str, str],
) -> None:
    """Add optional product columns when older databases are reused."""

    existing_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(products)").fetchall()
    }
    for column_name, column_type in columns.items():
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE products ADD COLUMN {column_name} {column_type}"
            )


def seed_database(
    connection: sqlite3.Connection,
    *,
    products: list[dict[str, Any]],
    policies: list[dict[str, Any]],
) -> dict[str, int]:
    """Replace database contents with the provided seed data."""

    with connection:
        initialize_schema(connection)
        connection.execute("DELETE FROM product_reviews")
        connection.execute("DELETE FROM product_policies")
        connection.execute("DELETE FROM products")
        connection.execute("DELETE FROM policies")

        policy_rows = [
            (item["code"], item["title"], item["body"])
            for item in policies
        ]
        connection.executemany(
            """
            INSERT INTO policies (code, title, body)
            VALUES (?, ?, ?)
            """,
            policy_rows,
        )

        product_rows = [
            (
                item["sku"],
                item["name"],
                item.get("description"),
                item["price_cents"],
                int(item.get("is_active", True)),
                item.get("category"),
                item.get("weight_g"),
                item.get("material"),
                item.get("color"),
                item.get("brand"),
                item.get("rating_stars"),
                item.get("rating_count"),
                item.get("rating_grade"),
            )
            for item in products
        ]
        connection.executemany(
            """
            INSERT INTO products (
                sku,
                name,
                description,
                price_cents,
                is_active,
                category,
                weight_g,
                material,
                color,
                brand,
                rating_stars,
                rating_count,
                rating_grade
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            product_rows,
        )

        policy_id_by_code = {
            code: policy_id
            for policy_id, code in connection.execute(
                "SELECT id, code FROM policies"
            ).fetchall()
        }
        product_id_by_sku = {
            sku: product_id
            for product_id, sku in connection.execute(
                "SELECT id, sku FROM products"
            ).fetchall()
        }

        association_rows: list[tuple[int, int]] = []
        for item in products:
            product_id = product_id_by_sku[item["sku"]]
            for code in item.get("policy_codes", []):
                if code not in policy_id_by_code:
                    msg = f"Unknown policy code: {code}"
                    raise ValueError(msg)
                association_rows.append((product_id, policy_id_by_code[code]))

        connection.executemany(
            """
            INSERT INTO product_policies (product_id, policy_id)
            VALUES (?, ?)
            """,
            association_rows,
        )

        review_rows: list[tuple[Any, ...]] = []
        for item in products:
            product_id = product_id_by_sku[item["sku"]]
            for review in item.get("reviews", []):
                review_rows.append(
                    (
                        product_id,
                        review.get("external_review_id") or review.get("id"),
                        review.get("stars"),
                        review.get("title"),
                        review.get("text"),
                        review.get("network") or review.get("nw"),
                        review.get("author"),
                        review.get("created_at") or review.get("createdAt"),
                    )
                )

        connection.executemany(
            """
            INSERT INTO product_reviews (
                product_id,
                external_review_id,
                stars,
                title,
                text,
                network,
                author,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            review_rows,
        )

    return {
        "products": len(product_rows),
        "policies": len(policy_rows),
        "product_policies": len(association_rows),
        "product_reviews": len(review_rows),
    }
