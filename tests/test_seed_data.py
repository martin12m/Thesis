"""Tests for the SQLite seed script."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.seed_data import seed_from_files


def test_seed_from_files_loads_records_into_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "seeded.db"
    project_root = Path(__file__).resolve().parents[1]

    summary = seed_from_files(
        db_path,
        products_path=project_root / "data" / "products.json",
        policies_path=project_root / "data" / "policies.json",
    )

    assert summary == {
        "products": 2,
        "policies": 2,
        "product_policies": 3,
        "product_reviews": 0,
    }

    connection = sqlite3.connect(db_path)
    try:
        product_count = connection.execute(
            "SELECT COUNT(*) FROM products"
        ).fetchone()[0]
        policy_count = connection.execute(
            "SELECT COUNT(*) FROM policies"
        ).fetchone()[0]
        joined_rows = connection.execute(
            """
            SELECT p.sku, pol.code
            FROM product_policies AS pp
            JOIN products AS p ON p.id = pp.product_id
            JOIN policies AS pol ON pol.id = pp.policy_id
            ORDER BY p.sku, pol.code
            """
        ).fetchall()
    finally:
        connection.close()

    assert product_count == 2
    assert policy_count == 2
    assert joined_rows == [
        ("SKU-100", "PRIVACY"),
        ("SKU-100", "RETURNS"),
        ("SKU-200", "PRIVACY"),
    ]
