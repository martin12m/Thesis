"""Deterministic tests for the catalog tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.seed_data import seed_from_files
from tools.catalog_tools import (
    ProductNotFoundError,
    compare_products,
    filter_products,
    get_product_by_id,
)

_LOCAL_ONLY_ENV = {"COMMERCE_PRODUCT_SOURCE": "local"}


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """Create a temporary seeded SQLite database for catalog tests."""

    db_path = tmp_path / "catalog.db"
    project_root = Path(__file__).resolve().parents[1]
    seed_from_files(
        db_path,
        products_path=project_root / "data" / "products.json",
        policies_path=project_root / "data" / "policies.json",
    )
    return db_path


def test_get_product_by_id_returns_expected_product(seeded_db: Path) -> None:
    product = get_product_by_id(seeded_db, 1, env=_LOCAL_ONLY_ENV)

    assert product == {
        "id": 1,
        "sku": "SKU-100",
        "name": "Starter Plan",
        "description": "Entry-level subscription for small teams.",
        "price_cents": 1900,
        "is_active": True,
        "policy_codes": ["PRIVACY", "RETURNS"],
        "source": "local_fixture",
    }


def test_get_product_by_id_raises_for_missing_product(seeded_db: Path) -> None:
    with pytest.raises(ProductNotFoundError, match="Product with id 999 was not found."):
        get_product_by_id(seeded_db, 999, env=_LOCAL_ONLY_ENV)


def test_filter_products_by_policy_and_price(seeded_db: Path) -> None:
    products = filter_products(
        seeded_db,
        max_price_cents=3000,
        policy_code="RETURNS",
    )

    assert products == [
        {
            "id": 1,
            "sku": "SKU-100",
            "name": "Starter Plan",
            "description": "Entry-level subscription for small teams.",
            "price_cents": 1900,
            "is_active": True,
            "policy_codes": ["PRIVACY", "RETURNS"],
        }
    ]


def test_filter_products_local_only_matches_previous_shape(seeded_db: Path) -> None:
    """With COMMERCE_PRODUCT_SOURCE=local, output shape is unchanged."""

    products = filter_products(seeded_db, max_price_cents=3000, env=_LOCAL_ONLY_ENV)

    assert products == [
        {
            "id": 1,
            "sku": "SKU-100",
            "name": "Starter Plan",
            "description": "Entry-level subscription for small teams.",
            "price_cents": 1900,
            "is_active": True,
            "policy_codes": ["PRIVACY", "RETURNS"],
        }
    ]


def test_compare_products_returns_deterministic_difference(seeded_db: Path) -> None:
    comparison = compare_products(seeded_db, 1, 2)

    assert comparison == {
        "left_product": {
            "id": 1,
            "sku": "SKU-100",
            "name": "Starter Plan",
            "description": "Entry-level subscription for small teams.",
            "price_cents": 1900,
            "is_active": True,
            "policy_codes": ["PRIVACY", "RETURNS"],
        },
        "right_product": {
            "id": 2,
            "sku": "SKU-200",
            "name": "Growth Plan",
            "description": "Expanded subscription with analytics features.",
            "price_cents": 4900,
            "is_active": True,
            "policy_codes": ["PRIVACY"],
        },
        "price_difference_cents": -3000,
        "common_policy_codes": ["PRIVACY"],
        "left_only_policy_codes": ["RETURNS"],
        "right_only_policy_codes": [],
    }
