"""Proves the external reviews provider is a true optional side-car."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from scripts.seed_data import seed_from_files
from tools.catalog_tools import (
    _default_external_reviews_client,
    filter_products,
    get_product_by_id,
    get_product_reviews,
)

_LOCAL_ONLY_ENV = {"COMMERCE_PRODUCT_SOURCE": "local"}


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "catalog.db"
    project_root = Path(__file__).resolve().parents[1]
    seed_from_files(
        db_path,
        products_path=project_root / "data" / "products.json",
        policies_path=project_root / "data" / "policies.json",
    )
    return db_path


@pytest.fixture
def _external_reviews_import_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any import of ``integrations.product_reviews_api`` raise ImportError."""

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "integrations.product_reviews_api" or name.startswith("integrations.product_reviews_api"):
            msg = "simulated missing external reviews integration"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_default_external_reviews_client_returns_none_when_module_unavailable(
    _external_reviews_import_blocked: None,
) -> None:
    assert _default_external_reviews_client() is None


def test_get_product_reviews_degrades_gracefully_when_provider_unavailable(
    seeded_db: Path,
    _external_reviews_import_blocked: None,
) -> None:
    result = get_product_reviews(
        seeded_db,
        1,
        env={"CATALOG_REVIEWS_SOURCE": "external", "COMMERCE_PRODUCT_SOURCE": "local"},
    )

    assert result["source"] == "external_reviews_http"
    assert result["reviews"] == []
    assert result["score"] is None
    assert "error" in result


def test_core_catalog_reads_unaffected_when_provider_unavailable(
    seeded_db: Path,
    _external_reviews_import_blocked: None,
) -> None:
    product = get_product_by_id(seeded_db, 1, env=_LOCAL_ONLY_ENV)
    assert product["sku"] == "SKU-100"

    products = filter_products(seeded_db, max_price_cents=3000, env=_LOCAL_ONLY_ENV)
    assert products[0]["sku"] == "SKU-100"
