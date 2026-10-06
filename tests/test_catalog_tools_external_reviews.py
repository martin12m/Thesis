"""Tests for catalog tools' optional live-external-reviews wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from integrations.product_reviews_api import ExternalProduct, ExternalReviewsProductError
from scripts.seed_data import seed_from_files
from tools.catalog_tools import filter_products, get_product_reviews

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


class _StubExternalReviewsClient:
    def __init__(self, product: ExternalProduct | Exception) -> None:
        self._product = product
        self.calls: list[dict[str, Any]] = []

    def get_product(self, **kwargs: Any) -> ExternalProduct:
        self.calls.append(kwargs)
        if isinstance(self._product, Exception):
            raise self._product
        return self._product


def _sample_external_product(product_id: str) -> ExternalProduct:
    return ExternalProduct(
        product_id=product_id,
        infos={"title": "Starter Plan", "brand": "Acme"},
        score={"stars": 4.5, "ratingsTotal": 120, "grade": "Sehr Gut"},
        reviews=[
            {
                "id": "rev-1",
                "stars": 5,
                "text": "Great product",
                "nw": "Amazon.de",
            }
        ],
        meta={},
    )


def test_get_product_reviews_defaults_to_local_fixture(seeded_db: Path) -> None:
    result = get_product_reviews(seeded_db, 1, env=_LOCAL_ONLY_ENV)

    assert result["source"] == "local_fixture"
    assert result["reviews"] == []
    assert result["score"] is None


def test_get_product_reviews_uses_live_client_when_configured(seeded_db: Path) -> None:
    stub = _StubExternalReviewsClient(_sample_external_product("SKU-100"))

    result = get_product_reviews(seeded_db, 1, reviews_client=stub, env=_LOCAL_ONLY_ENV)

    assert result["source"] == "external_reviews_http"
    assert result["score"]["stars"] == pytest.approx(4.5)
    assert len(result["reviews"]) == 1
    assert result["reviews"][0]["id"] == "rev-1"
    assert stub.calls == [{"product_id": "SKU-100"}]


def test_get_product_reviews_respects_env_flag(seeded_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubExternalReviewsClient(_sample_external_product("SKU-100"))
    monkeypatch.setattr(
        "tools.catalog_tools._default_external_reviews_client",
        lambda: stub,
    )

    result = get_product_reviews(
        seeded_db,
        1,
        env={"CATALOG_REVIEWS_SOURCE": "external", "COMMERCE_PRODUCT_SOURCE": "local"},
    )

    assert result["source"] == "external_reviews_http"
    assert result["score"]["ratingsTotal"] == 120


def test_get_product_reviews_handles_product_not_found_error(seeded_db: Path) -> None:
    stub = _StubExternalReviewsClient(ExternalReviewsProductError(4, "could not find product"))

    result = get_product_reviews(seeded_db, 1, reviews_client=stub, env=_LOCAL_ONLY_ENV)

    assert result["source"] == "external_reviews_http"
    assert result["reviews"] == []
    assert result["score"] is None
    assert result["error"] == {"status_code": 4, "status_message": "could not find product"}


def test_filter_products_can_attach_live_reviews(seeded_db: Path) -> None:
    stub = _StubExternalReviewsClient(_sample_external_product("SKU-100"))

    products = filter_products(
        seeded_db,
        max_price_cents=3000,
        include_live_reviews=True,
        reviews_client=stub,
        env=_LOCAL_ONLY_ENV,
    )

    assert len(products) == 1
    assert products[0]["score"]["stars"] == pytest.approx(4.5)
    assert products[0]["reviews"][0]["id"] == "rev-1"


def test_filter_products_without_flag_has_no_review_fields(seeded_db: Path) -> None:
    products = filter_products(seeded_db, max_price_cents=3000, env=_LOCAL_ONLY_ENV)

    assert "score" not in products[0]
    assert "reviews" not in products[0]
