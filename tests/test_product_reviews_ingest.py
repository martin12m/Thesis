"""Tests for data/product_reviews_ingest.py: mapping provider JSON to catalog rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.product_reviews_ingest import (
    ProductReviewsIngestError,
    external_reviews_products_from_json,
    external_reviews_result_to_product_dict,
    load_external_reviews_json,
    merge_products,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "fixtures" / "external_reviews_mock_tvs.json"
)

_BARE_RESULT = {
    "infos": {
        "title": "Widget Pro",
        "desc": "A very good widget.",
        "brand": "Acme",
        "cats": ["Widgets"],
        "avgPrice": 29.99,
    },
    "score": {
        "ratingsTotal": 120,
        "positiveRatio": 0.9,
        "stars": 4.3,
        "grade": "Gut",
    },
    "reviews": {
        "reviews": [
            {
                "id": "r1",
                "stars": 5,
                "text": "Works great.",
                "title": "Love it",
                "nw": "Amazon.de",
                "createdAt": "01.01.2026",
                "author": "Amazon.de Kunde",
            },
            {
                "id": "r2",
                "stars": None,
                "text": "",
                "title": "",
            },
        ]
    },
}

_FULL_ENVELOPE = {
    "status": True,
    "statusCode": 1,
    "statusMsg": "success",
    "result": _BARE_RESULT,
}

_ERROR_ENVELOPE = {
    "status": False,
    "statusCode": 4,
    "statusMsg": "could not find product",
    "result": False,
}


def test_load_external_reviews_json_reads_fixture_file() -> None:
    payload = load_external_reviews_json(FIXTURE_PATH)

    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["sku"] == "TV-SAMSUNG-OLED-65"


def test_external_reviews_result_to_product_dict_accepts_bare_result() -> None:
    product = external_reviews_result_to_product_dict(_BARE_RESULT, sku="ACME-WIDGET-1")

    assert product["sku"] == "ACME-WIDGET-1"
    assert product["name"] == "Widget Pro"
    assert product["description"] == "A very good widget."
    assert product["brand"] == "Acme"
    assert product["category"] == "Widgets"
    assert product["price_cents"] == 2999
    assert product["rating_stars"] == 4.3
    assert product["rating_count"] == 120
    assert product["rating_grade"] == "Gut"


def test_external_reviews_result_to_product_dict_accepts_full_envelope() -> None:
    product = external_reviews_result_to_product_dict(_FULL_ENVELOPE, sku="ACME-WIDGET-1")

    assert product["name"] == "Widget Pro"
    assert product["brand"] == "Acme"


def test_external_reviews_result_to_product_dict_skips_empty_review_but_keeps_valid_one() -> None:
    product = external_reviews_result_to_product_dict(_BARE_RESULT, sku="ACME-WIDGET-1")

    assert len(product["reviews"]) == 1
    assert product["reviews"][0]["text"] == "Works great."
    assert product["reviews"][0]["external_review_id"] == "r1"
    assert product["reviews"][0]["network"] == "Amazon.de"


def test_external_reviews_result_to_product_dict_requires_sku() -> None:
    with pytest.raises(ProductReviewsIngestError, match="No SKU"):
        external_reviews_result_to_product_dict(_BARE_RESULT)


def test_external_reviews_result_to_product_dict_raises_on_error_status_code() -> None:
    with pytest.raises(ProductReviewsIngestError, match="statusCode=4"):
        external_reviews_result_to_product_dict(_ERROR_ENVELOPE, sku="ACME-WIDGET-1")


def test_external_reviews_result_to_product_dict_explicit_price_overrides_avg_price() -> None:
    product = external_reviews_result_to_product_dict(
        _BARE_RESULT, sku="ACME-WIDGET-1", price_cents=1000
    )

    assert product["price_cents"] == 1000


def test_external_reviews_products_from_json_maps_fixture_list() -> None:
    payload = load_external_reviews_json(FIXTURE_PATH)
    products = external_reviews_products_from_json(payload)

    assert len(products) == 2
    skus = {item["sku"] for item in products}
    assert skus == {"TV-SAMSUNG-OLED-65", "TV-LG-QLED-65"}

    samsung = next(item for item in products if item["sku"] == "TV-SAMSUNG-OLED-65")
    assert samsung["brand"] == "Samsung"
    assert samsung["price_cents"] == 179900
    assert samsung["rating_stars"] == 4.5
    assert len(samsung["reviews"]) == 2


def test_external_reviews_products_from_json_skips_unmappable_documents() -> None:
    payload = [{"result": _BARE_RESULT}, {"not": "a provider document at all"}]

    products = external_reviews_products_from_json(payload)

    assert products == []


def test_external_reviews_products_from_json_handles_single_document_not_list() -> None:
    document = dict(_BARE_RESULT)
    products = external_reviews_products_from_json({"sku": "SINGLE-1", "result": document})

    assert len(products) == 1
    assert products[0]["sku"] == "SINGLE-1"


def test_merge_products_enriches_matching_sku_and_preserves_price() -> None:
    base_products = [
        {
            "sku": "SKU-100",
            "name": "Starter Plan",
            "description": "Entry-level plan.",
            "price_cents": 1900,
            "policy_codes": ["RETURNS"],
        }
    ]
    external_products = external_reviews_products_from_json(load_external_reviews_json(FIXTURE_PATH))
    external_products.append(
        {
            "sku": "SKU-100",
            "name": "Starter Plan (provider title ignored)",
            "brand": "Acme",
            "rating_stars": 4.9,
            "rating_count": 55,
            "rating_grade": "Exzellent",
            "reviews": [{"stars": 5, "text": "Love the Starter Plan."}],
            "price_cents": 999999,
            "policy_codes": [],
        }
    )

    merged = merge_products(base_products, external_products)
    starter = next(item for item in merged if item["sku"] == "SKU-100")

    assert starter["price_cents"] == 1900
    assert starter["name"] == "Starter Plan"
    assert starter["policy_codes"] == ["RETURNS"]
    assert starter["brand"] == "Acme"
    assert starter["rating_stars"] == 4.9
    assert starter["reviews"] == [{"stars": 5, "text": "Love the Starter Plan."}]


def test_merge_products_appends_external_reviews_only_skus() -> None:
    external_products = external_reviews_products_from_json(load_external_reviews_json(FIXTURE_PATH))

    merged = merge_products([], external_products)

    skus = {item["sku"] for item in merged}
    assert skus == {"TV-SAMSUNG-OLED-65", "TV-LG-QLED-65"}
