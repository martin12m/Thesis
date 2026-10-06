"""Ingest external product-review JSON (mock fixtures or live API responses)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SUCCESS_STATUS_CODE = 1


class ProductReviewsIngestError(ValueError):
    """Raised when external-reviews-shaped JSON cannot be mapped to a product row."""


def load_external_reviews_json(path: str | Path) -> Any:
    """Read a JSON file that holds one or more external-reviews-shaped documents."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_result(document: dict[str, Any]) -> dict[str, Any]:
    """Return the inner ``result`` object, accepting either input shape."""

    if any(key in document for key in ("infos", "score", "reviews")):
        return document

    status_code = document.get("statusCode")
    if status_code is not None and int(status_code) != _SUCCESS_STATUS_CODE:
        status_message = document.get("statusMsg", "Unknown error")
        msg = f"Envelope reports non-success statusCode={status_code} ({status_message})"
        raise ProductReviewsIngestError(msg)

    result = document.get("result")
    if not isinstance(result, dict):
        msg = "Document has no usable 'result', 'infos', 'score', or 'reviews' key."
        raise ProductReviewsIngestError(msg)
    return result


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion, tolerating provider ``str|float`` union fields."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    """Best-effort int coercion, tolerating provider ``str|int`` union fields."""

    coerced = _coerce_float(value)
    return int(coerced) if coerced is not None else None


def _extract_review_items(reviews_block: Any) -> list[dict[str, Any]]:
    """Normalize the ``reviews`` field, nested as ``{"reviews": [...]}`` or bare."""

    if isinstance(reviews_block, dict):
        items = reviews_block.get("reviews", [])
    elif isinstance(reviews_block, list):
        items = reviews_block
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def _map_review(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Map one raw review object to the ``product_reviews`` row shape."""

    stars = _coerce_float(raw.get("stars"))
    text = raw.get("text") or raw.get("text_long")
    if stars is None and not text:
        return None

    return {
        "external_review_id": raw.get("id"),
        "stars": stars,
        "title": raw.get("title") or None,
        "text": text,
        "network": raw.get("nw"),
        "author": raw.get("author"),
        "created_at": raw.get("createdAt"),
    }


def external_reviews_result_to_product_dict(
    document: dict[str, Any],
    *,
    sku: str | None = None,
    price_cents: int | None = None,
    policy_codes: list[str] | None = None,
    category: str | None = None,
    is_active: bool = True,
) -> dict[str, Any]:
    """Map a single external-reviews-shaped document to a seed-ready product dict."""

    result = _extract_result(document)
    infos = result.get("infos") or {}
    score = result.get("score") or {}
    review_items = _extract_review_items(result.get("reviews"))

    resolved_sku = (
        sku
        or document.get("sku")
        or document.get("productId")
        or document.get("product_id")
        or document.get("gtin")
        or document.get("asin")
    )
    if not resolved_sku:
        msg = (
            "No SKU could be resolved for this document -- pass sku= "
            "explicitly (provider identifiers such as GTIN/ASIN will not "
            "match your local catalog SKUs unless used consistently)."
        )
        raise ProductReviewsIngestError(msg)

    name = infos.get("title") or resolved_sku
    cats = infos.get("cats") or []
    resolved_category = category or (cats[0] if cats else None)

    avg_price = _coerce_float(infos.get("avgPrice"))
    resolved_price_cents = (
        price_cents if price_cents is not None else (round(avg_price * 100) if avg_price is not None else 0)
    )

    reviews = [mapped for raw in review_items if (mapped := _map_review(raw)) is not None]

    return {
        "sku": str(resolved_sku),
        "name": name,
        "description": infos.get("desc"),
        "price_cents": resolved_price_cents,
        "is_active": is_active,
        "category": resolved_category,
        "weight_g": None,
        "material": None,
        "color": None,
        "brand": infos.get("brand"),
        "rating_stars": _coerce_float(score.get("stars")),
        "rating_count": _coerce_int(score.get("ratingsTotal")),
        "rating_grade": score.get("grade"),
        "policy_codes": policy_codes or [],
        "reviews": reviews,
    }


def external_reviews_products_from_json(
    payload: Any,
    *,
    sku_field: str = "sku",
    price_cents_field: str = "price_cents",
    policy_codes_field: str = "policy_codes",
) -> list[dict[str, Any]]:
    """Convert an external-reviews-shaped JSON payload into product dicts."""

    documents = payload if isinstance(payload, list) else [payload]
    products: list[dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        try:
            products.append(
                external_reviews_result_to_product_dict(
                    document,
                    sku=document.get(sku_field),
                    price_cents=document.get(price_cents_field),
                    policy_codes=document.get(policy_codes_field),
                )
            )
        except ProductReviewsIngestError:
            continue
    return products


def merge_products(
    base_products: list[dict[str, Any]],
    external_products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge external-review enrichment onto base products by SKU."""

    external_by_sku = {item["sku"]: item for item in external_products}
    merged: list[dict[str, Any]] = []
    seen_skus: set[str] = set()

    for base in base_products:
        sku = base["sku"]
        seen_skus.add(sku)
        enrichment = external_by_sku.get(sku)
        if enrichment is None:
            merged.append(base)
            continue
        merged_item = dict(base)
        for field in ("brand", "rating_stars", "rating_count", "rating_grade"):
            if enrichment.get(field) is not None:
                merged_item[field] = enrichment[field]
        if enrichment.get("reviews"):
            merged_item["reviews"] = enrichment["reviews"]
        merged.append(merged_item)

    for sku, enrichment in external_by_sku.items():
        if sku not in seen_skus:
            merged.append(enrichment)

    return merged
