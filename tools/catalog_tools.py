"""Catalog query helpers."""

from __future__ import annotations

import functools
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from data.dummyjson_source import (
    DummyJsonError,
    DummyJsonProductClient,
    is_dummyjson_enabled,
)
from data.loaders import connect_database

_CACHE_TTL_SECONDS = 30.0
_cache: dict[tuple[Any, ...], tuple[float, Any]] = {}

_CACHE_BYPASS_KWARGS = ("dummyjson_client", "reviews_client", "env")


def _ttl_cache(key_prefix: str):  # noqa: ANN201 - returns a decorator
    """Cache a function's result briefly, keyed on its own arguments."""

    def decorator(func):  # noqa: ANN001, ANN202
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if any(kwargs.get(name) is not None for name in _CACHE_BYPASS_KWARGS):
                return func(*args, **kwargs)

            key = (key_prefix, args, tuple(sorted(kwargs.items())))
            entry = _cache.get(key)
            if entry is not None and time.monotonic() < entry[0]:
                return entry[1]

            result = func(*args, **kwargs)
            ttl = float(os.environ.get("CATALOG_CACHE_TTL_SECONDS", _CACHE_TTL_SECONDS))
            _cache[key] = (time.monotonic() + ttl, result)
            return result

        return wrapper

    return decorator


class ProductNotFoundError(LookupError):
    """Raised when a requested product does not exist."""


def _is_live_reviews_source(env: Mapping[str, str] | None = None) -> bool:
    """Return whether catalog tools should query the live external reviews API."""

    environment = env or os.environ
    return environment.get("CATALOG_REVIEWS_SOURCE", "local").lower() == "external"


def _default_external_reviews_client() -> Any:
    """Build an external reviews client, or return None if the side-car is unavailable."""

    try:
        from integrations.product_reviews_api import ExternalReviewClient  # noqa: PLC0415
    except ImportError:
        return None
    return ExternalReviewClient()


def _get_local_product_reviews(
    db_path: str | Path,
    product_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Read reviews and aggregate rating for a product from local SQLite."""

    connection = connect_database(db_path)
    connection.row_factory = sqlite3.Row
    try:
        product_row = connection.execute(
            "SELECT rating_stars, rating_count, rating_grade FROM products WHERE id = ?",
            (product_id,),
        ).fetchone()
        review_rows = connection.execute(
            """
            SELECT external_review_id, stars, title, text, network, author, created_at
            FROM product_reviews
            WHERE product_id = ?
            ORDER BY id
            """,
            (product_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return [], None
    finally:
        connection.close()

    score = None
    if product_row is not None and any(value is not None for value in product_row):
        score = dict(product_row)
    return [dict(row) for row in review_rows], score


@_ttl_cache("get_product_reviews")
def get_product_reviews(
    db_path: str | Path,
    product_id: int,
    *,
    reviews_identifier: str | None = None,
    reviews_client: Any | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return review and rating info for a product."""

    product = get_product_by_id(db_path, product_id, env=env)
    payload = {"product_id": product["id"], "sku": product["sku"]}

    if reviews_client is None and not _is_live_reviews_source(env):
        reviews, score = _get_local_product_reviews(db_path, product_id)
        return {**payload, "source": "local_fixture", "reviews": reviews, "score": score}

    external = {**payload, "source": "external_reviews_http"}

    def unavailable(status_code: int | None, status_message: str) -> dict[str, Any]:
        error = {"status_code": status_code, "status_message": status_message}
        return {**external, "reviews": [], "score": None, "error": error}

    try:
        from integrations.product_reviews_api import ExternalReviewsApiError, ExternalReviewsProductError
    except ImportError as exc:
        return unavailable(None, f"External reviews integration unavailable: {exc}")

    client = reviews_client or _default_external_reviews_client()
    if client is None:
        return unavailable(None, "External reviews integration unavailable.")

    try:
        external_product = client.get_product(product_id=reviews_identifier or product["sku"])
    except ExternalReviewsProductError as exc:
        return unavailable(exc.status_code, exc.status_message)
    except ExternalReviewsApiError as exc:
        return unavailable(None, str(exc))

    return {
        **external,
        "reviews": [review.model_dump() for review in external_product.reviews],
        "score": external_product.score.model_dump(),
    }


@dataclass(slots=True)
class ProductRecord:
    """Structured product data loaded from SQLite."""

    id: int
    sku: str
    name: str
    description: str | None
    price_cents: int
    is_active: bool
    policy_codes: list[str]


def _open_connection(db_path: str | Path) -> sqlite3.Connection:
    """Create a row-based SQLite connection for catalog queries."""

    connection = connect_database(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _fetch_product_row(
    connection: sqlite3.Connection,
    product_id: int,
) -> sqlite3.Row:
    """Fetch a single product row by identifier, raising if it does not exist."""

    row = connection.execute(
        """
        SELECT id, sku, name, description, price_cents, is_active
        FROM products
        WHERE id = ?
        """,
        (product_id,),
    ).fetchone()
    if row is None:
        msg = f"Product with id {product_id} was not found."
        raise ProductNotFoundError(msg)
    return row


def _fetch_policy_codes(
    connection: sqlite3.Connection,
    product_id: int,
) -> list[str]:
    """Return sorted policy codes for a product."""

    rows = connection.execute(
        """
        SELECT pol.code
        FROM product_policies AS pp
        JOIN policies AS pol ON pol.id = pp.policy_id
        WHERE pp.product_id = ?
        ORDER BY pol.code
        """,
        (product_id,),
    ).fetchall()
    return [str(row["code"]) for row in rows]


def _build_product_record(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> ProductRecord:
    """Convert a SQLite row into a product record."""

    return ProductRecord(
        id=int(row["id"]),
        sku=str(row["sku"]),
        name=str(row["name"]),
        description=row["description"],
        price_cents=int(row["price_cents"]),
        is_active=bool(row["is_active"]),
        policy_codes=_fetch_policy_codes(connection, int(row["id"])),
    )


def _get_product_by_id_local(db_path: str | Path, product_id: int) -> dict[str, Any]:
    """Return one product by identifier from the local SQLite catalog."""

    connection = _open_connection(db_path)
    try:
        row = _fetch_product_row(connection, product_id)
        return {**asdict(_build_product_record(connection, row)), "source": "local_fixture"}
    finally:
        connection.close()


@_ttl_cache("get_product_by_id")
def get_product_by_id(
    db_path: str | Path,
    product_id: int,
    *,
    dummyjson_client: DummyJsonProductClient | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return one product by identifier."""

    if is_dummyjson_enabled(env):
        client = dummyjson_client or DummyJsonProductClient()
        try:
            product = client.get_product(product_id)
        except DummyJsonError:
            pass
        else:
            product.setdefault("policy_codes", [])
            return product

    return _get_product_by_id_local(db_path, product_id)


@_ttl_cache("search_products")
def search_products(
    query: str,
    *,
    limit: int = 20,
    dummyjson_client: DummyJsonProductClient | None = None,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Search products by free-text query."""

    if not is_dummyjson_enabled(env):
        return []

    client = dummyjson_client or DummyJsonProductClient()
    try:
        return client.search_products(query, limit=limit)
    except DummyJsonError:
        return []


@_ttl_cache("filter_products")
def filter_products(
    db_path: str | Path,
    *,
    is_active: bool | None = None,
    max_price_cents: int | None = None,
    policy_code: str | None = None,
    include_live_reviews: bool = False,
    reviews_client: Any | None = None,
    dummyjson_client: DummyJsonProductClient | None = None,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return products matching the provided filters."""

    with_live_reviews = include_live_reviews or _is_live_reviews_source(env)

    if policy_code is None and is_dummyjson_enabled(env):
        client = dummyjson_client or DummyJsonProductClient()
        try:
            candidates = client.list_products(limit=100)
        except DummyJsonError:
            pass
        else:
            if not with_live_reviews:
                records = [
                    product
                    for product in candidates
                    if (is_active is None or product["is_active"] == is_active)
                    and (max_price_cents is None or product["price_cents"] <= max_price_cents)
                ]
                return sorted(records, key=lambda item: item["id"])

    query = """
        SELECT DISTINCT p.id, p.sku, p.name, p.description, p.price_cents, p.is_active
        FROM products AS p
        LEFT JOIN product_policies AS pp ON pp.product_id = p.id
        LEFT JOIN policies AS pol ON pol.id = pp.policy_id
    """
    conditions: list[str] = []
    parameters: list[Any] = []

    if is_active is not None:
        conditions.append("p.is_active = ?")
        parameters.append(int(is_active))
    if max_price_cents is not None:
        conditions.append("p.price_cents <= ?")
        parameters.append(max_price_cents)
    if policy_code is not None:
        conditions.append("pol.code = ?")
        parameters.append(policy_code)

    if conditions:
        query = f"{query} WHERE {' AND '.join(conditions)}"
    query = f"{query} ORDER BY p.id"

    connection = _open_connection(db_path)
    try:
        rows = connection.execute(query, parameters).fetchall()
        records = [asdict(_build_product_record(connection, row)) for row in rows]
    finally:
        connection.close()

    if not with_live_reviews:
        return records

    client = reviews_client or _default_external_reviews_client()
    for record in records:
        reviews_payload = get_product_reviews(
            db_path,
            record["id"],
            reviews_client=client,
            env=env,
        )
        record["reviews"] = reviews_payload["reviews"]
        record["score"] = reviews_payload["score"]
    return records


def compare_products(
    db_path: str | Path,
    left_product_id: int,
    right_product_id: int,
) -> dict[str, Any]:
    """Compare two products loaded strictly from local SQLite state."""

    connection = _open_connection(db_path)
    try:
        left = _build_product_record(connection, _fetch_product_row(connection, left_product_id))
        right = _build_product_record(connection, _fetch_product_row(connection, right_product_id))
    finally:
        connection.close()

    left_codes = set(left.policy_codes)
    right_codes = set(right.policy_codes)
    return {
        "left_product": asdict(left),
        "right_product": asdict(right),
        "price_difference_cents": left.price_cents - right.price_cents,
        "common_policy_codes": sorted(left_codes & right_codes),
        "left_only_policy_codes": sorted(left_codes - right_codes),
        "right_only_policy_codes": sorted(right_codes - left_codes),
    }


def get_lightest_product_under_price(
    db_path: str | Path,
    *,
    max_price_cents: int,
    category: str | None = None,
) -> dict[str, Any]:
    """Return the lightest product under the given price threshold."""

    query = """
        SELECT id, sku, name, price_cents, category, weight_g, material, color
        FROM products
        WHERE is_active = 1
          AND price_cents <= ?
          AND weight_g IS NOT NULL
    """
    parameters: list[Any] = [max_price_cents]
    if category is not None:
        query = f"{query} AND LOWER(category) = LOWER(?)"
        parameters.append(category)
    query = f"{query} ORDER BY weight_g ASC, price_cents ASC, id ASC LIMIT 1"

    connection = _open_connection(db_path)
    try:
        row = connection.execute(query, parameters).fetchone()
    finally:
        connection.close()
    if row is None:
        msg = "No product matched the requested price and category constraints."
        raise ProductNotFoundError(msg)

    return {
        "id": row["id"],
        "sku": row["sku"],
        "name": row["name"],
        "category": row["category"],
        "price_cents": row["price_cents"],
        "price_eur": round(row["price_cents"] / 100.0, 2),
        "weight_g": row["weight_g"],
        "material": row["material"],
        "color": row["color"],
    }
