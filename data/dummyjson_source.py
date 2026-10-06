"""Zero-authentication primary product source backed by DummyJSON."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DUMMYJSON_DEFAULT_BASE_URL = "https://dummyjson.com"
DEFAULT_TIMEOUT_SECONDS = 3.0


class DummyJsonError(RuntimeError):
    """Base error for any failure while talking to the DummyJSON API."""


class DummyJsonRequestError(DummyJsonError):
    """Raised for network failures, timeouts, or transport-level errors."""


class DummyJsonNotFoundError(DummyJsonError):
    """Raised when the requested product does not exist upstream."""


def _cents_from_price(price: Any) -> int:
    """Convert a DummyJSON float price (e.g. ``9.99``) into integer cents."""

    return round(float(price) * 100)


def _policy_codes_from_product(payload: dict[str, Any]) -> list[str]:
    """Derive deterministic pseudo policy codes from DummyJSON fields."""

    return_policy = str(payload.get("returnPolicy") or "").strip().lower()
    return ["RETURNS"] if return_policy and "no return" not in return_policy else []


def normalize_product(payload: dict[str, Any]) -> dict[str, Any]:
    """Map a raw DummyJSON product payload onto the local product shape."""

    price_cents = _cents_from_price(payload.get("price", 0))
    stock = payload.get("stock")
    return {
        "id": int(payload["id"]),
        "sku": str(payload.get("sku") or f"DJ-{payload['id']}"),
        "name": str(payload.get("title", "")),
        "description": payload.get("description"),
        "price_cents": price_cents,
        "is_active": bool(stock is None or stock > 0),
        "policy_codes": _policy_codes_from_product(payload),
        "category": payload.get("category"),
        "price_eur": round(price_cents / 100.0, 2),
        "weight_g": (
            round(float(payload["weight"]) * 1000)
            if payload.get("weight") is not None
            else None
        ),
        "material": payload.get("brand"),
        "color": None,
        "source": "dummyjson",
    }


@dataclass(slots=True)
class DummyJsonProductClient:
    """Minimal client for the public DummyJSON products endpoint."""

    base_url: str = DUMMYJSON_DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        env_base_url = os.environ.get("DUMMYJSON_BASE_URL")
        if env_base_url and self.base_url == DUMMYJSON_DEFAULT_BASE_URL:
            self.base_url = env_base_url.rstrip("/")

    def get_product(self, product_id: int) -> dict[str, Any]:
        """Fetch and normalize a single product by its DummyJSON id."""

        payload = self._get_json(f"/products/{int(product_id)}")
        if "message" in payload and "id" not in payload:
            msg = f"DummyJSON product {product_id} was not found."
            raise DummyJsonNotFoundError(msg)
        return normalize_product(payload)

    def list_products(self, *, limit: int = 100, skip: int = 0) -> list[dict[str, Any]]:
        """Fetch and normalize a page of products."""

        payload = self._get_json("/products", params={"limit": limit, "skip": skip})
        products = payload.get("products")
        if not isinstance(products, list):
            msg = "DummyJSON returned an unexpected list-products payload shape."
            raise DummyJsonRequestError(msg)
        return [normalize_product(item) for item in products]

    def search_products(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch and normalize products matching a free-text search query."""

        payload = self._get_json("/products/search", params={"q": query, "limit": limit})
        products = payload.get("products")
        if not isinstance(products, list):
            msg = "DummyJSON returned an unexpected search payload shape."
            raise DummyJsonRequestError(msg)
        return [normalize_product(item) for item in products]

    def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue a single GET request and return the parsed JSON body."""

        import httpx  # noqa: PLC0415

        url = f"{self.base_url}{path}"
        try:
            response = httpx.get(url, params=params, timeout=self.timeout_seconds)
        except httpx.TimeoutException as exc:
            msg = f"DummyJSON request to {url} timed out after {self.timeout_seconds}s."
            raise DummyJsonRequestError(msg) from exc
        except httpx.RequestError as exc:
            msg = f"DummyJSON request to {url} failed: {exc}"
            raise DummyJsonRequestError(msg) from exc

        if response.status_code == 404:
            msg = f"DummyJSON resource not found: {url}"
            raise DummyJsonNotFoundError(msg)
        if response.status_code >= 400:
            msg = f"DummyJSON request to {url} returned HTTP {response.status_code}."
            raise DummyJsonRequestError(msg)

        try:
            body = response.json()
        except ValueError as exc:
            msg = f"DummyJSON returned a non-JSON response from {url}."
            raise DummyJsonRequestError(msg) from exc

        if not isinstance(body, dict):
            msg = f"Unexpected DummyJSON response shape from {url}: {type(body).__name__}"
            raise DummyJsonRequestError(msg)
        return body


def is_dummyjson_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether DummyJSON should be used as the primary product source."""

    environment = env if env is not None else os.environ
    return environment.get("COMMERCE_PRODUCT_SOURCE", "dummyjson").lower() != "local"
