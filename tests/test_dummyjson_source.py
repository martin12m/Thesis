"""Tests for the DummyJSON primary product client, using mocked HTTP responses."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from data.dummyjson_source import (
    DummyJsonNotFoundError,
    DummyJsonProductClient,
    DummyJsonRequestError,
    is_dummyjson_enabled,
    normalize_product,
)


class _FakeResponse:
    """Minimal stand-in for an httpx.Response used in tests."""

    def __init__(self, payload: Any, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload


def _sample_payload(product_id: int = 1) -> dict[str, Any]:
    return {
        "id": product_id,
        "title": "FeatherFly Pro",
        "description": "A lightweight running shoe.",
        "category": "running-shoes",
        "price": 95.0,
        "stock": 42,
        "sku": "RUN-002",
        "weight": 0.21,
        "brand": "Acme",
        "returnPolicy": "30 days return policy",
    }


def test_normalize_product_maps_fields_and_units() -> None:
    normalized = normalize_product(_sample_payload())

    assert normalized["id"] == 1
    assert normalized["sku"] == "RUN-002"
    assert normalized["name"] == "FeatherFly Pro"
    assert normalized["price_cents"] == 9500
    assert normalized["price_eur"] == pytest.approx(95.0)
    assert normalized["weight_g"] == 210
    assert normalized["is_active"] is True
    assert normalized["policy_codes"] == ["RETURNS"]
    assert normalized["source"] == "dummyjson"


def test_normalize_product_treats_no_return_policy_as_no_returns_code() -> None:
    payload = _sample_payload()
    payload["returnPolicy"] = "No return policy"

    normalized = normalize_product(payload)

    assert normalized["policy_codes"] == []


def test_normalize_product_treats_zero_stock_as_inactive() -> None:
    payload = _sample_payload()
    payload["stock"] = 0

    normalized = normalize_product(payload)

    assert normalized["is_active"] is False


def test_get_product_returns_normalized_product(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_get(url: str, *, params: Any, timeout: float) -> _FakeResponse:
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse(_sample_payload(7))

    fake_httpx = SimpleNamespace(get=fake_get, TimeoutException=TimeoutError, RequestError=ConnectionError)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = DummyJsonProductClient()
    product = client.get_product(7)

    assert captured["url"] == "https://dummyjson.com/products/7"
    assert product["id"] == 7
    assert product["sku"] == "RUN-002"


def test_get_product_raises_not_found_for_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, params: Any, timeout: float) -> _FakeResponse:
        return _FakeResponse({"message": "Product with id '999' not found"}, status_code=404)

    fake_httpx = SimpleNamespace(get=fake_get, TimeoutException=TimeoutError, RequestError=ConnectionError)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = DummyJsonProductClient()
    with pytest.raises(DummyJsonNotFoundError):
        client.get_product(999)


def test_get_product_raises_request_error_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Timeout(Exception):
        pass

    def fake_get(*args: Any, **kwargs: Any) -> _FakeResponse:
        raise _Timeout("timed out")

    fake_httpx = SimpleNamespace(get=fake_get, TimeoutException=_Timeout, RequestError=ConnectionError)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = DummyJsonProductClient(timeout_seconds=1.0)
    with pytest.raises(DummyJsonRequestError, match="timed out"):
        client.get_product(1)


def test_list_products_normalizes_every_item(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, params: Any, timeout: float) -> _FakeResponse:
        return _FakeResponse({"products": [_sample_payload(1), _sample_payload(2)]})

    fake_httpx = SimpleNamespace(get=fake_get, TimeoutException=TimeoutError, RequestError=ConnectionError)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = DummyJsonProductClient()
    products = client.list_products(limit=10)

    assert [item["id"] for item in products] == [1, 2]


def test_search_products_normalizes_every_item(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, params: Any, timeout: float) -> _FakeResponse:
        assert params["q"] == "shoe"
        return _FakeResponse({"products": [_sample_payload(3)]})

    fake_httpx = SimpleNamespace(get=fake_get, TimeoutException=TimeoutError, RequestError=ConnectionError)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = DummyJsonProductClient()
    products = client.search_products("shoe")

    assert len(products) == 1
    assert products[0]["id"] == 3


def test_is_dummyjson_enabled_defaults_true() -> None:
    assert is_dummyjson_enabled({}) is True


def test_is_dummyjson_enabled_respects_local_override() -> None:
    assert is_dummyjson_enabled({"COMMERCE_PRODUCT_SOURCE": "local"}) is False
    assert is_dummyjson_enabled({"COMMERCE_PRODUCT_SOURCE": "LOCAL"}) is False
