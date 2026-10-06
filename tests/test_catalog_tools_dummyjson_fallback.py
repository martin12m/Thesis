"""Tests for the DummyJSON-primary / local-fallback wiring in catalog_tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from data.dummyjson_source import DummyJsonRequestError
from scripts.seed_data import seed_from_files
from tools.catalog_tools import filter_products, get_product_by_id, search_products


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


class _FailingDummyJsonClient:
    """Stub client that always raises, simulating an unreachable DummyJSON."""

    def get_product(self, product_id: int) -> dict[str, Any]:
        raise DummyJsonRequestError("simulated network failure")

    def list_products(self, *, limit: int = 100, skip: int = 0) -> list[dict[str, Any]]:
        raise DummyJsonRequestError("simulated network failure")

    def search_products(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        raise DummyJsonRequestError("simulated network failure")


class _StubDummyJsonClient:
    """Stub client returning a fixed, deterministic payload."""

    def __init__(self, product: dict[str, Any]) -> None:
        self._product = product

    def get_product(self, product_id: int) -> dict[str, Any]:
        assert product_id == self._product["id"]
        return dict(self._product)

    def list_products(self, *, limit: int = 100, skip: int = 0) -> list[dict[str, Any]]:
        return [dict(self._product)]

    def search_products(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        return [dict(self._product)]


def _sample_dummyjson_product() -> dict[str, Any]:
    return {
        "id": 1,
        "sku": "RUN-002",
        "name": "FeatherFly Pro",
        "description": "A lightweight running shoe.",
        "price_cents": 9500,
        "is_active": True,
        "policy_codes": ["RETURNS"],
        "category": "running-shoes",
        "price_eur": 95.0,
        "weight_g": 210,
        "material": "Acme",
        "color": None,
        "source": "dummyjson",
    }


def test_get_product_by_id_uses_dummyjson_by_default(seeded_db: Path) -> None:
    stub = _StubDummyJsonClient(_sample_dummyjson_product())

    product = get_product_by_id(seeded_db, 1, dummyjson_client=stub)

    assert product["source"] == "dummyjson"
    assert product["name"] == "FeatherFly Pro"


def test_get_product_by_id_falls_back_to_local_on_dummyjson_failure(seeded_db: Path) -> None:
    product = get_product_by_id(seeded_db, 1, dummyjson_client=_FailingDummyJsonClient())

    assert product["source"] == "local_fixture"
    assert product["sku"] == "SKU-100"
    assert product["name"] == "Starter Plan"


def test_get_product_by_id_local_override_skips_dummyjson_entirely(seeded_db: Path) -> None:
    product = get_product_by_id(
        seeded_db,
        1,
        dummyjson_client=_FailingDummyJsonClient(),
        env={"COMMERCE_PRODUCT_SOURCE": "local"},
    )

    assert product["source"] == "local_fixture"
    assert product["sku"] == "SKU-100"


def test_filter_products_uses_dummyjson_by_default(seeded_db: Path) -> None:
    stub = _StubDummyJsonClient(_sample_dummyjson_product())

    products = filter_products(seeded_db, max_price_cents=10000, dummyjson_client=stub)

    assert len(products) == 1
    assert products[0]["source"] == "dummyjson"


def test_filter_products_falls_back_to_local_on_dummyjson_failure(seeded_db: Path) -> None:
    products = filter_products(
        seeded_db,
        max_price_cents=3000,
        dummyjson_client=_FailingDummyJsonClient(),
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


def test_filter_products_with_policy_code_always_uses_local(seeded_db: Path) -> None:
    stub = _StubDummyJsonClient(_sample_dummyjson_product())

    products = filter_products(
        seeded_db,
        policy_code="RETURNS",
        dummyjson_client=stub,
    )

    assert all(product.get("source") != "dummyjson" for product in products)
    assert products[0]["sku"] == "SKU-100"


def test_search_products_uses_dummyjson_client(seeded_db: Path) -> None:
    stub = _StubDummyJsonClient(_sample_dummyjson_product())

    results = search_products("running shoe", dummyjson_client=stub)

    assert len(results) == 1
    assert results[0]["name"] == "FeatherFly Pro"


def test_search_products_returns_empty_list_on_failure() -> None:
    results = search_products("anything", dummyjson_client=_FailingDummyJsonClient())

    assert results == []


def test_search_products_returns_empty_list_when_disabled_locally() -> None:
    results = search_products(
        "anything",
        dummyjson_client=_StubDummyJsonClient(_sample_dummyjson_product()),
        env={"COMMERCE_PRODUCT_SOURCE": "local"},
    )

    assert results == []


def test_get_product_by_id_caches_default_call_shape(
    seeded_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default (no explicit client/env) call shape must be cached."""

    call_count = 0
    product = _sample_dummyjson_product()

    class _CountingClient:
        def get_product(self, product_id: int) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return dict(product)

    monkeypatch.setattr("tools.catalog_tools.DummyJsonProductClient", _CountingClient)

    first = get_product_by_id(seeded_db, 1)
    second = get_product_by_id(seeded_db, 1)

    assert call_count == 1  # second call was served from the cache
    assert first == second


def test_get_product_by_id_explicit_client_override_bypasses_cache(seeded_db: Path) -> None:
    """An explicit dummyjson_client must never be cached against, even though a plain test-stub class (unlike the real client) is technically hashable -- caching is bypassed by parameter name, not by hashability."""

    call_count = 0
    product = _sample_dummyjson_product()

    class _CountingClient:
        def get_product(self, product_id: int) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return dict(product)

    client = _CountingClient()
    get_product_by_id(seeded_db, 1, dummyjson_client=client)
    get_product_by_id(seeded_db, 1, dummyjson_client=client)

    assert call_count == 2  # never cached: called fresh both times


def test_get_product_by_id_explicit_env_override_bypasses_cache(
    seeded_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    product = _sample_dummyjson_product()

    class _CountingClient:
        def get_product(self, product_id: int) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return dict(product)

    monkeypatch.setattr("tools.catalog_tools.DummyJsonProductClient", _CountingClient)

    get_product_by_id(seeded_db, 1, env={"COMMERCE_PRODUCT_SOURCE": "dummyjson"})
    get_product_by_id(seeded_db, 1, env={"COMMERCE_PRODUCT_SOURCE": "dummyjson"})

    assert call_count == 2  # an explicit env dict is never cached against


def test_get_product_by_id_cache_entry_expires(
    seeded_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    product = _sample_dummyjson_product()
    fake_now = 1_000.0

    class _CountingClient:
        def get_product(self, product_id: int) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return dict(product)

    monkeypatch.setattr("tools.catalog_tools.DummyJsonProductClient", _CountingClient)
    monkeypatch.setattr("tools.catalog_tools.time.monotonic", lambda: fake_now)
    monkeypatch.setenv("CATALOG_CACHE_TTL_SECONDS", "30")

    get_product_by_id(seeded_db, 1)
    assert call_count == 1

    fake_now += 31.0  # past the 30s TTL
    get_product_by_id(seeded_db, 1)
    assert call_count == 2  # cache entry had expired, so this re-fetched
