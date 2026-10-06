"""Tests for the external product-reviews client, using mocked HTTP responses."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from integrations.product_reviews_api import (
    ExternalReviewClient,
    ExternalReviewsConfigurationError,
    ExternalReviewsProductError,
    ExternalReviewsRequestError,
)

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


class _FakeResponse:
    """Minimal stand-in for an httpx.Response used in tests."""

    def __init__(self, payload: dict[str, Any], *, text: str = "") -> None:
        self._payload = payload
        self.text = text or str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def _success_payload(product_id: str = "6925281937248") -> dict[str, Any]:
    return {
        "status": True,
        "statusMsg": "success",
        "result": {
            "metaInfos": {
                "lastUpdate": 1643442754,
                "lastUpdate_str": "29.01.2022",
                "profileUrl": "https://www.produktbewertungen.org/example",
            },
            "infos": {
                "title": "Apple iPhone 11 Pro Max",
                "desc": "Smartphone with 6.5 inch OLED display.",
                "brand": "Apple",
                "cats": ["Computer, Multimedia & Audio"],
                "avgPrice": 907.07,
                "avgPrice_str": "907,07",
            },
            "score": {
                "ratingsTotal": 16084,
                "ratingsTotal_str": "16.084",
                "positiveRatio": 0.9082,
                "stars": 4.54,
                "stars_str": "4,54",
                "grade": "Sehr Gut",
                "starsDiscrete": 4.5,
            },
            "reviews": {
                "reviews": [
                    {
                        "id": "507344afdf937ab680b108522e7bc09e",
                        "stars": 1,
                        "text": "Dieses Handy kostet jetzt noch immer so viel wie das neue iPhone 13 Pro.",
                        "text_long": "Dieses Handy kostet jetzt noch immer so viel wie das neue iPhone 13 Pro.",
                        "title": "Normalerweise verlieren die Handys einen gewissen Wert",
                        "nw": "Amazon.de",
                        "createdAt": "26.01.2022",
                        "author": "Amazon.de Kunde",
                        "lang": "de",
                        "helpful_votes": False,
                        "helpful_percent": False,
                    }
                ]
            },
        },
        "statusCode": 1,
        "lastUpdate": 1643442754,
        "lastUpdate_str": "29.01.2022",
        "lang": "de",
    }


@pytest.fixture(autouse=True)
def _reviews_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTERNAL_REVIEWS_USER", "test-user")
    monkeypatch.setenv("EXTERNAL_REVIEWS_PASSWORD", "test-password")
    monkeypatch.delenv("EXTERNAL_REVIEWS_BASE_URL", raising=False)


def test_get_product_parses_infos_score_and_reviews(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, auth: Any, data: dict[str, Any], headers: dict[str, Any], timeout: float) -> _FakeResponse:
        captured["url"] = url
        captured["auth"] = auth
        captured["data"] = data
        captured["headers"] = headers
        return _FakeResponse(_success_payload())

    fake_httpx = SimpleNamespace(
        post=fake_post,
        TimeoutException=TimeoutError,
        RequestError=ConnectionError,
    )
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = ExternalReviewClient()
    product = client.get_product(gtin="6925281937248")

    assert captured["url"] == "https://api.example-reviews.com/products/"
    assert captured["auth"] == ("test-user", "test-password")
    assert captured["data"]["gtin"] == "6925281937248"
    assert captured["headers"]["Content-Type"] == "application/x-www-form-urlencoded"

    assert product.infos.title == "Apple iPhone 11 Pro Max"
    assert product.infos.brand == "Apple"
    assert product.infos.avgPrice == pytest.approx(907.07)
    assert product.score.stars == pytest.approx(4.54)
    assert product.score.grade == "Sehr Gut"
    assert product.score.ratingsTotal == 16084
    assert len(product.reviews) == 1
    assert product.reviews[0].id == "507344afdf937ab680b108522e7bc09e"
    assert product.reviews[0].nw == "Amazon.de"


def test_get_product_parses_a_real_trustami_shaped_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate the client against a genuine (non-hand-written) provider response."""

    payload = json.loads(
        (_FIXTURES_DIR / "external_reviews_real_example_radio.json").read_text(encoding="utf-8")
    )

    def fake_post(url: str, *, auth: Any, data: dict[str, Any], headers: dict[str, Any], timeout: float) -> _FakeResponse:
        return _FakeResponse(payload)

    fake_httpx = SimpleNamespace(
        post=fake_post,
        TimeoutException=TimeoutError,
        RequestError=ConnectionError,
    )
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = ExternalReviewClient()
    product = client.get_product(gtin="4260046875032")

    assert product.infos.title.startswith("Teufel Radio 3SIXTY")
    assert product.infos.brand == "Teufel"
    assert product.infos.avgPrice == pytest.approx(284.99)

    assert product.score.stars == pytest.approx(4.68)
    assert product.score.grade == "Sehr Gut"
    assert product.score.ratingsTotal == 934
    assert product.score.stars_distribution["star_5"]["total"] == 765
    assert len(product.score.mergedNetworkScores) == 3
    assert product.score.mergedNetworkScores[0]["name"] == "Otto.de"

    assert len(product.reviews) == 50
    assert product.reviews[0].id == "47534512cc3daf9c7bc4d8e36938ed0d"
    assert product.reviews[0].stars == 5
    assert product.reviews[-1].nw == "Amazon.es"

    assert product.meta.lastUpdate_str == "13.04.2026"
    assert product.meta.profileUrlDe.startswith("https://www.produktbewertungen.org/")


def test_get_product_requires_at_least_one_identifier() -> None:
    client = ExternalReviewClient()

    with pytest.raises(ValueError, match="At least one of"):
        client.get_product()


def test_missing_credentials_raise_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXTERNAL_REVIEWS_USER", raising=False)
    monkeypatch.delenv("EXTERNAL_REVIEWS_PASSWORD", raising=False)

    client = ExternalReviewClient(user=None, password=None)

    with pytest.raises(ExternalReviewsConfigurationError):
        client.get_product(asin="B07XJ8C8F5")


@pytest.mark.parametrize(
    ("status_code", "status_message"),
    [
        (4, "could not find product"),
        (9, "(daily/monthly) limit reached"),
    ],
)
def test_non_success_status_codes_raise_product_error(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    status_message: str,
) -> None:
    def fake_post(url: str, *, auth: Any, data: dict[str, Any], headers: dict[str, Any], timeout: float) -> _FakeResponse:
        return _FakeResponse(
            {
                "status": False,
                "statusMsg": status_message,
                "result": False,
                "statusCode": status_code,
            }
        )

    fake_httpx = SimpleNamespace(
        post=fake_post,
        TimeoutException=TimeoutError,
        RequestError=ConnectionError,
    )
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = ExternalReviewClient()
    with pytest.raises(ExternalReviewsProductError) as exc_info:
        client.get_product(gtin="0000000000000")

    assert exc_info.value.status_code == status_code
    assert exc_info.value.status_message == status_message


def test_timeout_raises_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Timeout(Exception):
        pass

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        raise _Timeout("timed out")

    fake_httpx = SimpleNamespace(
        post=fake_post,
        TimeoutException=_Timeout,
        RequestError=ConnectionError,
    )
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    client = ExternalReviewClient(timeout_seconds=1.0)
    with pytest.raises(ExternalReviewsRequestError, match="timed out"):
        client.get_product(asin="B07XJ8C8F5")
