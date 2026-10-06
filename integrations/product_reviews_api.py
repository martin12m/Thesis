"""External product-reviews API integration -- optional side-car, not core path."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from models import StrictModel

logger = logging.getLogger(__name__)

EXTERNAL_REVIEWS_DEFAULT_BASE_URL = "https://api.example-reviews.com/products/"

EXTERNAL_REVIEWS_STATUS_MESSAGES: dict[int, str] = {
    0: "Unknown error",
    1: "success",
    2: "in process",
    3: "searching",
    4: "could not find product",
    5: "product under review",
    6: "Input invalid",
    7: "identifier in blacklist or not in whitelist",
    8: "not enough stars/not enough ratings",
    9: "(daily/monthly) limit reached",
}

_EXTERNAL_REVIEWS_SUCCESS_CODE = 1


class ExternalReviewsApiError(RuntimeError):
    """Base error for any failure while talking to the external reviews API."""


class ExternalReviewsConfigurationError(ExternalReviewsApiError):
    """Raised when required provider credentials or config are missing."""


class ExternalReviewsRequestError(ExternalReviewsApiError):
    """Raised for network failures, timeouts, or non-2xx transport errors."""


class ExternalReviewsProductError(ExternalReviewsApiError):
    """Raised when the provider responds with a non-success status code."""

    def __init__(self, status_code: int, status_message: str) -> None:
        self.status_code = status_code
        self.status_message = status_message
        super().__init__(f"External reviews API returned status_code={status_code}: {status_message}")


class ExternalReviewsMetaInfo(BaseModel):
    """Freshness and profile-link metadata for a product review record."""

    model_config = ConfigDict(extra="ignore")

    lastUpdate: int | str | None = None
    nextUpdate: int | str | None = None
    lastUpdate_str: str | None = None
    nextUpdate_str: str | None = None
    profileUrl: str | None = None
    profileUrlDe: str | None = None
    profileUrlEn: str | None = None


class ExternalProductInfos(BaseModel):
    """General product information returned by the external reviews API."""

    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    desc: str | None = None
    brand: str | None = None
    cats: list[str] = Field(default_factory=list)
    avgPrice: float | str | None = None
    avgPrice_str: str | None = None


class ExternalProductScore(BaseModel):
    """Aggregated rating information returned by the external reviews API."""

    model_config = ConfigDict(extra="ignore")

    ratingsTotal: int | str | None = None
    ratingsTotal_str: str | None = None
    positiveRatio: float | str | None = None
    positiveRatio_str: str | None = None
    stars: float | str | None = None
    stars_str: str | None = None
    grade: str | None = None
    starsDiscrete: float | str | None = None
    starsDiscrete_str: str | None = None
    stars_distribution: dict[str, Any] = Field(default_factory=dict)
    mergedNetworkScores: list[Any] = Field(default_factory=list)


class ExternalProductReview(BaseModel):
    """A single normalized customer review from the external reviews API."""

    model_config = ConfigDict(extra="ignore")

    id: str
    stars: float | int | None = None
    text: str | None = None
    text_long: str | None = None
    title: str | None = None
    nw: str | None = None
    createdAt: str | None = None
    author: str | None = None
    lang: str | None = None
    helpful_votes: int | bool | None = None
    helpful_percent: float | bool | None = None


class ExternalProduct(StrictModel):
    """Normalized, typed view of a single external-reviews product result."""

    product_id: str
    infos: ExternalProductInfos = Field(default_factory=ExternalProductInfos)
    score: ExternalProductScore = Field(default_factory=ExternalProductScore)
    reviews: list[ExternalProductReview] = Field(default_factory=list)
    meta: ExternalReviewsMetaInfo = Field(default_factory=ExternalReviewsMetaInfo)


@dataclass(slots=True)
class ExternalReviewClient:
    """Client for a product-reviews aggregator API (``POST /products/``)."""

    user: str | None = None
    password: str | None = None
    base_url: str = EXTERNAL_REVIEWS_DEFAULT_BASE_URL
    timeout_seconds: float = 10.0
    lang: str = "de"

    def __post_init__(self) -> None:
        self.user = self.user or os.environ.get("EXTERNAL_REVIEWS_USER")
        self.password = self.password or os.environ.get("EXTERNAL_REVIEWS_PASSWORD")
        env_base_url = os.environ.get("EXTERNAL_REVIEWS_BASE_URL")
        if env_base_url and self.base_url == EXTERNAL_REVIEWS_DEFAULT_BASE_URL:
            self.base_url = env_base_url.rstrip("/") + "/"

    def get_product(
        self,
        *,
        gtin: str | None = None,
        asin: str | None = None,
        pzn: str | None = None,
        product_id: str | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> ExternalProduct:
        """Fetch a single product's info, score, and reviews from the provider."""

        identifiers = {
            "gtin": gtin,
            "asin": asin,
            "pzn": pzn,
            "productId": product_id,
        }
        payload = {key: value for key, value in identifiers.items() if value}
        if not payload:
            msg = "At least one of gtin, asin, pzn, or product_id is required."
            raise ValueError(msg)

        payload["lang"] = self.lang
        if extra_params:
            payload.update(extra_params)

        response_json = self._post(payload)
        return self._parse_product_response(
            response_json,
            fallback_id=product_id or gtin or asin or pzn or "",
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send the authenticated form-encoded POST request to the provider."""

        if not self.user or not self.password:
            msg = (
                "External reviews credentials are missing. Set "
                "EXTERNAL_REVIEWS_USER and EXTERNAL_REVIEWS_PASSWORD "
                "environment variables or pass them explicitly."
            )
            raise ExternalReviewsConfigurationError(msg)

        import httpx  # noqa: PLC0415

        try:
            response = httpx.post(
                self.base_url,
                auth=(self.user, self.password),
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            msg = f"External reviews API request timed out after {self.timeout_seconds}s."
            raise ExternalReviewsRequestError(msg) from exc
        except httpx.RequestError as exc:
            msg = f"External reviews API request failed: {exc}"
            raise ExternalReviewsRequestError(msg) from exc

        try:
            body = response.json()
        except ValueError as exc:
            response.raise_for_status()
            msg = f"External reviews API returned a non-JSON response: {response.text[:200]}"
            raise ExternalReviewsRequestError(msg) from exc

        if not isinstance(body, dict):
            msg = f"Unexpected external reviews API response shape: {type(body).__name__}"
            raise ExternalReviewsRequestError(msg)

        return body

    def _parse_product_response(
        self,
        body: dict[str, Any],
        *,
        fallback_id: str,
    ) -> ExternalProduct:
        """Validate the status envelope and build a typed product model."""

        status_code = int(body.get("statusCode", 0))
        status_message = str(
            body.get("statusMsg", EXTERNAL_REVIEWS_STATUS_MESSAGES.get(status_code, "Unknown error"))
        )

        if status_code != _EXTERNAL_REVIEWS_SUCCESS_CODE or not body.get("status", False):
            logger.warning(
                "External reviews API non-success response: status_code=%s status_message=%s",
                status_code,
                status_message,
            )
            raise ExternalReviewsProductError(status_code, status_message)

        result = body.get("result")
        if not isinstance(result, dict):
            msg = "External reviews API reported success but returned no result payload."
            raise ExternalReviewsRequestError(msg)

        reviews_block = result.get("reviews", {})
        review_items: list[dict[str, Any]] = []
        if isinstance(reviews_block, dict):
            review_items = list(reviews_block.get("reviews", []))
        elif isinstance(reviews_block, list):
            review_items = list(reviews_block)

        return ExternalProduct(
            product_id=fallback_id,
            infos=result.get("infos", {}) or {},
            score=result.get("score", {}) or {},
            reviews=review_items,
            meta=result.get("metaInfos", {}) or {},
        )
