"""Data source abstractions for commerce fixture and HTTP-backed data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Protocol

from data.http_source import HttpDataSource
from data.local_fixture_source import LocalFixtureDataSource


class CommerceDataSource(Protocol):
    """Protocol for reading commerce data records."""

    def get_products(self) -> list[dict[str, object]]:
        """Return product records."""

    def get_policies(self) -> list[dict[str, object]]:
        """Return policy records."""


def get_configured_data_source(
    env: Mapping[str, str] | None = None,
) -> CommerceDataSource:
    """Return the configured data source with safe local fallback."""

    environment = env or os.environ
    source_name = environment.get("COMMERCE_DATA_SOURCE", "local").lower()
    fixture_dir = Path(
        environment.get(
            "COMMERCE_FIXTURE_DIR",
            str(Path(__file__).resolve().parent / "fixtures"),
        )
    )

    if source_name == "http":
        base_url = environment.get("COMMERCE_HTTP_BASE_URL")
        if base_url:
            return HttpDataSource(
                base_url=base_url,
                products_path=environment.get("COMMERCE_HTTP_PRODUCTS_PATH", "/products"),
                policies_path=environment.get("COMMERCE_HTTP_POLICIES_PATH", "/policies"),
            )
    return LocalFixtureDataSource(fixture_dir=fixture_dir)
