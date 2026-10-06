"""Local JSON fixture data source for commerce records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data.loaders import load_json_records


@dataclass(slots=True)
class LocalFixtureDataSource:
    """Load commerce data from local JSON fixture files."""

    fixture_dir: Path

    def get_products(self) -> list[dict[str, object]]:
        """Return fixture products."""

        return load_json_records(self.fixture_dir / "products.json")

    def get_policies(self) -> list[dict[str, object]]:
        """Return fixture policies."""

        return load_json_records(self.fixture_dir / "policies.json")
