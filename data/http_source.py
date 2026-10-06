"""Generic HTTP-backed commerce data source used by the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(slots=True)
class HttpDataSource:
    """Load commerce data from a configurable HTTP endpoint."""

    base_url: str
    products_path: str = "/products"
    policies_path: str = "/policies"
    timeout_seconds: float = 5.0

    def get_products(self) -> list[dict[str, object]]:
        """Return products from the configured HTTP endpoint."""

        return self._get_json_list(self.products_path, "products")

    def get_policies(self) -> list[dict[str, object]]:
        """Return policies from the configured HTTP endpoint."""

        return self._get_json_list(self.policies_path, "policies")

    def _get_json_list(
        self,
        path: str,
        label: str,
    ) -> list[dict[str, object]]:
        """Fetch a JSON list from the configured endpoint."""

        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            msg = f"Failed to load {label} from {url}: {exc}"
            raise RuntimeError(msg) from exc

        payload = response.json()
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            msg = f"Expected a list of objects from {url}"
            raise RuntimeError(msg)
        return payload
