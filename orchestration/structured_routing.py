"""Shared structured (policy/catalog) tool-routing logic."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tools.catalog_tools import (
    compare_products,
    filter_products,
    get_lightest_product_under_price,
    get_product_by_id,
    get_product_reviews,
)
from tools.policy_tools import (
    get_return_policy_info,
    get_shipping_policy_info,
    get_warranty_policy_info,
)


def collect_structured_routes(user_turn: str) -> list[dict[str, Any]]:
    """Collect all structured (policy/catalog) routes that match the query."""

    lowered = user_turn.lower()
    routes: list[dict[str, Any]] = []
    if "return" in lowered:
        routes.append({"type": "policy", "tool_name": "get_return_policy_info"})
    elif "shipping" in lowered:
        routes.append({"type": "policy", "tool_name": "get_shipping_policy_info"})
    elif "warranty" in lowered:
        routes.append({"type": "policy", "tool_name": "get_warranty_policy_info"})

    if "lightest" in lowered:
        max_price_eur = extract_price_limit_eur(user_turn)
        routes.append(
            {
                "type": "catalog",
                "tool_name": "get_lightest_product_under_price",
                "arguments": {
                    "max_price_cents": max_price_eur * 100,
                    "category": "running shoe" if "running shoe" in lowered else None,
                },
            }
        )
        return routes

    product_ids = [int(value) for value in re.findall(r"\b\d+\b", user_turn)]
    if "compare" in lowered and len(product_ids) >= 2:
        routes.append(
            {
                "type": "catalog",
                "tool_name": "compare_products",
                "arguments": {
                    "left_product_id": product_ids[0],
                    "right_product_id": product_ids[1],
                },
            }
        )
    elif ("review" in lowered or "rating" in lowered) and product_ids:
        routes.append(
            {
                "type": "catalog",
                "tool_name": "get_product_reviews",
                "arguments": {"product_id": product_ids[0]},
            }
        )
    elif ("product" in lowered or "plan" in lowered) and product_ids:
        routes.append(
            {
                "type": "catalog",
                "tool_name": "get_product_by_id",
                "arguments": {"product_id": product_ids[0]},
            }
        )
    elif "active products" in lowered:
        routes.append(
            {
                "type": "catalog",
                "tool_name": "filter_products",
                "arguments": {"is_active": True},
            }
        )
    return routes


def extract_price_limit_eur(user_turn: str) -> int:
    """Extract an integer euro price limit from the query."""

    match = re.search(r"(\d+)\s*euros?", user_turn.lower())
    if match is None:
        return 100
    return int(match.group(1))


def run_policy_tool(db_path: str | Path, tool_name: str) -> dict[str, Any]:
    """Execute a policy lookup tool by name."""

    tool_map = {
        "get_return_policy_info": get_return_policy_info,
        "get_shipping_policy_info": get_shipping_policy_info,
        "get_warranty_policy_info": get_warranty_policy_info,
    }
    return tool_map[tool_name](db_path)


def run_catalog_tool(
    db_path: str | Path,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Execute a catalog tool by name."""

    tool_map = {
        "get_product_by_id": get_product_by_id,
        "filter_products": filter_products,
        "compare_products": compare_products,
        "get_lightest_product_under_price": get_lightest_product_under_price,
        "get_product_reviews": get_product_reviews,
    }
    tool = tool_map[tool_name]
    if tool_name in ("get_product_by_id", "filter_products", "get_product_reviews"):
        return tool(db_path, **arguments, env={"COMMERCE_PRODUCT_SOURCE": "local"})
    return tool(db_path, **arguments)


def to_context_tool_output(route_type: str, tool_name: str, output: Any) -> dict[str, Any]:
    """Normalize a raw tool output into `ContextManager`'s tool-output shape."""

    source_type = "policy" if route_type == "policy" else "product"
    if isinstance(output, dict):
        title = str(output.get("title") or output.get("name") or output.get("sku") or tool_name)
        content = str(output.get("body") or output.get("description") or output)
    else:
        title = tool_name
        content = str(output)
    return {
        "source_type": source_type,
        "title": title,
        "content": content,
        "tool_name": tool_name,
    }
