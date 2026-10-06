"""Tests for the shared structured (policy/catalog) routing logic."""

from __future__ import annotations

from orchestration.structured_routing import collect_structured_routes, to_context_tool_output


def test_collect_structured_routes_matches_policy_keyword() -> None:
    routes = collect_structured_routes("What is your return policy?")

    assert routes == [{"type": "policy", "tool_name": "get_return_policy_info"}]


def test_collect_structured_routes_lightest_short_circuits() -> None:
    routes = collect_structured_routes(
        "What is the lightest running shoe under 100 euros? Also, what about returns?"
    )

    assert routes == [
        {"type": "policy", "tool_name": "get_return_policy_info"},
        {
            "type": "catalog",
            "tool_name": "get_lightest_product_under_price",
            "arguments": {"max_price_cents": 10000, "category": "running shoe"},
        },
    ]


def test_collect_structured_routes_matches_review_keyword() -> None:
    routes = collect_structured_routes("What do the reviews say about product 3?")

    assert routes == [
        {
            "type": "catalog",
            "tool_name": "get_product_reviews",
            "arguments": {"product_id": 3},
        }
    ]


def test_collect_structured_routes_returns_empty_for_unmatched_query() -> None:
    assert collect_structured_routes("Tell me something unexpected.") == []


def test_to_context_tool_output_normalizes_policy_dict() -> None:
    output = {
        "title": "Shipping Policy",
        "body": "Ships in 3-5 days.",
        "source_table": "policies",
        "source_id": 2,
    }

    normalized = to_context_tool_output("policy", "get_shipping_policy_info", output)

    assert normalized == {
        "source_type": "policy",
        "title": "Shipping Policy",
        "content": "Ships in 3-5 days.",
        "tool_name": "get_shipping_policy_info",
    }


def test_to_context_tool_output_normalizes_catalog_list() -> None:
    output = [{"id": 1, "name": "Starter Plan"}, {"id": 2, "name": "Growth Plan"}]

    normalized = to_context_tool_output("catalog", "filter_products", output)

    assert normalized["source_type"] == "product"
    assert normalized["title"] == "filter_products"
    assert "Starter Plan" in normalized["content"]
