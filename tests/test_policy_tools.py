"""Deterministic tests for the SQLite policy tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.loaders import connect_database, initialize_schema
from tools.policy_tools import (
    get_return_policy_info,
    get_shipping_policy_info,
    get_warranty_policy_info,
)


@pytest.fixture
def policy_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with policy rows for testing."""

    db_path = tmp_path / "policies.db"
    connection = connect_database(db_path)
    try:
        initialize_schema(connection)
        with connection:
            connection.executemany(
                """
                INSERT INTO policies (code, title, body)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        "RETURNS",
                        "Returns Policy",
                        "Returns are accepted within 30 days.",
                    ),
                    (
                        "SHIPPING",
                        "Shipping Policy",
                        "Standard shipping is available in 3 to 5 business days.",
                    ),
                ],
            )
    finally:
        connection.close()

    return db_path


def test_get_return_policy_info_uses_exact_match(policy_db: Path) -> None:
    result = get_return_policy_info(policy_db)

    assert result == {
        "requested_type": "return",
        "resolved_type": "return",
        "title": "Returns Policy",
        "body": "Returns are accepted within 30 days.",
        "source_table": "policies",
        "source_id": 1,
        "source_code": "RETURNS",
        "used_fallback": False,
    }


def test_get_shipping_policy_info_uses_exact_match(policy_db: Path) -> None:
    result = get_shipping_policy_info(policy_db)

    assert result == {
        "requested_type": "shipping",
        "resolved_type": "shipping",
        "title": "Shipping Policy",
        "body": "Standard shipping is available in 3 to 5 business days.",
        "source_table": "policies",
        "source_id": 2,
        "source_code": "SHIPPING",
        "used_fallback": False,
    }


def test_get_warranty_policy_info_falls_back_to_available_policy(policy_db: Path) -> None:
    result = get_warranty_policy_info(policy_db)

    assert result == {
        "requested_type": "warranty",
        "resolved_type": "return",
        "title": "Returns Policy",
        "body": "Returns are accepted within 30 days.",
        "source_table": "policies",
        "source_id": 1,
        "source_code": "RETURNS",
        "used_fallback": True,
    }
