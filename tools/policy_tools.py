"""SQLite-backed helpers for policy information lookups."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from data.loaders import connect_database

_POLICY_CODES: dict[str, tuple[str, ...]] = {
    "return": ("RETURNS", "RETURN"),
    "shipping": ("SHIPPING", "SHIP"),
    "warranty": ("WARRANTY", "WARRANTIES"),
}
_POLICY_TYPE_BY_CODE = {
    code: policy_type for policy_type, codes in _POLICY_CODES.items() for code in codes
}


def _lookup_policy(
    db_path: str | Path,
    requested_type: str,
) -> dict[str, str | int | bool]:
    """Look up a policy by preferred code and fall back to the best available row."""

    normalized_type = requested_type.lower()
    if normalized_type not in _POLICY_CODES:
        msg = f"Unsupported policy type: {requested_type}"
        raise ValueError(msg)

    connection = connect_database(db_path)
    connection.row_factory = sqlite3.Row
    try:
        row = None
        for code in _POLICY_CODES[normalized_type]:
            row = connection.execute(
                "SELECT id, code, title, body FROM policies WHERE UPPER(code) = ?",
                (code,),
            ).fetchone()
            if row is not None:
                break

        used_fallback = row is None
        if used_fallback:
            row = connection.execute(
                """
                SELECT id, code, title, body
                FROM policies
                ORDER BY
                    CASE
                        WHEN UPPER(code) = 'RETURNS' THEN 1
                        WHEN UPPER(code) = 'SHIPPING' THEN 2
                        WHEN UPPER(code) = 'WARRANTY' THEN 3
                        ELSE 10
                    END,
                    id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                msg = f"No policy information is available for {requested_type}."
                raise LookupError(msg)
    finally:
        connection.close()

    resolved_type = (
        _POLICY_TYPE_BY_CODE.get(row["code"].upper(), "return") if used_fallback else normalized_type
    )
    return {
        "requested_type": normalized_type,
        "resolved_type": resolved_type,
        "title": row["title"],
        "body": row["body"],
        "source_table": "policies",
        "source_id": row["id"],
        "source_code": row["code"],
        "used_fallback": used_fallback,
    }


def get_return_policy_info(db_path: str | Path) -> dict[str, str | int | bool]:
    """Return return policy information from SQLite."""

    return _lookup_policy(db_path, "return")


def get_shipping_policy_info(db_path: str | Path) -> dict[str, str | int | bool]:
    """Return shipping policy information from SQLite."""

    return _lookup_policy(db_path, "shipping")


def get_warranty_policy_info(db_path: str | Path) -> dict[str, str | int | bool]:
    """Return warranty policy information from SQLite."""

    return _lookup_policy(db_path, "warranty")
