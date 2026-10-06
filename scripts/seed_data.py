"""Seed the local SQLite database from JSON fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.loaders import connect_database, load_json_records, seed_database


def seed_from_files(
    db_path: str | Path,
    *,
    products_path: str | Path,
    policies_path: str | Path,
) -> dict[str, int]:
    """Load JSON files and seed the database."""

    products = load_json_records(products_path)
    policies = load_json_records(policies_path)
    connection = connect_database(db_path)
    try:
        return seed_database(connection, products=products, policies=policies)
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the seed script."""

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Seed SQLite data from JSON files.")
    parser.add_argument("--db", default=project_root / "app.db", type=Path)
    parser.add_argument(
        "--products",
        default=project_root / "data" / "products.json",
        type=Path,
    )
    parser.add_argument(
        "--policies",
        default=project_root / "data" / "policies.json",
        type=Path,
    )
    return parser


def main() -> int:
    """Run the seed script from the command line."""

    args = build_parser().parse_args()
    summary = seed_from_files(
        args.db,
        products_path=args.products,
        policies_path=args.policies,
    )
    print(
        "Seeded database with "
        f"{summary['products']} products, "
        f"{summary['policies']} policies, "
        f"and {summary['product_policies']} policy links."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
