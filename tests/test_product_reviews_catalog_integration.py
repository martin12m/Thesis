"""End-to-end tests: external-reviews JSON -> seeded catalog -> retrieval/tools."""

from __future__ import annotations

from pathlib import Path

from data.loaders import connect_database, seed_database
from data.product_reviews_ingest import external_reviews_products_from_json, load_external_reviews_json
from retrieval.indexer import rebuild_index
from retrieval.sparse import search_index
from tools.catalog_tools import get_product_reviews

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "fixtures" / "external_reviews_mock_tvs.json"
)


def _seed_tv_db(db_path: Path) -> dict[str, int]:
    products = external_reviews_products_from_json(load_external_reviews_json(FIXTURE_PATH))
    connection = connect_database(db_path)
    try:
        return seed_database(connection, products=products, policies=[])
    finally:
        connection.close()


def test_seed_database_persists_external_reviews_brand_and_rating_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "tv_catalog.db"
    summary = _seed_tv_db(db_path)

    assert summary["products"] == 2
    assert summary["product_reviews"] == 4  # 2 reviews per TV x 2 TVs

    connection = connect_database(db_path)
    try:
        row = connection.execute(
            "SELECT brand, rating_stars, rating_count, rating_grade FROM products WHERE sku = ?",
            ("TV-SAMSUNG-OLED-65",),
        ).fetchone()
    finally:
        connection.close()

    assert row == ("Samsung", 4.5, 842, "Sehr gut")


def test_get_product_reviews_reads_locally_ingested_external_reviews(tmp_path: Path) -> None:
    db_path = tmp_path / "tv_catalog.db"
    _seed_tv_db(db_path)

    connection = connect_database(db_path)
    try:
        samsung_id = connection.execute(
            "SELECT id FROM products WHERE sku = ?", ("TV-SAMSUNG-OLED-65",)
        ).fetchone()[0]
    finally:
        connection.close()

    result = get_product_reviews(db_path, samsung_id, env={"COMMERCE_PRODUCT_SOURCE": "local"})

    assert result["source"] == "local_fixture"
    assert len(result["reviews"]) == 2
    assert result["score"] == {
        "rating_stars": 4.5,
        "rating_count": 842,
        "rating_grade": "Sehr gut",
    }
    review_texts = {review["text"] for review in result["reviews"]}
    assert "Incredible black levels and the AI upscaling makes older content look fantastic." in review_texts


def test_get_product_reviews_returns_empty_for_product_with_no_reviews(tmp_path: Path) -> None:
    db_path = tmp_path / "plain_catalog.db"
    connection = connect_database(db_path)
    try:
        seed_database(
            connection,
            products=[
                {
                    "sku": "SKU-PLAIN",
                    "name": "Plain Product",
                    "description": "No reviews here.",
                    "price_cents": 500,
                }
            ],
            policies=[],
        )
    finally:
        connection.close()

    result = get_product_reviews(db_path, 1, env={"COMMERCE_PRODUCT_SOURCE": "local"})

    assert result["reviews"] == []
    assert result["score"] is None


def test_retrieval_finds_brand_and_review_evidence_for_comparison_query(tmp_path: Path) -> None:
    db_path = tmp_path / "tv_catalog.db"
    _seed_tv_db(db_path)
    rebuild_index(db_path)

    results = search_index(
        db_path,
        "Which one is better, the Samsung OLED or the LG QLED TV 65 inch?",
        top_k=5,
    )

    titles = {item["title"] for item in results}
    assert "Samsung OLED 65\" S90D Smart TV" in titles
    assert "LG QNED 65\" QLED Smart TV" in titles


def test_retrieval_surfaces_review_text_as_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "tv_catalog.db"
    _seed_tv_db(db_path)
    rebuild_index(db_path)

    results = search_index(db_path, "black levels AI upscaling", top_k=3)

    assert results
    assert results[0]["title"] == "Samsung OLED 65\" S90D Smart TV"


def test_retrieval_surfaces_brand_as_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "tv_catalog.db"
    _seed_tv_db(db_path)
    rebuild_index(db_path)

    results = search_index(db_path, "Samsung", top_k=3)

    assert results
    assert results[0]["title"] == "Samsung OLED 65\" S90D Smart TV"
