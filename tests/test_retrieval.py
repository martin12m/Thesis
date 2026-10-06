"""Deterministic tests for sparse and hybrid (sparse + dense) retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval.indexer import rebuild_index
from retrieval.sparse import pack_vector, search_hybrid, search_index, unpack_vector
from scripts.seed_data import seed_from_files


def _seed_plans_db(db_path: Path, *, embedder: Any = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    seed_from_files(
        db_path,
        products_path=project_root / "data" / "products.json",
        policies_path=project_root / "data" / "policies.json",
    )
    rebuild_index(db_path, embedder=embedder)


def _fake_embedder(texts: list[str]) -> list[list[float]]:
    """Deterministic stand-in for the real ONNX model, keyed on known content."""

    vectors: list[list[float]] = []
    for text in texts:
        if "Growth Plan" in text or "analytics" in text.lower():
            vectors.append([1.0, 0.0, 0.0])
        elif "Starter Plan" in text:
            vectors.append([0.0, 1.0, 0.0])
        elif "returns are accepted" in text.lower():
            vectors.append([0.0, 0.0, 1.0])
        elif "privacy" in text.lower():
            vectors.append([0.0, 0.0, 0.0])
        elif "premium tier" in text.lower() or "reporting dashboards" in text.lower():
            vectors.append([0.9, 0.1, 0.0])
        else:
            vectors.append([0.0, 0.0, 0.0])
    return vectors


def _raising_embedder(texts: list[str]) -> list[list[float]]:
    raise RuntimeError("embedding backend unavailable")


def test_sparse_retrieval_returns_growth_plan_for_analytics_query(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.db"
    project_root = Path(__file__).resolve().parents[1]
    seed_from_files(
        db_path,
        products_path=project_root / "data" / "products.json",
        policies_path=project_root / "data" / "policies.json",
    )
    rebuild_index(db_path)

    results = search_index(db_path, "analytics features", top_k=2)

    assert len(results) == 1
    assert results[0]["document_id"] == "product:2"
    assert results[0]["source_type"] == "product"
    assert results[0]["source_record_id"] == 2
    assert results[0]["title"] == "Growth Plan"


def test_sparse_retrieval_returns_returns_policy_for_returns_query(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.db"
    project_root = Path(__file__).resolve().parents[1]
    seed_from_files(
        db_path,
        products_path=project_root / "data" / "products.json",
        policies_path=project_root / "data" / "policies.json",
    )
    rebuild_index(db_path)

    results = search_index(db_path, "returns 30 days", top_k=2)

    assert len(results) >= 1
    assert results[0]["document_id"] == "policy:1"
    assert results[0]["source_type"] == "policy"
    assert results[0]["source_record_id"] == 1
    assert results[0]["title"] == "Returns Policy"


def test_sparse_retrieval_handles_natural_language_comparison_query(tmp_path: Path) -> None:
    """Full-sentence queries still return results."""

    db_path = tmp_path / "retrieval.db"
    project_root = Path(__file__).resolve().parents[1]
    seed_from_files(
        db_path,
        products_path=project_root / "data" / "products.json",
        policies_path=project_root / "data" / "policies.json",
    )
    rebuild_index(db_path)

    results = search_index(
        db_path,
        "Which plan is better, the Starter Plan or the Growth Plan?",
        top_k=5,
    )

    document_ids = {item["document_id"] for item in results}
    assert document_ids == {"product:1", "product:2"}


def test_sparse_retrieval_ranks_more_matching_terms_higher(tmp_path: Path) -> None:
    """OR-based matching still lets BM25 rank multi-term overlap above one term."""

    db_path = tmp_path / "retrieval.db"
    project_root = Path(__file__).resolve().parents[1]
    seed_from_files(
        db_path,
        products_path=project_root / "data" / "products.json",
        policies_path=project_root / "data" / "policies.json",
    )
    rebuild_index(db_path)

    results = search_index(db_path, "analytics features", top_k=5)

    assert results[0]["document_id"] == "product:2"


def test_pack_and_unpack_vector_round_trip() -> None:
    vector = [0.125, -0.25, 0.375, 1.0]

    assert unpack_vector(pack_vector(vector)) == vector


def test_search_hybrid_finds_semantic_match_with_zero_keyword_overlap(tmp_path: Path) -> None:
    """The core capability gain over BM25-only: no shared vocabulary, still retrieved."""

    db_path = tmp_path / "retrieval.db"
    _seed_plans_db(db_path, embedder=_fake_embedder)

    query = "premium tier upgrade with reporting dashboards"
    assert search_index(db_path, query, top_k=5) == []

    results = search_hybrid(db_path, query, top_k=1, embedder=_fake_embedder)

    assert len(results) == 1
    assert results[0]["document_id"] == "product:2"


def test_search_hybrid_still_finds_exact_keyword_matches(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.db"
    _seed_plans_db(db_path, embedder=_fake_embedder)

    results = search_hybrid(db_path, "returns policy", top_k=1, embedder=_fake_embedder)

    assert results[0]["document_id"] == "policy:1"


def test_search_hybrid_degrades_to_bm25_ranking_without_embeddings(tmp_path: Path) -> None:
    """If the embedding backend was unavailable at index time, hybrid search must still work -- falling back to the same ranking search_index gives."""

    db_path = tmp_path / "retrieval.db"
    _seed_plans_db(db_path, embedder=_raising_embedder)

    sparse_results = search_index(db_path, "returns policy", top_k=5)
    hybrid_results = search_hybrid(db_path, "returns policy", top_k=5, embedder=_raising_embedder)

    assert [item["document_id"] for item in hybrid_results] == [
        item["document_id"] for item in sparse_results
    ]


def test_search_hybrid_returns_empty_for_completely_unmatched_query(tmp_path: Path) -> None:
    db_path = tmp_path / "retrieval.db"
    _seed_plans_db(db_path, embedder=_raising_embedder)

    assert search_hybrid(db_path, "completely unrelated gibberish query", embedder=_raising_embedder) == []


def test_search_hybrid_grounds_cross_domain_price_comparison_from_external_reviews_data(
    tmp_path: Path,
) -> None:
    """The core 'not hardcoded to any one domain' claim: ingest whatever a Trustami-shaped external API returns -- a real captured radio response plus mock TV/vacuum responses, three unrelated product categories -- and retrieval must surface the right grounding for a price-comparison question about products from *different* domains, with no special-case code anywhere for "radio" or "vacuum" specifically."""

    from data.loaders import connect_database, seed_database
    from data.product_reviews_ingest import (
        external_reviews_products_from_json,
        load_external_reviews_json,
    )

    fixtures_dir = Path(__file__).resolve().parents[1] / "data" / "fixtures"
    radio_doc = dict(
        load_external_reviews_json(fixtures_dir / "external_reviews_real_example_radio.json")
    )
    radio_doc["sku"] = "TEUFEL-RADIO-3SIXTY"  # the real capture has no top-level identifier
    tv_docs = load_external_reviews_json(fixtures_dir / "external_reviews_mock_tvs.json")
    vacuum_docs = load_external_reviews_json(fixtures_dir / "external_reviews_mock_vacuums.json")

    products = external_reviews_products_from_json([radio_doc, *tv_docs, *vacuum_docs])
    assert len(products) == 5  # 1 radio + 2 TVs + 2 vacuums: three unrelated domains

    db_path = tmp_path / "cross_domain.db"
    connection = connect_database(db_path)
    seed_database(connection, products=products, policies=[])
    connection.close()
    rebuild_index(db_path)

    results = search_hybrid(
        db_path, "Which is cheaper, the Dyson vacuum or the Teufel radio?", top_k=5
    )

    radio_item = next(item for item in results if "Teufel" in item["title"])
    dyson_item = next(item for item in results if "Dyson" in item["title"])
    assert "Price 28499 cents" in radio_item["content"]
    assert "Price 74900 cents" in dyson_item["content"]
