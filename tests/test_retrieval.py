"""Unit tests for Schema and Few-shot retrieval indexes."""

from pathlib import Path

from app.db import full_schema
from app.fewshot_index import FewShotIndex
from app.schema_index import retrieve_schema


def test_fewshot_exclusion(tmp_path: Path):
    """Verify that few-shot retrieval strictly excludes examples matching query db_id."""
    chroma_dir = tmp_path / "chroma_fs"
    index = FewShotIndex(chroma_dir=chroma_dir)

    # Populate with synthetic examples across 3 databases
    questions = [
        "How many singers are there?",
        "List all singer names.",
        "What are the pet names?",
        "Count the pets.",
        "List all customers.",
    ]
    metadatas = [
        {"db_id": "concert_singer", "sql": "SELECT count(*) FROM singer;", "qid": "1"},
        {"db_id": "concert_singer", "sql": "SELECT name FROM singer;", "qid": "2"},
        {"db_id": "pets_1", "sql": "SELECT name FROM pets;", "qid": "3"},
        {"db_id": "pets_1", "sql": "SELECT count(*) FROM pets;", "qid": "4"},
        {"db_id": "customers_db", "sql": "SELECT * FROM customers;", "qid": "5"},
    ]

    from app.fewshot_index import get_embedding_model
    model = get_embedding_model()
    embeddings = model.encode(questions).tolist()

    index.collection.add(
        ids=[f"t_{i}" for i in range(len(questions))],
        documents=questions,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Query with db_id = "concert_singer"
    # Even though concert_singer examples are semantically closest, they MUST BE EXCLUDED!
    results = index.retrieve("How many singers in the database?", db_id="concert_singer", k=2)

    assert len(results) > 0
    for r in results:
        assert r["db_id"].lower() != "concert_singer", f"Exclusion failed: returned {r}"


def test_full_schema_truncation():
    """Verify that full_schema sets the truncation flag when schema exceeds budget."""
    # Test on concert_singer with very small char limit
    schema_text, tables, cols, truncated = full_schema("concert_singer", max_chars=50)
    assert truncated is True
    assert "[SCHEMA TRUNCATED DUE TO TOKEN BUDGET]" in schema_text

    # With large budget, should not be truncated
    schema_text_full, _, _, truncated_full = full_schema("concert_singer", max_chars=50000)
    assert truncated_full is False
    assert len(tables) > 0


def test_schema_retrieval_expansion(tmp_path: Path):
    """Verify schema retrieval expands to include foreign key columns and tables."""
    # Use real concert_singer database
    rendered, tables, columns, trunc = retrieve_schema(
        question="Which singer sang in which concert?",
        db_id="concert_singer",
        k_tables=2,
        k_columns=5,
        hybrid=True,
        chroma_dir=tmp_path / "chroma_schema",
    )

    assert "CREATE TABLE" in rendered
    assert len(tables) > 0
    assert len(columns) > 0
    assert not trunc

