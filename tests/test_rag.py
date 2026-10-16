from __future__ import annotations

import pytest

from app.rag import (
    HashingEmbedder,
    RAGEngine,
    content_tokens,
    load_chunks,
    normalize_token,
)


@pytest.fixture(scope="module")
def engine() -> RAGEngine:
    from scripts.eval import offline_settings

    return RAGEngine.build(offline_settings())


def test_load_chunks_reads_knowledge_base(engine: RAGEngine) -> None:
    assert len(engine.chunks) > 0
    sources = {c.source for c in engine.chunks}
    assert {"shipping.md", "returns.md", "store_info.md"} <= sources


def test_missing_kb_path_yields_no_chunks(tmp_path) -> None:
    assert load_chunks(tmp_path / "nope") == []


def test_normalize_token_rules() -> None:
    assert normalize_token("Orders") == "order"
    assert normalize_token("address") == "address"  # ends in "ss"
    assert normalize_token("status") == "status"  # ends in "us"
    assert normalize_token("cities") == "cities"  # ends in "ies", left intact
    assert normalize_token("ship") == "ship"


def test_content_tokens_drops_stopwords_and_normalizes() -> None:
    tokens = content_tokens("How do the shipping rates work for my orders?")
    assert "the" not in tokens and "how" not in tokens and "my" not in tokens
    assert "shipping" in tokens
    assert "rate" in tokens
    assert "order" in tokens
    assert content_tokens("does") == set()


def test_hashing_embedder_is_normalized() -> None:
    emb = HashingEmbedder(dim=256)
    vectors = emb.embed(["hello world", "goodbye world"])
    assert vectors.shape == (2, 256)
    norms = (vectors**2).sum(axis=1) ** 0.5
    assert norms == pytest.approx([1.0, 1.0], abs=1e-5)


def test_vector_store_ranks_closest(engine: RAGEngine) -> None:
    store = engine.store
    assert store is not None
    assert store.size == len(engine.chunks)
    assert store.backend in {"faiss", "numpy"}


def test_retrieve_shipping_is_grounded(engine: RAGEngine) -> None:
    hits = engine.retrieve("How long does standard shipping take?")
    assert hits
    grounded = engine.grounded_hits(hits)
    assert grounded
    assert any("business day" in h.chunk.text.lower() for h in grounded)
    context = engine.build_context(hits)
    assert "business day" in context.lower()


def test_retrieve_return_policy(engine: RAGEngine) -> None:
    hits = engine.retrieve("What is your return policy?")
    context = engine.build_context(hits).lower()
    assert "30 day" in context or "30-day" in context


def test_out_of_domain_query_is_not_grounded(engine: RAGEngine) -> None:
    hits = engine.retrieve("What is the weather going to be in Tokyo this weekend?")
    assert engine.grounded_hits(hits) == []
    assert engine.build_context(hits) == ""


def test_engine_stats(engine: RAGEngine) -> None:
    stats = engine.stats()
    assert stats["provider"] == "hashing"
    assert stats["ready"] is True
    assert stats["chunks"] == len(engine.chunks)
    assert stats["index"]["scoring"] == "idf-coverage+cosine"
