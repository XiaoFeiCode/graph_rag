"""Search function factories for HybridRetriever.

Each factory returns a Callable[[str], list[RetrievalHit]] that can be plugged
into HybridRetriever as a recall source.
"""

from __future__ import annotations

from typing import Callable

from neo4j import Driver

from src.retrieval.hybrid_retriever import RetrievalHit

# Per-label index name mappings
_INDEXES: dict[str, tuple[str, str]] = {
    "Trademark": ("trademark_fulltext_index", "trademark_embedding_index"),
    "SPU":       ("spu_fulltext_index",       "spu_embedding_index"),
    "SKU":       ("sku_fulltext_index",       "sku_embedding_index"),
    "Category1": ("category1_fulltext_index", "category1_embedding_index"),
    "Category2": ("category2_fulltext_index", "category2_embedding_index"),
    "Category3": ("category3_fulltext_index", "category3_embedding_index"),
}


def make_neo4j_fulltext_searcher(driver: Driver, label: str, top_k: int = 10) -> Callable:
    """Return a search function that queries Neo4j fulltext index for a label."""
    indexes = _INDEXES.get(label)
    if not indexes:
        raise ValueError(f"No fulltext index configured for label: {label}")
    ft_index = indexes[0]

    def search(query_text: str) -> list[RetrievalHit]:
        records, _, _ = driver.execute_query(
            """
            CALL db.index.fulltext.queryNodes($index_name, $query_text)
            YIELD node, score
            RETURN node.id AS entity_id, node.name AS name, labels(node)[0] AS label, score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            index_name=ft_index,
            query_text=query_text,
            top_k=top_k,
        )
        return [
            RetrievalHit(
                entity_id=str(r["entity_id"]),
                name=r["name"],
                label=r["label"],
                score=float(r["score"]),
                source="neo4j_ft",
            )
            for r in records
            if r["name"]
        ]

    return search


def make_neo4j_vector_searcher(driver: Driver, label: str, embedding: Callable, top_k: int = 10) -> Callable:
    """Return a search function that does vector similarity via Neo4j vector index."""
    indexes = _INDEXES.get(label)
    if not indexes:
        raise ValueError(f"No vector index configured for label: {label}")
    emb_index = indexes[1]

    def search(query_text: str) -> list[RetrievalHit]:
        query_vector = embedding.embed_query(query_text)
        records, _, _ = driver.execute_query(
            f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
            YIELD node, score
            RETURN node.id AS entity_id, node.name AS name, labels(node)[0] AS label, score
            ORDER BY score DESC
            """,
            index_name=emb_index,
            query_vector=query_vector,
            top_k=top_k,
        )
        return [
            RetrievalHit(
                entity_id=str(r["entity_id"]),
                name=r["name"],
                label=r["label"],
                score=float(r["score"]),
                source="neo4j_vec",
            )
            for r in records
            if r["name"]
        ]

    return search


def make_milvus_vector_searcher(milvus_store, embedding: Callable, top_k: int = 10) -> Callable:
    """Return a search function that queries Milvus vector store."""

    def search(query_text: str) -> list[RetrievalHit]:
        query_vector = embedding.embed_query(query_text)
        hits = milvus_store.search(query_vector, top_k=top_k)
        return [
            RetrievalHit(
                entity_id=h["entity_id"],
                name=h["name"],
                label=h["label"],
                score=float(h["score"]),
                source="milvus",
            )
            for h in hits
        ]

    return search
