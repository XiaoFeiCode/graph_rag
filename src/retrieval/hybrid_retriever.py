"""Hybrid retrieval with explicit RRF fusion and optional BGE-Reranker re-rank."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RetrievalHit:
    entity_id: str
    name: str
    label: str
    score: float
    source: str = ""


def _rrf_fusion(result_sets: list[list[RetrievalHit]], k: int = 60) -> list[RetrievalHit]:
    """Reciprocal Rank Fusion across multiple ranked result lists."""
    scores: dict[str, float] = {}
    entities: dict[str, RetrievalHit] = {}

    for result_list in result_sets:
        for rank, hit in enumerate(result_list, start=1):
            key = f"{hit.label}:{hit.entity_id}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in entities:
                entities[key] = hit

    fused = []
    for key, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        hit = entities[key]
        hit.score = rrf_score
        fused.append(hit)
    return fused


class HybridRetriever:
    """Multi-source retriever with RRF fusion and optional BGE-Reranker re-rank."""

    def __init__(
        self,
        neo4j_fulltext_search: Callable | None = None,
        neo4j_vector_search: Callable | None = None,
        milvus_vector_search: Callable | None = None,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        self._sources: list[Callable] = []
        for src in [neo4j_fulltext_search, neo4j_vector_search, milvus_vector_search]:
            if src is not None:
                self._sources.append(src)

        self._reranker_model_name = reranker_model_name
        self._reranker = None

    @property
    def source_count(self) -> int:
        return len(self._sources)

    def _load_reranker(self):
        if self._reranker is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._reranker_tokenizer = AutoTokenizer.from_pretrained(self._reranker_model_name)
        self._reranker_model = AutoModelForSequenceClassification.from_pretrained(
            self._reranker_model_name
        )
        self._reranker_model.eval()

    def _rerank(self, query: str, candidates: list[RetrievalHit]) -> list[RetrievalHit]:
        if not candidates:
            return candidates

        self._load_reranker()
        import torch

        pairs = [[query, hit.name] for hit in candidates]
        inputs = self._reranker_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            scores = self._reranker_model(**inputs).logits.squeeze(-1).tolist()
        if isinstance(scores, float):
            scores = [scores]

        for hit, score in zip(candidates, scores):
            hit.score = float(score)
        candidates.sort(key=lambda h: h.score, reverse=True)
        return candidates

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        enable_rerank: bool = True,
    ) -> list[RetrievalHit]:
        """Run multi-source recall → RRF fusion → optional rerank → top-k."""
        if not self._sources:
            return []

        # Step 1: recall from each source
        result_sets: list[list[RetrievalHit]] = []
        for search_fn in self._sources:
            try:
                results = search_fn(query)
                if results:
                    result_sets.append(results)
            except Exception:
                continue

        if not result_sets:
            return []

        # Step 2: RRF fusion
        fused = _rrf_fusion(result_sets)

        # Step 3: BGE-Reranker re-rank
        if enable_rerank:
            fused = self._rerank(query, fused)

        return fused[:top_k]
