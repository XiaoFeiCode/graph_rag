"""Retrieval evaluation: compare fulltext, vector, hybrid (RRF), and RRF+Reranker."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable

from langchain_huggingface import HuggingFaceEmbeddings

from scripts.neo4j_client import neo4j_driver
from src.configuration.config import EMBEDDING_MODEL_NAME
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalHit
from src.retrieval.search_factory import (
    make_milvus_vector_searcher,
    make_neo4j_fulltext_searcher,
    make_neo4j_vector_searcher,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLES = ROOT_DIR / "data" / "questions.json"
DEFAULT_OUT_DIR = ROOT_DIR / "reports"

FULLTEXT_INDEXES = [
    "trademark_fulltext_index", "spu_fulltext_index", "sku_fulltext_index",
    "category1_fulltext_index", "category2_fulltext_index", "category3_fulltext_index",
    "tag_fulltext_index",
]
ALL_LABELS = ["Trademark", "SPU", "SKU", "Category1", "Category2", "Category3"]


def _normalize(text: str) -> str:
    return text.strip().lower()


def _load_examples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_fulltext_searcher(driver, top_k: int) -> Callable:
    def search(query: str) -> list[RetrievalHit]:
        hits: dict[str, RetrievalHit] = {}
        with driver.session() as session:
            for index_name in FULLTEXT_INDEXES:
                rows = session.run(
                    """
                    CALL db.index.fulltext.queryNodes($index_name, $query_text)
                    YIELD node, score
                    RETURN node.name AS name, labels(node)[0] AS label, score
                    ORDER BY score DESC LIMIT $top_k
                    """,
                    index_name=index_name, query_text=query, top_k=top_k,
                )
                for row in rows:
                    if not row["name"]:
                        continue
                    key = _normalize(row["name"])
                    current = hits.get(key)
                    candidate = RetrievalHit(
                        entity_id=str(row.get("entity_id", key)),
                        name=row["name"],
                        label=row["label"],
                        score=float(row["score"]),
                        source="neo4j_ft",
                    )
                    if current is None or candidate.score > current.score:
                        hits[key] = candidate
        return sorted(hits.values(), key=lambda h: h.score, reverse=True)[:top_k]
    return search


def _score_case(expected: list[str], hits: list[RetrievalHit]) -> dict:
    expected_set = {_normalize(item) for item in expected}
    hit_names = [_normalize(hit.name) for hit in hits]
    matched = expected_set & set(hit_names)

    rr = 0.0
    for idx, name in enumerate(hit_names, start=1):
        if name in expected_set:
            rr = 1.0 / idx
            break

    return {
        "expected_count": len(expected_set),
        "hit_count": len(hit_names),
        "matched_count": len(matched),
        "recall": len(matched) / len(expected_set) if expected_set else 0.0,
        "precision": len(matched) / len(hit_names) if hit_names else 0.0,
        "mrr": rr,
        "matched_entities": sorted(matched),
    }


def _eval_method(
    name: str,
    examples: list[dict],
    top_k: int,
    retriever_fn: Callable[[str | None, str | None], list[RetrievalHit] | None],
) -> dict:
    cases = []
    for example in examples:
        hits = retriever_fn(example["question"]) or []
        metrics = _score_case(example.get("expected_entities", []), hits)
        cases.append({
            "question": example["question"],
            "expected_entities": example.get("expected_entities", []),
            "hits": [{"name": h.name, "label": h.label, "score": h.score, "source": h.source} for h in hits],
            "metrics": metrics,
        })

    return {
        "method": name,
        "top_k": top_k,
        "case_count": len(cases),
        "metrics": {
            f"recall@{top_k}": mean(c["metrics"]["recall"] for c in cases) if cases else 0.0,
            f"precision@{top_k}": mean(c["metrics"]["precision"] for c in cases) if cases else 0.0,
            "mrr": mean(c["metrics"]["mrr"] for c in cases) if cases else 0.0,
        },
        "cases": cases,
    }


def evaluate_all(examples_path: Path, top_k: int) -> dict:
    examples = _load_examples(examples_path)
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    results = {}

    with neo4j_driver() as driver:
        ft_search = _build_fulltext_searcher(driver, top_k)

        # --- Method 1: fulltext baseline ---
        results["bm25_fulltext"] = _eval_method(
            "BM25 Fulltext", examples, top_k,
            lambda q: ft_search(q),
        )

        # --- Method 2: vector only ---
        vec_searchers = {
            label: make_neo4j_vector_searcher(driver, label, embedding, top_k)
            for label in ALL_LABELS
        }

        def vec_search(q):
            all_hits = []
            for s in vec_searchers.values():
                all_hits.extend(s(q))
            return sorted(all_hits, key=lambda h: h.score, reverse=True)[:top_k]

        results["neo4j_vector"] = _eval_method(
            "Neo4j Vector", examples, top_k, vec_search,
        )

        # --- Method 3: RRF (fulltext + vector) ---
        def rrf_search(q):
            ft_hits = ft_search(q)
            all_vec = []
            for s in vec_searchers.values():
                all_vec.extend(s(q))
            from src.retrieval.hybrid_retriever import _rrf_fusion
            return _rrf_fusion([ft_hits, all_vec])[:top_k]

        results["hybrid_rrf"] = _eval_method(
            "RRF (FT + Vector)", examples, top_k, rrf_search,
        )

        # --- Method 4: RRF + Reranker ---
        retriever_with_rerank = HybridRetriever(
            neo4j_fulltext_search=ft_search,
            neo4j_vector_search=lambda q: [h for s in vec_searchers.values() for h in s(q)],
        )

        def rrf_rerank_search(q):
            return retriever_with_rerank.retrieve(q, top_k=top_k, enable_rerank=True)

        results["hybrid_rrf_rerank"] = _eval_method(
            "RRF + BGE-Reranker", examples, top_k, rrf_rerank_search,
        )

    # --- Method 5: Milvus (if available) ---
    try:
        from src.retrieval.milvus_entity_store import MilvusEntityStore
        milvus = MilvusEntityStore()
        milvus.ensure_collection()
        mv_search = make_milvus_vector_searcher(milvus, embedding, top_k)
        results["milvus_vector"] = _eval_method(
            "Milvus Vector", examples, top_k,
            lambda q: mv_search(q),
        )
    except Exception:
        pass

    # Build comparison summary
    summary = {}
    for key, res in results.items():
        summary[key] = {
            f"recall@{top_k}": res["metrics"][f"recall@{top_k}"],
            f"precision@{top_k}": res["metrics"][f"precision@{top_k}"],
            "mrr": res["metrics"]["mrr"],
        }

    return {"top_k": top_k, "summary": summary, "details": results}


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    top_k = result["top_k"]
    summary = result["summary"]

    # JSON
    json_path = out_dir / "retrieval_eval.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown
    lines = [
        "# Retrieval Evaluation — Multi-Method Comparison",
        "",
        f"**Top-K:** {top_k}",
        "",
        "## Summary",
        "",
        "| Method | Recall | Precision | MRR |",
        "| --- | --- | --- | --- |",
    ]
    baseline_recall = summary.get("bm25_fulltext", {}).get(f"recall@{top_k}", 0.0)
    for name, metrics in summary.items():
        r = metrics[f"recall@{top_k}"]
        p = metrics[f"precision@{top_k}"]
        delta = ""
        if name != "bm25_fulltext" and baseline_recall > 0:
            gain = (r - baseline_recall) / baseline_recall * 100
            delta = f" (+{gain:.1f}%)"
        lines.append(
            f"| {name} | {_format_percent(r)}{delta} | {_format_percent(p)} | {metrics['mrr']:.4f} |"
        )

    # Per-case details
    for method_key, detail in result["details"].items():
        lines.append("")
        lines.append(f"## {detail['method']}")
        lines.append("")
        lines.append("| Question | Expected | Top Hits | Recall | Precision | MRR |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for case in detail["cases"]:
            cm = case["metrics"]
            expected = ", ".join(case["expected_entities"])
            hits = ", ".join(h["name"] for h in case["hits"]) or "-"
            lines.append(
                f"| {case['question']} | {expected} | {hits} | {_format_percent(cm['recall'])} | {_format_percent(cm['precision'])} | {cm['mrr']:.4f} |"
            )

    md_path = out_dir / "retrieval_eval.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    # Print summary
    print("\n=== Summary ===")
    for name, metrics in summary.items():
        r = metrics[f"recall@{top_k}"]
        p = metrics[f"precision@{top_k}"]
        print(f"  {name:25s}  Recall@{top_k}: {_format_percent(r)}  Precision@{top_k}: {_format_percent(p)}  MRR: {metrics['mrr']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-method retrieval evaluation.")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Starting retrieval evaluation...")
    result = evaluate_all(args.examples, args.top_k)
    write_report(result, args.out_dir)


if __name__ == "__main__":
    main()
