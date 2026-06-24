from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLES = ROOT_DIR / "data" / "questions.json"
DEFAULT_OUT_DIR = ROOT_DIR / "reports"

FULLTEXT_INDEXES = [
    "trademark_fulltext_index",
    "spu_fulltext_index",
    "sku_fulltext_index",
    "category1_fulltext_index",
    "category2_fulltext_index",
    "category3_fulltext_index",
    "tag_fulltext_index",
]


@dataclass
class RetrievalHit:
    name: str
    label: str
    score: float


def _normalize(text: str) -> str:
    return text.strip().lower()


def _load_examples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _query_fulltext(session, question: str, top_k: int) -> list[RetrievalHit]:
    hits: dict[str, RetrievalHit] = {}
    for index_name in FULLTEXT_INDEXES:
        rows = session.run(
            """
            CALL db.index.fulltext.queryNodes($index_name, $query_text)
            YIELD node, score
            RETURN node.name AS name, labels(node)[0] AS label, score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            index_name=index_name,
            query_text=question,
            top_k=top_k,
        )
        for row in rows:
            if not row["name"]:
                continue
            key = _normalize(row["name"])
            current = hits.get(key)
            candidate = RetrievalHit(
                name=row["name"],
                label=row["label"],
                score=float(row["score"]),
            )
            if current is None or candidate.score > current.score:
                hits[key] = candidate

    return sorted(hits.values(), key=lambda item: item.score, reverse=True)[:top_k]


def _score_case(expected: list[str], hits: list[RetrievalHit]) -> dict:
    expected_set = {_normalize(item) for item in expected}
    hit_names = [_normalize(hit.name) for hit in hits]
    matched = expected_set & set(hit_names)

    reciprocal_rank = 0.0
    for index, name in enumerate(hit_names, start=1):
        if name in expected_set:
            reciprocal_rank = 1.0 / index
            break

    return {
        "expected_count": len(expected_set),
        "hit_count": len(hit_names),
        "matched_count": len(matched),
        "recall": len(matched) / len(expected_set) if expected_set else 0.0,
        "precision": len(matched) / len(hit_names) if hit_names else 0.0,
        "mrr": reciprocal_rank,
        "matched_entities": sorted(matched),
    }


def evaluate(examples_path: Path, top_k: int) -> dict:
    from scripts.neo4j_client import neo4j_driver

    examples = _load_examples(examples_path)
    cases = []

    with neo4j_driver() as driver:
        with driver.session() as session:
            for example in examples:
                hits = _query_fulltext(session, example["question"], top_k)
                metrics = _score_case(example.get("expected_entities", []), hits)
                cases.append(
                    {
                        "question": example["question"],
                        "expected_entities": example.get("expected_entities", []),
                        "hits": [hit.__dict__ for hit in hits],
                        "metrics": metrics,
                    }
                )

    return {
        "method": "neo4j_fulltext",
        "top_k": top_k,
        "case_count": len(cases),
        "metrics": {
            f"recall@{top_k}": mean(case["metrics"]["recall"] for case in cases) if cases else 0.0,
            f"precision@{top_k}": mean(case["metrics"]["precision"] for case in cases) if cases else 0.0,
            "mrr": mean(case["metrics"]["mrr"] for case in cases) if cases else 0.0,
        },
        "cases": cases,
    }


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "retrieval_eval.json"
    md_path = out_dir / "retrieval_eval.md"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    top_k = result["top_k"]
    metrics = result["metrics"]
    lines = [
        "# Retrieval Evaluation",
        "",
        f"- Method: `{result['method']}`",
        f"- Cases: `{result['case_count']}`",
        f"- Recall@{top_k}: `{_format_percent(metrics[f'recall@{top_k}'])}`",
        f"- Precision@{top_k}: `{_format_percent(metrics[f'precision@{top_k}'])}`",
        f"- MRR: `{metrics['mrr']:.4f}`",
        "",
        "| Question | Expected | Top Hits | Recall | Precision | MRR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for case in result["cases"]:
        case_metrics = case["metrics"]
        expected = ", ".join(case["expected_entities"])
        hits = ", ".join(hit["name"] for hit in case["hits"]) or "-"
        lines.append(
            "| {question} | {expected} | {hits} | {recall} | {precision} | {mrr:.4f} |".format(
                question=case["question"],
                expected=expected,
                hits=hits,
                recall=_format_percent(case_metrics["recall"]),
                precision=_format_percent(case_metrics["precision"]),
                mrr=case_metrics["mrr"],
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate entity retrieval on example questions.")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args.examples, args.top_k)
    write_report(result, args.out_dir)


if __name__ == "__main__":
    main()
