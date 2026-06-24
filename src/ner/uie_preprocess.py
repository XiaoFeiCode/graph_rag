"""Convert Label Studio BIO annotations to UIE structured extraction format.

UIE format (per sample):
{
    "text": "original product text",
    "record": "<卖点>extracted entity</卖点><卖点>another</卖点>"
}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.configuration.config import DATA_DIR, ROOT_DIR
from src.configuration.config import UIE_TRAINING_CONFIG as CFG

OUTPUT_DIR = ROOT_DIR / "data" / "uie"
RANDOM_SEED = 42


def _read_label_studio(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_tag_spans(sample: dict) -> list[str]:
    """Extract all TAG-annotated text spans from a Label Studio sample."""
    spans = []
    for ann in sample.get("label", []):
        if "TAG" in ann.get("labels", []):
            spans.append(ann["text"])
    return spans


def _build_record(spans: list[str]) -> str:
    """Build UIE record string: <卖点>text</卖点> for each span."""
    return "".join(f"<卖点>{span}</卖点>" for span in spans)


def _convert(raw_path: Path) -> list[dict]:
    samples = _read_label_studio(raw_path)
    records = []
    for sample in samples:
        text = sample["text"]
        spans = _extract_tag_spans(sample)
        if not spans:
            continue
        records.append({"text": text, "record": _build_record(spans)})
    return records


def _split_save(records: list[dict], train_ratio: float = 0.8) -> None:
    random.seed(RANDOM_SEED)
    random.shuffle(records)
    n = len(records)
    train_n = int(n * train_ratio)
    valid_n = int(n * 0.1)

    train = records[:train_n]
    valid = records[train_n : train_n + valid_n]
    test = records[train_n + valid_n :]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        path = OUTPUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} samples → {path}")


def main():
    raw_path = DATA_DIR / "ner" / "raw" / "data.json"
    print(f"Reading {raw_path} ...")
    records = _convert(raw_path)
    print(f"Converted {len(records)} samples with TAG spans.")
    _split_save(records)
    print("Done.")


if __name__ == "__main__":
    main()
