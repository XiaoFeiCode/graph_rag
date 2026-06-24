"""UIE inference: extract structured entities from product text.

Model: damo/nlp_structbert_siamese-uie_chinese-base (or fine-tuned checkpoint)
Schema: 商品名, 品牌, 品类, 属性, 卖点, 规格

The UIE model takes text + SSI prompt and outputs entity spans.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from src.configuration.config import CHECKPOINTS_DIR

BEST_MODEL_PATH = str(CHECKPOINTS_DIR / "uie" / "best_model")
FALLBACK_MODEL = "damo/nlp_structbert_siamese-uie_chinese-base"

SCHEMA = ["商品名", "品牌", "品类", "属性", "卖点", "规格"]


class UIEPredictor:
    """Extract product entities using a fine-tuned UIE model."""

    def __init__(self, model_path: str | None = None):
        path = model_path or BEST_MODEL_PATH

        # Try local checkpoint first, fall back to ModelScope hub
        import os

        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            self.pipe = pipeline(
                task=Tasks.information_extraction,
                model=path,
            )
        else:
            self.pipe = pipeline(
                task=Tasks.information_extraction,
                model=FALLBACK_MODEL,
            )

    def extract(self, text: str, entity_types: list[str] | None = None) -> dict[str, list[str]]:
        """Extract entities from text. Returns {entity_type: [values]}."""
        types = entity_types or SCHEMA
        results: dict[str, list[str]] = {}

        for entity_type in types:
            try:
                output = self.pipe(input=text, schema=[entity_type])
                if output and isinstance(output, list):
                    values = [
                        item["span"]
                        for item in output
                        if isinstance(item, dict) and "span" in item
                    ]
                    if values:
                        results[entity_type] = values
            except Exception:
                continue

        return results


def predict():
    """CLI demo."""
    predictor = UIEPredictor()

    texts = [
        "380克x3袋装韩国风味炒粘糕辣酱炒年糕条韩式部队火锅辣酱",
        "2018秋冬季新款韩版平底高帮鞋女休闲二棉鞋加绒运动厚底高邦鞋潮",
        "Apple iPhone 16 Pro 256GB 原色钛金属 5G 双卡双待",
    ]
    for text in texts:
        print(f"\n文本: {text}")
        result = predictor.extract(text)
        for etype, values in result.items():
            print(f"  {etype}: {values}")
        if not result:
            print("  (未抽取到实体)")


if __name__ == "__main__":
    predict()
