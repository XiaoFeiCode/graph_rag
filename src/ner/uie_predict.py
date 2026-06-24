"""UIE inference: extract structured entities from product text.

Supports 6 entity types matching the UIE schema:
    商品名, 品牌, 品类, 属性, 卖点, 规格

For entity types beyond 卖点 (which had training data), the model uses
zero-shot structure extraction via prompt engineering.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.configuration.config import CHECKPOINTS_DIR

BEST_MODEL_PATH = CHECKPOINTS_DIR / "uie" / "best_model"
FALLBACK_MODEL = "fnlp/bart-base-chinese"

# UIE schema: entity types to extract
SCHEMA = ["商品名", "品牌", "品类", "属性", "卖点", "规格"]

# Prompts per entity type (for zero-shot extraction)
_EXTRACT_PROMPTS = {
    "商品名": "提取商品名称: {text}",
    "品牌":   "提取品牌名称: {text}",
    "品类":   "提取商品品类: {text}",
    "属性":   "提取商品属性: {text}",
    "卖点":   "提取卖点: {text}",
    "规格":   "提取规格参数: {text}",
}


def _parse_record(text: str) -> list[str]:
    """Parse UIE record like '<卖点>text</卖点><卖点>other</卖点>' into list."""
    results = []
    for match in re.finditer(r"<([^>]+)>([^<]*)</\1>", text):
        results.append(match.group(2).strip())
    return [r for r in results if r]


def _clean_output(text: str) -> str:
    """Remove special tokens from model output."""
    text = text.replace("</s>", "").replace("<s>", "").replace("<pad>", "").strip()
    return text


class UIEPredictor:
    """Extract product entities using fine-tuned UIE model."""

    def __init__(self, model_path: str | None = None):
        path = model_path or str(BEST_MODEL_PATH)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_MODEL)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def extract(self, text: str, entity_types: list[str] | None = None) -> dict[str, list[str]]:
        """Extract entities from product text. Returns {entity_type: [values]}."""
        types = entity_types or SCHEMA
        results: dict[str, list[str]] = {}

        for entity_type in types:
            prompt = _EXTRACT_PROMPTS.get(entity_type, f"提取{entity_type}: {{text}}")
            source = prompt.format(text=text)

            inputs = self.tokenizer(
                source, return_tensors="pt", truncation=True, max_length=256
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=3,
                    early_stopping=True,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            decoded = _clean_output(decoded)
            extracted = _parse_record(decoded)
            if extracted:
                results[entity_type] = extracted

        return results

    def extract_all(self, text: str) -> dict[str, Any]:
        """Extract all schema entities and return flat dict."""
        entities = self.extract(text, SCHEMA)
        flat: dict[str, Any] = {"text": text}
        for etype in SCHEMA:
            flat[etype] = entities.get(etype, [])
        return flat


def predict():
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


if __name__ == "__main__":
    predict()
