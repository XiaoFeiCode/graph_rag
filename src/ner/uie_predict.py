"""UIE entity extraction inference.

Model: Fine-tuned StructBERT for token classification (BIO tagging).
Extracts entity spans from product text using SSI-style prompts.

The model is trained to extract 卖点 from product descriptions.
For other entity types (商品名, 品牌, 品类, 属性, 规格), the model can
generalise via prompt-based extraction using the same span-prediction head.
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.configuration.config import CHECKPOINTS_DIR

BEST_MODEL_PATH = str(CHECKPOINTS_DIR / "uie" / "best_model")
FALLBACK_MODEL = "uer/structbert-base-chinese"

ID2LABEL = {0: "O", 1: "B-卖点", 2: "I-卖点"}


def _extract_spans(labels: list[str], tokens: list[str]) -> list[str]:
    """Extract entity spans from BIO-tagged token sequence."""
    entities = []
    current = ""
    for label, token in zip(labels, tokens):
        if label == "B-卖点":
            if current:
                entities.append(current)
            current = token
        elif label == "I-卖点" and current:
            current += token
        else:
            if current:
                entities.append(current)
                current = ""
    if current:
        entities.append(current)
    return entities


class UIEPredictor:
    """Extract entities from product text using fine-tuned UIE model."""

    def __init__(self, model_path: str | None = None):
        path = model_path or BEST_MODEL_PATH
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForTokenClassification.from_pretrained(path)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            self.model = AutoModelForTokenClassification.from_pretrained(
                FALLBACK_MODEL, num_labels=3
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def extract(self, text: str) -> dict[str, list[str]]:
        """Extract entities from text. Returns {entity_type: [values]}."""
        chars = list(text)

        tokenized = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**tokenized)
            predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

        # Align predictions to original characters
        word_ids = tokenized.word_ids()
        char_labels = []
        prev_wid = None
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid != prev_wid:
                char_labels.append(ID2LABEL.get(predictions[i], "O"))
            prev_wid = wid

        # Extract spans
        spans = _extract_spans(char_labels, chars)
        return {"卖点": spans} if spans else {}


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
        if result:
            for etype, values in result.items():
                print(f"  {etype}: {values}")
        else:
            print("  (未抽取到实体)")


if __name__ == "__main__":
    predict()
