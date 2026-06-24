"""Fine-tune UIE-style entity extraction using StructBERT backbone.

Inspired by ModelScope UIE architecture but built on pure HuggingFace for reliability.
Uses the same StructBERT base (`damo/nlp_structbert_siamese-uie_chinese-base`) weights
loaded via HuggingFace, trained as token classification with SSI prompts.

Model: damo/nlp_structbert_siamese-uie_chinese-base (lazy: downloads on first run)
Task: Given text + "提取[实体类型]" prompt → extract entity spans (BIO tagging)
Schema: 商品名, 品牌, 品类, 属性, 卖点, 规格

On AutoDL:
    pip install -r requirements-uie.txt
    python -m src.ner.uie_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.configuration.config import CHECKPOINTS_DIR, ROOT_DIR
from src.configuration.config import UIE_TRAINING_CONFIG as CFG

# ERNIE 3.0 Base — Baidu's Chinese pre-trained model, same family as UIE backbone
MODEL_ID = "nghuyong/ernie-3.0-base-zh"
DATA_DIR = ROOT_DIR / "data" / "uie"
OUTPUT_DIR = CHECKPOINTS_DIR / "uie"
MAX_LENGTH = CFG.get("max_length", 256)

# BIO labels for entity extraction
LABEL_LIST = ["O", "B-卖点", "I-卖点"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def _load_jsonl(path: Path) -> list[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _bio_tokenize(sample: dict, tokenizer) -> dict:
    """Convert UIE record format to BIO token classification format."""
    text = sample["text"]
    record = sample["record"]  # e.g. "<卖点>韩国风味</卖点><卖点>双汇</卖点>"

    # Parse record to find entity spans in text
    import re
    chars = list(text)
    labels = ["O"] * len(chars)

    for m in re.finditer(r"<([^>]+)>([^<]*)</\1>", record):
        entity_text = m.group(2)
        if not entity_text:
            continue
        # Find entity_text in original text
        idx = text.find(entity_text)
        if idx >= 0:
            labels[idx] = "B-卖点"
            for j in range(idx + 1, idx + len(entity_text)):
                labels[j] = "I-卖点"

    # Tokenize with character-level alignment
    tokenized = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    word_ids = tokenized.word_ids()
    aligned_labels = []
    prev_wid = None
    for wid in word_ids:
        if wid is None:
            aligned_labels.append(-100)
        elif wid != prev_wid:
            aligned_labels.append(LABEL2ID[labels[wid]])
        else:
            aligned_labels.append(LABEL2ID[labels[wid]] if labels[wid].startswith("I") else -100)
        prev_wid = wid

    tokenized["labels"] = aligned_labels
    return tokenized


def train() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID, num_labels=len(LABEL_LIST))
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID, config=config).to(device)

    # Load & tokenize data
    train_raw = _load_jsonl(DATA_DIR / "train.jsonl")
    valid_raw = _load_jsonl(DATA_DIR / "valid.jsonl")
    print(f"Train: {len(train_raw)}, Valid: {len(valid_raw)}")

    train_ds = Dataset.from_list(train_raw).map(
        lambda x: _bio_tokenize(x, tokenizer), remove_columns=["text", "record"]
    )
    valid_ds = Dataset.from_list(valid_raw).map(
        lambda x: _bio_tokenize(x, tokenizer), remove_columns=["text", "record"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=CFG.get("epochs", 5),
        per_device_train_batch_size=CFG.get("batch_size", 8),
        per_device_eval_batch_size=CFG.get("batch_size", 8),
        learning_rate=CFG.get("learning_rate", 2e-5),
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=(device == "cuda"),
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    best_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Best model saved to {best_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UIE entity extraction model.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        print(f"Model: {MODEL_ID}")
        print(f"Data:  {DATA_DIR}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"Schema: {LABEL_LIST}")
        print("Dry-run OK.")
        return
    train()


if __name__ == "__main__":
    main()
