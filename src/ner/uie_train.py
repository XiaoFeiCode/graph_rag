"""Fine-tune a Chinese BART model for UIE-style structure extraction.

Model: fnlp/bart-base-chinese (encoder-decoder, suitable for generation)
Task: given product text, generate structured entity records like "<卖点>text</卖点>"

After training, this model can extract: 商品名, 品牌, 品类, 属性, 卖点, 规格
from product descriptions — matching the UIE schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.configuration.config import CHECKPOINTS_DIR, ROOT_DIR
from src.configuration.config import UIE_TRAINING_CONFIG as CFG

MODEL_ID = "fnlp/bart-base-chinese"
DATA_DIR = ROOT_DIR / "data" / "uie"
OUTPUT_DIR = CHECKPOINTS_DIR / "uie"
MAX_LENGTH = CFG.get("max_length", 256)


class UIEDataset(Dataset):
    """Dataset for UIE structure extraction: text → record."""

    def __init__(self, path: Path, tokenizer, max_length: int):
        self.samples: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        record = sample["record"]

        source = f"提取卖点: {text}"

        tokenized = self.tokenizer(
            source,
            text_target=record,
            max_length=self.max_length,
            truncation=True,
        )
        return tokenized  # return raw dict, trainer handles tensor conversion


def _load_data(tokenizer) -> tuple[UIEDataset, UIEDataset, UIEDataset]:
    train_path = DATA_DIR / "train.jsonl"
    valid_path = DATA_DIR / "valid.jsonl"
    test_path = DATA_DIR / "test.jsonl"

    for name, p in [("train", train_path), ("valid", valid_path), ("test", test_path)]:
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run uie_preprocess.py first.")

    return (
        UIEDataset(train_path, tokenizer, MAX_LENGTH),
        UIEDataset(valid_path, tokenizer, MAX_LENGTH),
        UIEDataset(test_path, tokenizer, MAX_LENGTH),
    )


def train() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)

    train_ds, valid_ds, test_ds = _load_data(tokenizer)
    print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=CFG.get("epochs", 5),
        per_device_train_batch_size=CFG.get("batch_size", 8),
        per_device_eval_batch_size=CFG.get("batch_size", 8),
        learning_rate=CFG.get("learning_rate", 2e-5),
        warmup_ratio=CFG.get("warmup_ratio", 0.1),
        weight_decay=CFG.get("weight_decay", 0.01),
        fp16=(device == "cuda"),
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    # Save best model
    best_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Best model saved to {best_dir}")

    # Evaluate on test set
    metrics = trainer.evaluate(test_ds)
    print(f"Test metrics: {metrics}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UIE structure extraction model.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        print(f"Model: {MODEL_ID}")
        print(f"Data:  {DATA_DIR}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"Config: {CFG}")
        print("Dry-run OK.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train()


if __name__ == "__main__":
    main()
