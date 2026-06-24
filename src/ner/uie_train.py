"""Fine-tune the ModelScope UIE model for product entity extraction.

Model: damo/nlp_structbert_siamese-uie_chinese-base (StructBERT + Siamese UIE)
Task: Structured extraction — text → <entity_type>span</entity_type>
Schema: 商品名, 品牌, 品类, 属性, 卖点, 规格

Data format (JSONL): {"text": "...", "record": "<卖点>text</卖点><商品名>text</商品名>"}

Usage on AutoDL:
    conda create -n uie python=3.12 -y && conda activate uie
    pip install -r requirements-uie.txt
    python -m src.ner.uie_train
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config

from src.configuration.config import CHECKPOINTS_DIR, ROOT_DIR
from src.configuration.config import UIE_TRAINING_CONFIG as CFG

MODEL_ID = "damo/nlp_structbert_siamese-uie_chinese-base"
DATA_DIR = ROOT_DIR / "data" / "uie"
OUTPUT_DIR = CHECKPOINTS_DIR / "uie"


def _load_jsonl(path: Path) -> list[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _save_jsonl(data: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_dataset_jsonl(records: list[dict], output_dir: Path, filename: str = "dataset.jsonl") -> Path:
    """Write records to a JSONL file that ModelScope MsDataset can read."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    _save_jsonl(records, path)
    return path


def train() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")

    train_data = _load_jsonl(DATA_DIR / "train.jsonl")
    valid_data = _load_jsonl(DATA_DIR / "valid.jsonl")
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Stage data for ModelScope MsDataset
    tmpdir = Path(tempfile.mkdtemp(prefix="uie_data_"))
    train_path = _write_dataset_jsonl(train_data, tmpdir / "train")
    valid_path = _write_dataset_jsonl(valid_data, tmpdir / "validation")

    # Load as MsDataset
    dataset_dict = MsDataset.load(
        str(tmpdir),
        split=["train", "validation"],
    )

    # Build trainer with the SiameseUIETrainer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "model": MODEL_ID,
        "train_dataset": dataset_dict["train"],
        "eval_dataset": dataset_dict["validation"],
        "work_dir": str(OUTPUT_DIR),
        "cfg_modify_fn": _cfg_modify_fn,
    }

    trainer = build_trainer(
        name="siamese-uie-trainer",
        default_args=kwargs,
    )

    print("Starting training...")
    trainer.train()

    print(f"Training complete. Output: {OUTPUT_DIR}")


def _cfg_modify_fn(cfg: Config) -> Config:
    cfg.train.max_epochs = CFG.get("epochs", 5)
    cfg.train.batch_size_per_gpu = CFG.get("batch_size", 8)
    cfg.train.optimizer.lr = CFG.get("learning_rate", 2e-5)
    cfg.preprocessor.max_length = CFG.get("max_length", 256)
    return cfg


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

    train()


if __name__ == "__main__":
    main()
