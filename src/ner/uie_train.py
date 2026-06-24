from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.configuration.config import UIE_TRAINING_CONFIG


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="UIE fine-tuning entry for product information extraction.")
    parser.add_argument("--train-file", type=Path, default=Path("data/uie/train.jsonl"))
    parser.add_argument("--valid-file", type=Path, default=Path("data/uie/valid.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/uie"))
    parser.add_argument("--dry-run", action="store_true", help="Validate config and data format only.")
    args = parser.parse_args()

    print("UIE training config:")
    for key, value in UIE_TRAINING_CONFIG.items():
        print(f"- {key}: {value}")

    if args.train_file.exists():
        train_rows = load_jsonl(args.train_file)
        print(f"Loaded train examples: {len(train_rows)}")
    else:
        print(f"Train file not found: {args.train_file}")

    if args.valid_file.exists():
        valid_rows = load_jsonl(args.valid_file)
        print(f"Loaded valid examples: {len(valid_rows)}")
    else:
        print(f"Valid file not found: {args.valid_file}")

    if args.dry_run:
        return

    raise NotImplementedError(
        "UIE fine-tuning scaffold is configured. Add a task-specific trainer after the "
        "data schema is finalized; model weights should remain local under checkpoints/uie."
    )


if __name__ == "__main__":
    main()
