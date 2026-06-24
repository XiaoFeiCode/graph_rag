from __future__ import annotations

from src.configuration.config import UIE_MODEL_DIR, UIE_MODEL_ID


def main() -> None:
    from modelscope import snapshot_download

    model_dir = snapshot_download(UIE_MODEL_ID, local_dir=str(UIE_MODEL_DIR))
    print(f"Downloaded UIE model `{UIE_MODEL_ID}` to {model_dir}")
    print("Model weights are local runtime artifacts and are ignored by Git.")


if __name__ == "__main__":
    main()
