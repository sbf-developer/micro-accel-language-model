"""One-command entry: prepare data, train, and chat."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "checkpoints" / "accel-ft" / "best"
CORPUS = ROOT / "data" / "ace_corpus.jsonl"
PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}\n", flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def ensure_corpus() -> None:
    if not CORPUS.exists():
        run([PYTHON, str(ROOT / "data" / "generate_corpus.py")])


def ensure_model(force: bool = False) -> None:
    if force or not (CHECKPOINT / "config.json").exists():
        ensure_corpus()
        run([PYTHON, "-m", "accel.finetune_pretrained"])


def main() -> None:
    parser = argparse.ArgumentParser(description="ACCEL: train and chat")
    parser.add_argument("--train", action="store_true", help="Train (or retrain) the model")
    parser.add_argument("--eval", action="store_true", help="Run sample prompts and exit")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.train:
        ensure_model(force=True)
    else:
        ensure_model(force=False)

    if args.eval:
        run([PYTHON, str(ROOT / "eval_ft.py")])
        return

    run([PYTHON, str(ROOT / "chat_ft.py"), "--temperature", str(args.temperature)])


if __name__ == "__main__":
    main()
