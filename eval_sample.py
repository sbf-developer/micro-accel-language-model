"""Quick non-interactive generation smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from accel.model import MicroGPT, ModelConfig
from accel.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/accel-v1"))
    parser.add_argument("--weights", type=str, default="best_final", choices=["best", "best_final", "last"])
    args = parser.parse_args()

    device = torch.device("cpu")
    tok = Tokenizer.load(args.checkpoint / "tokenizer.json")
    ckpt_path = args.checkpoint / f"{args.weights}.pt"
    if not ckpt_path.exists():
        ckpt_path = args.checkpoint / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MicroGPT(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompts = [
        "<|user|>Hi<|assistant|>",
        "<|user|>What is a variable?<|assistant|>",
        "<|user|>I'm stressed about exams<|assistant|>",
        "<|user|>What's 24 divided by 6?<|assistant|>",
    ]

    print(f"Checkpoint val_loss={ckpt['val_loss']:.4f} step={ckpt['step']}\n")
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=80, temperature=0.6, top_k=25, eos_id=tok.eos_id)
        text = tok.decode(out[0].tolist())
        reply = text.split("<|assistant|>")[-1].split("<|user|>")[0].strip()
        user = prompt.split("<|user|>")[1].split("<|assistant|>")[0]
        print(f"User: {user}")
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()
