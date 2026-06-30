"""Interactive chat with a trained ACCEL model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from accel.model import MicroGPT, ModelConfig
from accel.tokenizer import Tokenizer


def load_model(checkpoint_dir: Path, device: torch.device, weights: str = "best_final") -> tuple[MicroGPT, Tokenizer]:
    tok = Tokenizer.load(checkpoint_dir / "tokenizer.json")
    ckpt_path = checkpoint_dir / f"{weights}.pt"
    if not ckpt_path.exists():
        ckpt_path = checkpoint_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = MicroGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok


def build_prompt(history: list[tuple[str, str]], user_msg: str) -> str:
    parts = []
    for user, assistant in history:
        parts.append(f"<|user|>{user}<|assistant|>{assistant}")
    parts.append(f"<|user|>{user_msg}<|assistant|>")
    return "".join(parts)


def extract_assistant_reply(decoded: str, prompt: str) -> str:
    if prompt in decoded:
        reply = decoded[len(prompt) :]
    else:
        reply = decoded.split("<|assistant|>")[-1]
    for stop in ("<|user|>", "<|eos|>"):
        if stop in reply:
            reply = reply.split(stop)[0]
    return reply.strip()


@torch.no_grad()
def chat(checkpoint_dir: Path, temperature: float, top_k: int, max_tokens: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(checkpoint_dir, device)
    history: list[tuple[str, str]] = []

    print("ACCEL chat (type 'quit' to exit)\n")
    while True:
        user_msg = input("You: ").strip()
        if not user_msg or user_msg.lower() in {"quit", "exit", "q"}:
            break

        prompt = build_prompt(history, user_msg)
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(
            ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tok.eos_id,
        )
        decoded = tok.decode(out[0].tolist())
        reply = extract_assistant_reply(decoded, prompt)
        print(f"Assistant: {reply}\n")
        history.append((user_msg, reply))
        if len(history) > 4:
            history = history[-4:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/accel-v1"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()
    chat(args.checkpoint, args.temperature, args.top_k, args.max_tokens)


if __name__ == "__main__":
    main()
