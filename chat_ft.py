"""Chat with the fine-tuned ACCEL model."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(history: list[tuple[str, str]], user_msg: str) -> str:
    parts = []
    for user, assistant in history:
        parts.append(f"<|user|>{user}<|assistant|>{assistant}")
    parts.append(f"<|user|>{user_msg}<|assistant|>")
    return "".join(parts)


def clean_reply(text: str) -> str:
    for stop in ("<|user|>", "<|assistant|>", "<|endoftext|>", "<|eos|>"):
        if stop in text:
            text = text.split(stop)[0]
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()


def load_chat_model(checkpoint: Path, device: torch.device):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def generate_reply(
    model,
    tok,
    prompt: str,
    device: torch.device,
    temperature: float,
    max_tokens: int,
) -> str:
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        min_new_tokens=3,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    inputs = tok(prompt, return_tensors="pt").to(device)

    for attempt, sample in enumerate((True, True, False)):
        out = model.generate(
            **inputs,
            do_sample=sample and temperature > 0,
            temperature=max(temperature, 0.01) if sample else 1.0,
            **gen_kwargs,
        )
        decoded = tok.decode(out[0], skip_special_tokens=False)
        raw = decoded[len(prompt):] if decoded.startswith(prompt) else decoded.split("<|assistant|>")[-1]
        reply = clean_reply(raw)
        if reply:
            return reply

    return ""


def chat(checkpoint: Path, temperature: float, max_tokens: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {checkpoint} ({device})...", flush=True)
    model, tok = load_chat_model(checkpoint, device)

    history: list[tuple[str, str]] = []
    print("\nACCEL chat ready. Type your message (quit / exit to stop).\n")
    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_msg or user_msg.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        prompt = build_prompt(history, user_msg)
        reply = generate_reply(model, tok, prompt, device, temperature, max_tokens)
        if not reply:
            reply = "I'm still learning — try rephrasing or ask something from coding, math, or study help."
        print(f"Assistant: {reply}\n")
        history.append((user_msg, reply))
        if len(history) > 4:
            history = history[-4:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with ACCEL")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/accel-ft/best"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    if not (args.checkpoint / "config.json").exists():
        raise SystemExit(
            f"No model at {args.checkpoint}. Run: py -3.13 run.py --train"
        )
    chat(args.checkpoint, args.temperature, args.max_tokens)


if __name__ == "__main__":
    main()
