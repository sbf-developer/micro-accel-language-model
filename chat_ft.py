"""Chat with ACCEL fine-tuned HuggingFace model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(history: list[tuple[str, str]], user_msg: str) -> str:
    parts = []
    for user, assistant in history:
        parts.append(f"<|user|>{user}<|assistant|>{assistant}")
    parts.append(f"<|user|>{user_msg}<|assistant|>")
    return "".join(parts)


def extract_reply(full: str, prompt: str, tok) -> str:
    text = full[len(prompt):] if full.startswith(prompt) else full.split("<|assistant|>")[-1]
    eos = tok.eos_token or ""
    for stop in ("<|user|>", eos):
        if stop and stop in text:
            text = text.split(stop)[0]
    return text.strip()


def chat(checkpoint: Path, temperature: float, max_tokens: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    model.eval()

    history: list[tuple[str, str]] = []
    print("ACCEL fine-tuned chat (quit to exit)\n")
    while True:
        user_msg = input("You: ").strip()
        if not user_msg or user_msg.lower() in {"quit", "exit", "q"}:
            break
        prompt = build_prompt(history, user_msg)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=40,
                pad_token_id=tok.eos_token_id,
            )
        decoded = tok.decode(out[0], skip_special_tokens=False)
        reply = extract_reply(decoded, prompt, tok)
        print(f"Assistant: {reply}\n")
        history.append((user_msg, reply))
        if len(history) > 4:
            history = history[-4:]


tokenizer = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/accel-ft/best"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()
    chat(args.checkpoint, args.temperature, args.max_tokens)
