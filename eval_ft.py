"""Non-interactive eval for fine-tuned ACCEL model."""

from pathlib import Path

import torch
from chat_ft import clean_reply, generate_reply, load_chat_model

checkpoint = Path("checkpoints/accel-ft/best")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not (checkpoint / "config.json").exists():
    raise SystemExit("No checkpoint. Run: py -3.13 run.py --train")

print(f"Loading {checkpoint}...", flush=True)
model, tok = load_chat_model(checkpoint, device)

prompts = [
    ("Hi", "<|user|>Hi<|assistant|>"),
    ("What is a variable?", "<|user|>What is a variable?<|assistant|>"),
    ("I'm stressed about exams", "<|user|>I'm stressed about exams<|assistant|>"),
    ("What's 24 divided by 6?", "<|user|>What's 24 divided by 6?<|assistant|>"),
    ("Help me learn Python", "<|user|>Help me learn Python<|assistant|>"),
]

print()
for label, prompt in prompts:
    reply = generate_reply(model, tok, prompt, device, temperature=0.6, max_tokens=100)
    print(f"You: {label}")
    print(f"Assistant: {reply or '(empty)'}\n")
