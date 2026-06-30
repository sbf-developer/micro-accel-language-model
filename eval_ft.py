"""Non-interactive eval for fine-tuned ACCEL model."""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = Path("checkpoints/accel-ft/best")
tok = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
model.eval()

prompts = [
    "<|user|>Hi<|assistant|>",
    "<|user|>What is a variable?<|assistant|>",
    "<|user|>I'm stressed about exams<|assistant|>",
    "<|user|>What's 24 divided by 6?<|assistant|>",
]

for prompt in prompts:
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7, top_k=40)
    text = tok.decode(out[0], skip_special_tokens=False)
    reply = text[len(prompt):].split("<|user|>")[0].split("<|eos|>")[0]
    user = prompt.split("<|user|>")[1].split("<|assistant|>")[0]
    print(f"User: {user}")
    print(f"Assistant: {reply.strip()}\n")
