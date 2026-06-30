"""
Fine-tune a pretrained GPT-2 on the ACE corpus.

Recommended path for functional chat from minimal data: pretrained weights +
ACCEL high-density corpus + assistant-only loss masking.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from accel.data import collate_batch, load_corpus, split_corpus


SPECIAL = [
    "<|user|>", "<|assistant|>", "<|greet|>", "<|ack|>", "<|question|>",
    "<|explain|>", "<|reason|>", "<|empathize|>", "<|refuse|>", "<|summarize|>",
    "<|suggest|>", "<|reflect|>", "<|answer|>",
]


class HFDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int):
        self.samples = []
        user_id = tokenizer.convert_tokens_to_ids("<|user|>")
        asst_id = tokenizer.convert_tokens_to_ids("<|assistant|>")

        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_len, add_special_tokens=False)
            ids = enc["input_ids"] + [tokenizer.eos_token_id]
            if len(ids) < 4:
                continue
            mask = [0.0] * len(ids)
            in_asst = False
            for i, tid in enumerate(ids):
                if tid == asst_id:
                    in_asst = True
                    continue
                if tid == user_id:
                    in_asst = False
                    continue
                if in_asst:
                    mask[i] = 1.0
            if sum(mask) == 0:
                continue
            self.samples.append((ids, ids[1:] + [tokenizer.eos_token_id], mask))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, targets, mask = self.samples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.float32),
        }


def finetune(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    examples = load_corpus(Path(cfg["corpus_path"]))
    train_ex, val_ex = split_corpus(examples)
    train_texts = [ex.text for ex in train_ex]
    val_texts = [ex.text for ex in val_ex]

    base_model = cfg.get("base_model", "distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL})

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    max_len = cfg.get("max_seq_len", 256)
    train_ds = HFDataset(train_texts, tokenizer, max_len)
    val_ds = HFDataset(val_texts, tokenizer, max_len)
    print(f"Train samples: {len(train_ds)}, val: {len(val_ds)}", flush=True)

    batch_size = cfg.get("batch_size", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, max(1, len(val_ds))),
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
    )

    total_steps = cfg.get("total_steps", len(train_loader) * cfg.get("epochs", 30))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("base_lr", 2e-4), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.get("warmup_steps", 100), total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    out_dir = Path(cfg.get("output_dir", "checkpoints/accel-ft"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_state = None
    patience = cfg.get("early_stop_patience", 4)
    stale = 0
    eval_every = cfg.get("eval_every", 25)
    loader_iter = iter(train_loader)

    for step in range(total_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        model.train()
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        logits = model(input_ids).logits
        per_token = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1)).view_as(targets)
        loss = (per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0 or step == total_steps - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    v_logits = model(vb["input_ids"].to(device)).logits
                    v_pt = loss_fn(v_logits.view(-1, v_logits.size(-1)), vb["targets"].to(device).view(-1))
                    v_pt = v_pt.view_as(vb["targets"])
                    vl = (v_pt * vb["loss_mask"].to(device)).sum() / vb["loss_mask"].sum().clamp(min=1.0)
                    val_losses.append(vl.item())
            val_loss = sum(val_losses) / max(1, len(val_losses))
            print(f"step {step:5d}/{total_steps} | train={loss.item():.4f} | val={val_loss:.4f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"Early stop at step {step} (val plateaued).", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.save_pretrained(out_dir / "best")
    tokenizer.save_pretrained(out_dir / "best")
    print(f"Done. Best val={best_val:.4f} -> {out_dir / 'best'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/finetune.yaml"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    finetune(cfg)


if __name__ == "__main__":
    main()
