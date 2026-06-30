"""ACCEL training loop with curriculum + Fisher-weighted loss."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from accel.curriculum import CurriculumScheduler
from accel.data import ChatExample, collate_batch, load_corpus, split_corpus
from accel.loss import fisher_weighted_loss, focal_token_loss
from accel.model import MicroGPT, ModelConfig
from accel.tokenizer import Tokenizer


def log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def lr_at_step(step: int, base_lr: float, warmup: int, total: int, scale: float) -> float:
    if step < warmup:
        return base_lr * scale * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * scale * max(cosine, 0.1)


class PrebuiltACCELDataset(Dataset):
    """Pre-tokenize once; filter by curriculum stage with optional replay."""

    def __init__(
        self,
        examples: list[ChatExample],
        tokenizer: Tokenizer,
        max_seq_len: int,
        max_stage: int,
        replay_below: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_stage = max_stage
        self.replay_below = replay_below
        self.samples: list[tuple[list[int], list[int], list[float], int]] = []

        for ex in examples:
            if ex.stage > max_stage:
                continue
            ids = tokenizer.encode(ex.text, add_eos=True)
            if len(ids) < 4 or len(ids) > max_seq_len:
                continue
            targets = ids[1:] + [tokenizer.eos_id]
            mask = self._assistant_mask(ids)
            if sum(mask) == 0:
                continue
            self.samples.append((ids, targets, mask, ex.stage))

    def _assistant_mask(self, ids: list[int]) -> list[float]:
        mask = [0.0] * len(ids)
        in_assistant = False
        for i, tid in enumerate(ids):
            if tid == self.tokenizer.assistant_id:
                in_assistant = True
                continue
            if tid == self.tokenizer.user_id:
                in_assistant = False
                continue
            if in_assistant:
                mask[i] = 1.0
        return mask

    def with_max_stage(self, max_stage: int) -> PrebuiltACCELDataset:
        clone = PrebuiltACCELDataset.__new__(PrebuiltACCELDataset)
        clone.tokenizer = self.tokenizer
        clone.max_stage = max_stage
        clone.replay_below = self.replay_below
        clone.samples = [s for s in self.samples if s[3] <= max_stage]
        return clone

    def sample_batch_indices(self, batch_size: int, rng: random.Random) -> list[int]:
        """Mix current-stage examples with replay from earlier stages."""
        if not self.samples:
            return []
        by_stage: dict[int, list[int]] = {}
        for i, (_, _, _, stage) in enumerate(self.samples):
            by_stage.setdefault(stage, []).append(i)

        indices: list[int] = []
        stages = sorted(by_stage.keys())
        current = stages[-1]
        n_replay = batch_size // 2 if self.replay_below and len(stages) > 1 else 0
        n_current = batch_size - n_replay

        for _ in range(n_current):
            indices.append(rng.choice(by_stage[current]))
        for _ in range(n_replay):
            earlier = rng.choice(stages[:-1])
            indices.append(rng.choice(by_stage[earlier]))
        rng.shuffle(indices)
        return indices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, targets, mask, _ = self.samples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.float32),
        }


def train(cfg: dict) -> None:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    corpus_path = Path(cfg["corpus_path"])
    if not corpus_path.exists():
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.generate_corpus import main as gen

        gen()

    examples = load_corpus(corpus_path)
    train_ex, val_ex = split_corpus(examples)
    log(f"Corpus: {len(examples)} total, {len(train_ex)} train, {len(val_ex)} val")

    texts = [ex.text for ex in examples]
    tokenizer = Tokenizer.train(texts, vocab_size=cfg["vocab_size"])
    log(f"Tokenizer vocab: {tokenizer.vocab_size}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        ffn_dim=cfg["ffn_dim"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
    )
    model = MicroGPT(model_cfg).to(device)
    log(f"Model parameters: {model.count_parameters():,}")

    full_train = PrebuiltACCELDataset(train_ex, tokenizer, cfg["max_seq_len"], max_stage=3)
    full_val = PrebuiltACCELDataset(val_ex, tokenizer, cfg["max_seq_len"], max_stage=3)

    curriculum = CurriculumScheduler()
    total_steps = curriculum.total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["base_lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    best_val = float("inf")
    best_final_val = float("inf")
    history = []
    current_stage = -1
    stage_train: PrebuiltACCELDataset | None = None
    stage_val: PrebuiltACCELDataset | None = None
    val_loader: DataLoader | None = None
    batch_rng = random.Random(cfg["seed"])

    for step in range(total_steps):
        max_stage = curriculum.max_data_stage(step)
        if max_stage != current_stage:
            current_stage = max_stage
            stage_train = full_train.with_max_stage(max_stage)
            stage_val = full_val.with_max_stage(max_stage)
            val_loader = DataLoader(
                stage_val,
                batch_size=min(cfg["batch_size"], max(1, len(stage_val))),
                shuffle=False,
                collate_fn=lambda b: collate_batch(b, tokenizer.pad_id),
            )
            log(f"Stage '{curriculum.stage_name(step)}' — train={len(stage_train)}, val={len(stage_val)}")

        assert stage_train is not None and val_loader is not None
        lr = lr_at_step(step, cfg["base_lr"], cfg["warmup_steps"], total_steps, curriculum.lr_scale(step))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        idx = stage_train.sample_batch_indices(
            min(cfg["batch_size"], len(stage_train)),
            batch_rng,
        )
        batch_items = [stage_train[i] for i in idx]
        batch = collate_batch(batch_items, tokenizer.pad_id)
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        model.train()
        logits, out = model(input_ids, targets, loss_mask)
        per_token = out["per_token_loss"]

        if cfg["loss_mode"] == "fisher":
            loss = fisher_weighted_loss(per_token, loss_mask, gamma=cfg["fisher_gamma"])
        elif cfg["loss_mode"] == "focal":
            loss, _ = focal_token_loss(logits, targets, loss_mask, gamma=cfg["focal_gamma"])
        else:
            loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        if step % 25 == 0 or step == total_steps - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    v_logits, v_out = model(
                        vb["input_ids"].to(device),
                        vb["targets"].to(device),
                        vb["loss_mask"].to(device),
                    )
                    if cfg["loss_mode"] == "fisher":
                        vl = fisher_weighted_loss(
                            v_out["per_token_loss"],
                            vb["loss_mask"].to(device),
                            gamma=cfg["fisher_gamma"],
                        )
                    elif cfg["loss_mode"] == "focal":
                        vl, _ = focal_token_loss(
                            v_logits,
                            vb["targets"].to(device),
                            vb["loss_mask"].to(device),
                            gamma=cfg["focal_gamma"],
                        )
                    else:
                        vl = v_out["loss"]
                    val_losses.append(vl.item())

            val_loss = sum(val_losses) / max(1, len(val_losses))
            record = {
                "step": step,
                "stage": curriculum.stage_name(step),
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "lr": lr,
            }
            history.append(record)
            log(
                f"step {step:4d} | stage={record['stage']:8s} | "
                f"train={loss.item():.4f} | val={val_loss:.4f} | lr={lr:.2e}"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": model_cfg.__dict__,
                        "step": step,
                        "val_loss": val_loss,
                        "stage": curriculum.stage_name(step),
                    },
                    out_dir / "best.pt",
                )

            if max_stage == 3 and val_loss < best_final_val:
                best_final_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": model_cfg.__dict__,
                        "step": step,
                        "val_loss": val_loss,
                        "stage": curriculum.stage_name(step),
                    },
                    out_dir / "best_final.pt",
                )

    torch.save(model.state_dict(), out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    log(f"Done. Best val loss: {best_val:.4f}. Checkpoints in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACCEL MicroGPT")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
