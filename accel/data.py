"""Dataset loading with assistant-only loss masking."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from accel.tokenizer import Tokenizer


@dataclass
class ChatExample:
    text: str
    stage: int
    skills: list[str]
    difficulty: float


class ACCELDataset(Dataset):
    def __init__(
        self,
        examples: list[ChatExample],
        tokenizer: Tokenizer,
        max_seq_len: int = 512,
        max_stage: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = self._build_samples(examples, max_stage)

    def _build_samples(
        self,
        examples: list[ChatExample],
        max_stage: int,
    ) -> list[tuple[list[int], list[int], list[float]]]:
        samples = []
        for ex in examples:
            if ex.stage > max_stage:
                continue
            ids = self.tokenizer.encode(ex.text, add_eos=True)
            if len(ids) < 4 or len(ids) > self.max_seq_len:
                continue
            targets = ids[1:] + [self.tokenizer.eos_id]
            mask = self._assistant_mask(ids)
            if mask.sum() == 0:
                continue
            samples.append((ids, targets, mask.tolist(), ex.difficulty))
        return samples

    def _assistant_mask(self, ids: list[int]) -> torch.Tensor:
        """Train only on assistant tokens (SFT best practice)."""
        mask = torch.zeros(len(ids), dtype=torch.float32)
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, targets, mask, difficulty = self.samples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.float32),
            "difficulty": torch.tensor(difficulty, dtype=torch.float32),
        }


def collate_batch(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    fields = ("input_ids", "targets", "loss_mask")
    out: dict[str, torch.Tensor] = {}
    for field in fields:
        rows = []
        for item in batch:
            t = item[field]
            pad_len = max_len - t.size(0)
            if field == "input_ids":
                rows.append(torch.cat([t, torch.full((pad_len,), pad_id, dtype=torch.long)]))
            elif field == "targets":
                rows.append(torch.cat([t, torch.full((pad_len,), pad_id, dtype=torch.long)]))
            else:
                rows.append(torch.cat([t, torch.zeros(pad_len, dtype=torch.float32)]))
        out[field] = torch.stack(rows)
    if "difficulty" in batch[0]:
        out["difficulty"] = torch.stack([item["difficulty"] for item in batch])
    return out


def load_corpus(path: Path) -> list[ChatExample]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(
                ChatExample(
                    text=row["text"],
                    stage=row["stage"],
                    skills=row.get("skills", []),
                    difficulty=row.get("difficulty", 0.5),
                )
            )
    return examples


def split_corpus(
    examples: list[ChatExample],
    val_ratio: float = 0.08,
    seed: int = 42,
) -> tuple[list[ChatExample], list[ChatExample]]:
    rng = random.Random(seed)
    by_stage: dict[int, list[ChatExample]] = {}
    for ex in examples:
        by_stage.setdefault(ex.stage, []).append(ex)

    train, val = [], []
    for stage_examples in by_stage.values():
        rng.shuffle(stage_examples)
        n_val = max(1, int(len(stage_examples) * val_ratio))
        val.extend(stage_examples[:n_val])
        train.extend(stage_examples[n_val:])
    rng.shuffle(train)
    return train, val
