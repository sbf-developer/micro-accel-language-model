"""Byte-level BPE tokenizer with chat and skill special tokens."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|user|>",
    "<|assistant|>",
    "<|eos|>",
    "<|greet|>",
    "<|ack|>",
    "<|question|>",
    "<|explain|>",
    "<|reason|>",
    "<|empathize|>",
    "<|refuse|>",
    "<|summarize|>",
    "<|suggest|>",
    "<|reflect|>",
    "<|answer|>",
]


@dataclass
class Tokenizer:
    vocab: dict[str, int] = field(default_factory=dict)
    merges: list[tuple[str, str]] = field(default_factory=list)
    special_tokens: list[str] = field(default_factory=lambda: list(SPECIAL_TOKENS))

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 512) -> Tokenizer:
        tok = cls()
        tok.vocab = {t: i for i, t in enumerate(tok.special_tokens)}
        special_set = set(tok.special_tokens)

        corpus: dict[tuple[str, ...], int] = {}
        for text in texts:
            for word in cls._pretokenize(text):
                if word in special_set:
                    if word not in tok.vocab:
                        tok.vocab[word] = len(tok.vocab)
                    continue
                key = tuple(word)
                corpus[key] = corpus.get(key, 0) + 1

        while len(tok.vocab) < vocab_size and corpus:
            pairs = Counter()
            for word, freq in corpus.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            tok.merges.append(best)
            new_token = best[0] + best[1]
            tok.vocab[new_token] = len(tok.vocab)
            corpus = tok._merge_vocab(corpus, best)

        return tok

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        pattern = re.compile(
            r"<\|[^|]+\|>|'[\w']+|[^\s\w<>|]+|\w+|\s+",
            re.UNICODE,
        )
        return pattern.findall(text)

    @staticmethod
    def _merge_vocab(
        corpus: dict[tuple[str, ...], int],
        pair: tuple[str, str],
    ) -> dict[tuple[str, ...], int]:
        merged: dict[tuple[str, ...], int] = {}
        bigram = pair[0] + pair[1]
        first, second = pair
        for word, freq in corpus.items():
            i = 0
            new_word: list[str] = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            merged[tuple(new_word)] = merged.get(tuple(new_word), 0) + freq
        return merged

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        special_set = set(self.special_tokens)
        ids: list[int] = []
        for word in self._pretokenize(text):
            if word in special_set:
                ids.append(self.vocab[word])
            else:
                ids.extend(self._encode_word(word))
        if add_eos:
            ids.append(self.vocab["<|eos|>"])
        return ids

    def _encode_word(self, word: str) -> list[int]:
        if word in self.vocab:
            return [self.vocab[word]]
        pieces = list(word)
        for a, b in self.merges:
            i = 0
            merged: list[str] = []
            while i < len(pieces):
                if i < len(pieces) - 1 and pieces[i] == a and pieces[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(pieces[i])
                    i += 1
            pieces = merged
        unk = self.vocab.get("<|pad|>", 0)
        return [self.vocab.get(p, unk) for p in pieces]

    def decode(self, ids: list[int]) -> str:
        inv = {i: t for t, i in self.vocab.items()}
        parts: list[str] = []
        for i in ids:
            tok = inv.get(i, "")
            if tok == "<|pad|>":
                continue
            if tok in self.special_tokens:
                parts.append(tok)
            else:
                parts.append(tok)
        return "".join(parts)

    @property
    def pad_id(self) -> int:
        return self.vocab["<|pad|>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<|eos|>"]

    @property
    def user_id(self) -> int:
        return self.vocab["<|user|>"]

    @property
    def assistant_id(self) -> int:
        return self.vocab["<|assistant|>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            vocab=data["vocab"],
            merges=[tuple(m) for m in data["merges"]],
            special_tokens=data.get("special_tokens", SPECIAL_TOKENS),
        )
