# micro-accel-language-model

> **Status:** Work in progress — research prototype for sample-efficient chat LM training.

**ACCEL** (Atomic Composable Curriculum for Efficient Language learning) trains conversational models from ~100–200 curated examples using atomic skill tags, curriculum learning, and Fisher-weighted loss.

Includes the ACE corpus generator, MicroGPT (~460K params, from scratch), and DistilGPT-2 fine-tuning.

## Quick start

## The idea

Most LLMs need massive data because they learn everything at once. ACCEL inverts that:

1. **Atomic skills** — teach greetings, reasoning, refusal, etc. as separate primitives
2. **Composable tags** — `<<skill>>` markers (Composable CoT, arXiv:2505.22635) let the model chain skills at inference
3. **Curriculum stages** — format → atoms → composition → dialogue (CLPD / L2M-KD inspired)
4. **Fisher-weighted loss** — upweight high-information tokens (FisherSFT, ICML 2025)
5. **MicroGPT** — ~1M params, GQA + RoPE + SwiGLU, trains on CPU in minutes

## Quick start

```bash
cd "c:\Users\scott\Desktop\dev\language model"
py -3.13 -m pip install -r requirements.txt

# Train (first time) + chat in one command
py -3.13 run.py --train

# Chat only (after training)
py -3.13 run.py

# Or directly:
py -3.13 chat_ft.py
```

On Windows you can also double-click `chat.bat`.

## Project layout

```
accel/
  model.py       # MicroGPT architecture
  tokenizer.py   # BPE + skill special tokens
  data.py        # Dataset + assistant-only loss mask
  loss.py        # Fisher / focal token weighting
  curriculum.py  # Staged easy→hard scheduler
  train.py       # Training loop
data/
  generate_corpus.py   # ACE corpus generator
  ace_corpus.jsonl     # Generated training data
config/default.yaml
chat.py
```

## Training data (ACE corpus)

Each JSONL row has:

| Field | Meaning |
|-------|---------|
| `text` | Full conversation with `<\|user\|>` / `<\|assistant\|>` tokens |
| `stage` | Curriculum stage 0–3 |
| `skills` | Atomic skills taught |
| `difficulty` | 0.1 (easy) → 0.8 (hard) |

Example (stage 2 — compositional):

```
<|user|>What's 24 divided by 6?
<|assistant|><|reason|>24 ÷ 6 asks how many groups of 6 fit in 24; that's 4 groups.<|answer|>24 divided by 6 equals 4.
```

## Research foundations

- **Composable CoT** — atomic skill tags for zero-shot composition
- **FisherSFT** — information-gain-weighted SFT examples
- **Data diversity for compositional generalization** — balanced structural + semantic variety
- **Curriculum learning** — easy-to-hard staging (DA-KD, CLPD, L2M-KD)

## Config

Edit `config/default.yaml`:

- `loss_mode`: `fisher` (default), `focal`, or `standard`
- Model size: `d_model`, `n_layers`, etc.
- Curriculum step counts in `accel/curriculum.py`

## Limitations (honest)

This is a **proof-of-concept micro-model**, not GPT-4. It learns chat *structure* and *patterns* from tiny data. For broad world knowledge you still need scale or a pretrained base. ACCEL shows how far principled data + curriculum + loss design can push sample efficiency.

## Next steps

- Fine-tune a pretrained small model (Qwen2.5-0.5B) on the ACE corpus for stronger results
- Expand corpus with your domain-specific atomic skills
- Add teacher distillation from a larger model into MicroGPT
