"""
Generate the ACE (Atomic Chat Elements) corpus.

Design principles (research-backed):
  - Atomic skills: each example teaches ONE transferable conversational primitive
  - Composable CoT tags: <<<skill>>> markers enable inference-time composition
  - Balanced structural + semantic diversity (OpenReview compositional gen. paper)
  - Curriculum stages 0-3: format -> atoms -> composition -> full dialogue
  - High information density: every token teaches format OR skill OR reasoning
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def _fmt(user: str, assistant: str) -> str:
    return f"<|user|>{user}<|assistant|>{assistant}"


def stage0_format() -> list[dict]:
    """Stage 0: learn chat structure only."""
    pairs = [
        ("Hi", "Hello!"),
        ("Hello there", "Hi! Nice to meet you."),
        ("Hey", "Hey! What can I help with?"),
        ("Good morning", "Good morning! How are you today?"),
        ("Thanks", "You're welcome!"),
        ("Thank you", "Happy to help!"),
        ("Bye", "Goodbye! Take care."),
        ("See you", "See you later!"),
        ("Ok", "Sounds good."),
        ("Yes", "Great, noted."),
        ("No", "Understood."),
        ("Help", "I'm here to help. What's on your mind?"),
        ("?", "Could you tell me more?"),
        ("Test", "I'm working and ready to chat."),
    ]
    return [
        {"text": _fmt(u, a), "stage": 0, "skills": ["format"], "difficulty": 0.1}
        for u, a in pairs
    ]


def stage1_atomic() -> list[dict]:
    """Stage 1: atomic conversational skills with explicit skill tags."""
    examples = []

    greets = [
        ("Good evening!", "<|greet|>Good evening! Hope you're having a nice day."),
        ("Hi, I'm Alex.", "<|greet|>Hi Alex! Great to meet you."),
        ("Hello from Denmark!", "<|greet|>Hello! Welcome — how can I assist you today?"),
        ("Yo", "<|greet|>Hey! What brings you here today?"),
        ("Good afternoon", "<|greet|>Good afternoon! Ready when you are."),
    ]
    for u, a in greets:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["greet"], "difficulty": 0.2})

    acks = [
        ("I need help with Python.", "<|ack|>Got it — you want help with Python. I'll focus on that."),
        ("My code keeps crashing.", "<|ack|>I understand your code is crashing. Let's debug it together."),
        ("I'm learning machine learning.", "<|ack|>You're learning ML — I'll keep explanations clear and practical."),
        ("Can we talk about APIs?", "<|ack|>Sure, APIs — I'll explain how programs talk to each other."),
        ("I want to optimize my model.", "<|ack|>You want to optimize your model. We'll look at data, architecture, and training."),
    ]
    for u, a in acks:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["ack"], "difficulty": 0.25})

    questions = [
        ("Fix my bug.", "<|question|>What error message do you see, and when does it happen?"),
        ("Explain transformers.", "<|question|>Are you familiar with neural networks, or should I start from basics?"),
        ("I'm stuck on homework.", "<|question|>Which problem number is giving you trouble?"),
        ("Help me design a database.", "<|question|>What kind of data will you store, and how will you query it?"),
        ("My model overfits.", "<|question|>How much training data do you have, and what's your validation loss doing?"),
    ]
    for u, a in questions:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["question"], "difficulty": 0.3})

    explains = [
        (
            "What is a variable?",
            "<|explain|>A variable is a named box in memory that stores a value you can change later.",
        ),
        (
            "What is gradient descent?",
            "<|explain|>Gradient descent adjusts model weights step by step to reduce prediction error.",
        ),
        (
            "What is overfitting?",
            "<|explain|>Overfitting means the model memorizes training data but fails on new examples.",
        ),
        (
            "What is a token?",
            "<|explain|>A token is a text chunk — often a word or subword — that the model reads and predicts.",
        ),
        (
            "What is fine-tuning?",
            "<|explain|>Fine-tuning adapts a pretrained model to your task using a smaller, targeted dataset.",
        ),
    ]
    for u, a in explains:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["explain"], "difficulty": 0.35})

    reasons = [
        (
            "Why is 7+5=12?",
            "<|reason|>7+5 splits into 7+3=10 plus 2 more, giving 12.",
        ),
        (
            "Should I use a list or dict?",
            "<|reason|>Use a list for ordered items; use a dict when you need fast lookup by key.",
        ),
        (
            "Why normalize inputs?",
            "<|reason|>Normalization keeps feature scales similar so gradients stay stable during training.",
        ),
        (
            "Why use dropout?",
            "<|reason|>Dropout randomly disables neurons so the model can't rely on any single path — reducing overfitting.",
        ),
    ]
    for u, a in reasons:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["reason"], "difficulty": 0.4})

    empathize = [
        (
            "I'm frustrated with this project.",
            "<|empathize|>That sounds exhausting — frustration usually means you've been trying hard.",
        ),
        (
            "Exams stress me out.",
            "<|empathize|>Exam stress is common; breaking study into small blocks often helps.",
        ),
        (
            "I feel behind everyone else.",
            "<|empathize|>It's easy to feel behind online — your pace is valid if you're learning consistently.",
        ),
    ]
    for u, a in empathize:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["empathize"], "difficulty": 0.35})

    refuses = [
        (
            "Write my essay for me.",
            "<|refuse|>I can't write the essay for you, but I can help you outline ideas and improve drafts.",
        ),
        (
            "Help me cheat on the test.",
            "<|refuse|>I won't help with cheating, but I'm happy to explain concepts so you can learn.",
        ),
        (
            "Generate malware code.",
            "<|refuse|>I can't help create malware, but I can explain security concepts defensively.",
        ),
    ]
    for u, a in refuses:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["refuse"], "difficulty": 0.4})

    summarize = [
        (
            "We discussed loops, functions, and debugging for an hour.",
            "<|summarize|>Summary: we covered loops, functions, and practical debugging steps.",
        ),
        (
            "Meeting notes: launch Friday, fix login bug, update docs.",
            "<|summarize|>Action items: launch Friday, fix login bug, refresh documentation.",
        ),
    ]
    for u, a in summarize:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["summarize"], "difficulty": 0.45})

    suggest = [
        (
            "How do I learn Python fast?",
            "<|suggest|>Try 30 minutes daily: tiny scripts, then one small project each week.",
        ),
        (
            "I'm bad at math for ML.",
            "<|suggest|>Start with linear algebra basics and practice with NumPy exercises.",
        ),
        (
            "How do I pick a project?",
            "<|suggest|>Pick a problem you personally have — motivation carries you through the hard parts.",
        ),
    ]
    for u, a in suggest:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["suggest"], "difficulty": 0.4})

    reflect = [
        (
            "I think I need more practice.",
            "<|reflect|>You're saying you need more practice — that's a solid self-assessment.",
        ),
        (
            "Maybe I'm overcomplicating this.",
            "<|reflect|>You feel you're overcomplicating it — simplifying is often the right instinct.",
        ),
    ]
    for u, a in reflect:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["reflect"], "difficulty": 0.35})

    answers = [
        ("Capital of France?", "<|answer|>The capital of France is Paris."),
        ("2+2?", "<|answer|>2 plus 2 equals 4."),
        ("Pi roughly?", "<|answer|>Pi is approximately 3.14159."),
    ]
    for u, a in answers:
        examples.append({"text": _fmt(u, a), "stage": 1, "skills": ["answer"], "difficulty": 0.3})

    return examples


def stage2_compose() -> list[dict]:
    """Stage 2: compose multiple atomic skills (Composable CoT format)."""
    compositions = [
        (
            "I'm new to coding and feel lost.",
            "<|empathize|>Starting fresh can feel overwhelming.<|ack|>You want a clear path into coding."
            "<|suggest|>Begin with Python: variables, loops, then one tiny project.<|question|>Do you prefer games or data as a first project?",
            ["empathize", "ack", "suggest", "question"],
            0.55,
        ),
        (
            "What is 24 divided by 6?",
            "<|reason|>24 ÷ 6 asks how many groups of 6 fit in 24; that's 4 groups."
            "<|answer|>24 divided by 6 equals 4.",
            ["reason", "answer"],
            0.5,
        ),
        (
            "Explain recursion simply.",
            "<|ack|>You want a simple recursion explanation."
            "<|explain|>Recursion is when a function calls itself on a smaller piece of the problem."
            "<|reason|>It needs a base case to stop, otherwise it runs forever.",
            ["ack", "explain", "reason"],
            0.6,
        ),
        (
            "Can you do my homework?",
            "<|refuse|>I can't complete homework for you."
            "<|suggest|>Share the problem and I'll guide you step by step.",
            ["refuse", "suggest"],
            0.55,
        ),
        (
            "Summarize: we fixed a null pointer and added tests.",
            "<|ack|>You fixed a null pointer and added tests."
            "<|summarize|>Key takeaway: bug fixed with a null check plus regression tests.",
            ["ack", "summarize"],
            0.5,
        ),
        (
            "Why use Git?",
            "<|explain|>Git tracks code history so you can collaborate and undo mistakes."
            "<|reason|>Without version control, teams overwrite each other and lose work.",
            ["explain", "reason"],
            0.55,
        ),
        (
            "I'm happy I finally got it working!",
            "<|reflect|>You finally got it working — that's a real win."
            "<|empathize|>That relief after a long bug hunt is the best feeling.",
            ["reflect", "empathize"],
            0.5,
        ),
        (
            "How do I sort a list in Python?",
            "<|question|>Do you want ascending order or a custom key?"
            "<|explain|>Use sorted(my_list) for a new list, or my_list.sort() in place."
            "<|reason|>sort() is faster when you don't need the original order preserved.",
            ["question", "explain", "reason"],
            0.65,
        ),
    ]
    return [
        {
            "text": _fmt(u, a),
            "stage": 2,
            "skills": skills,
            "difficulty": diff,
        }
        for u, a, skills, diff in compositions
    ]


def stage3_dialogues() -> list[dict]:
    """Stage 3: natural multi-turn dialogues composing learned skills."""
    dialogues = [
        (
            "<|user|>Hi, can you help me learn Python?<|assistant|>"
            "Hi! Absolutely — what have you tried so far?"
            "<|user|>Just print statements.<|assistant|>"
            "Great start. Next, try variables and if-statements — want a 5-minute exercise?"
        ),
        (
            "<|user|>What's the difference between list and tuple?<|assistant|>"
            "Lists are mutable; tuples are fixed once created."
            "<|user|>When should I use tuples?<|assistant|>"
            "Use tuples for records that shouldn't change, like coordinates (x, y)."
        ),
        (
            "<|user|>I'm stressed about my deadline.<|assistant|>"
            "Deadlines are tough. What's the smallest deliverable you could finish today?"
            "<|user|>Maybe the README and tests.<|assistant|>"
            "Perfect — ship README + tests first; that momentum usually unlocks the rest."
        ),
        (
            "<|user|>Explain overfitting like I'm 12.<|assistant|>"
            "It's like memorizing answers for one quiz but failing a new quiz."
            "<|user|>How do I fix it?<|assistant|>"
            "Use more varied examples, simplify the model, or stop training earlier."
        ),
        (
            "<|user|>Write my thesis for me.<|assistant|>"
            "I can't write your thesis, but I can help structure chapters and sharpen arguments."
            "<|user|>Ok, help with the intro.<|assistant|>"
            "Share your research question and I'll suggest an intro outline with key sections."
        ),
        (
            "<|user|>Why is my loss not decreasing?<|assistant|>"
            "Common causes: learning rate too high/low, bad data pipeline, or wrong loss mask."
            "<|user|>Learning rate is 1e-1.<|assistant|>"
            "1e-1 is often too high for fine-tuning — try 1e-4 to 3e-4 and watch val loss."
        ),
        (
            "<|user|>Teach me attention mechanisms.<|assistant|>"
            "Attention lets each token weigh other tokens — 'which words matter for this word?'"
            "<|user|>Why multi-head?<|assistant|>"
            "Multiple heads learn different relationship types: syntax, coreference, position, etc."
        ),
        (
            "<|user|>Give me a study plan for ML.<|assistant|>"
            "Week 1-2: Python + NumPy. Week 3-4: linear models. Week 5-6: small neural net project."
            "<|user|>I only have 30 min/day.<|assistant|>"
            "Then one focused micro-topic daily + weekend project hour. Consistency beats cramming."
        ),
    ]
    return [
        {"text": d, "stage": 3, "skills": ["dialogue"], "difficulty": 0.7 + i * 0.02}
        for i, d in enumerate(dialogues)
    ]


def expand_templates(rng: random.Random) -> list[dict]:
    """High-density template expansion for balanced structural + semantic diversity."""
    out: list[dict] = []

    names = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley"]
    for name in names:
        out.append({
            "text": _fmt(f"Hi, I'm {name}.", f"<|greet|>Hi {name}! Great to meet you."),
            "stage": 1, "skills": ["greet"], "difficulty": 0.22,
        })

    concepts = [
        ("a function", "A function is a reusable block of code that takes input and returns output."),
        ("a loop", "A loop repeats code until a condition is met — for or while in Python."),
        ("an API", "An API is a contract that lets programs request data or actions from another service."),
        ("a class", "A class is a blueprint for creating objects with shared behavior and state."),
        ("a tensor", "A tensor is a multi-dimensional array used to represent data in deep learning."),
        ("backpropagation", "Backpropagation computes gradients by chaining the chain rule through the network."),
        ("a learning rate", "The learning rate controls how big each weight update step is during training."),
        ("batch size", "Batch size is how many examples the model sees before one optimizer update."),
    ]
    for term, definition in concepts:
        out.append({
            "text": _fmt(f"What is {term}?", f"<|explain|>{definition}"),
            "stage": 1, "skills": ["explain"], "difficulty": 0.35,
        })

    math_pairs = [
        ("3+4", "3 plus 4 equals 7.", "7+3=10 plus 4 more gives 7."),
        ("8+9", "8 plus 9 equals 17.", "8+9 is 8+2=10 plus 7, giving 17."),
        ("12-5", "12 minus 5 equals 7.", "From 12, subtract 5 to get 7."),
        ("6*7", "6 times 7 equals 42.", "6*7 is six sevens: 7+7+7+7+7+7=42."),
        ("20/4", "20 divided by 4 equals 5.", "20 ÷ 4 counts groups of 4 in 20 — that's 5."),
    ]
    for expr, ans, reason in math_pairs:
        out.append({
            "text": _fmt(f"What is {expr}?", f"<|reason|>{reason}<|answer|>{ans}"),
            "stage": 2, "skills": ["reason", "answer"], "difficulty": 0.5,
        })

    stress_lines = [
        "I'm overwhelmed.",
        "This is too hard.",
        "I want to give up.",
        "I feel stuck.",
    ]
    stress_replies = [
        "<|empathize|>Feeling overwhelmed is a signal you've been pushing hard — that's not failure.",
        "<|empathize|>When something feels too hard, shrinking the next step often helps.",
        "<|empathize|>Wanting to give up usually means the problem needs a smaller entry point.",
        "<|empathize|>Feeling stuck is normal — it often comes right before a breakthrough.",
    ]
    for user, reply in zip(stress_lines, stress_replies):
        out.append({"text": _fmt(user, reply), "stage": 1, "skills": ["empathize"], "difficulty": 0.35})

    refuse_prompts = [
        ("Do my assignment.", "I can't do assignments for you, but I can explain steps and check your work."),
        ("Hack this account.", "I can't help with unauthorized access — I can explain security best practices instead."),
        ("Tell me exam answers.", "I won't provide exam answers, but I can quiz you on concepts to prepare."),
    ]
    for user, reply in refuse_prompts:
        out.append({
            "text": _fmt(user, f"<|refuse|>{reply}"),
            "stage": 1, "skills": ["refuse"], "difficulty": 0.4,
        })

    rng.shuffle(out)
    return out


def augment_variants(examples: list[dict], rng: random.Random) -> list[dict]:
    """Semantic diversity: paraphrase user side while keeping assistant structure."""
    user_variants = {
        "Hi": ["Hello", "Hey there", "Hi there"],
        "Help": ["I need help", "Can you assist?", "Support please"],
        "Thanks": ["Thank you!", "Appreciate it", "Thanks a lot"],
    }
    out = list(examples)
    for ex in examples:
        if ex["stage"] > 1:
            continue
        for u, alts in user_variants.items():
            if u in ex["text"]:
                for alt in alts:
                    new = dict(ex)
                    new["text"] = ex["text"].replace(f"<|user|>{u}", f"<|user|>{alt}", 1)
                    new["difficulty"] = ex["difficulty"] + 0.02
                    out.append(new)
    rng.shuffle(out)
    return out


def generate_corpus(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    corpus = []
    corpus.extend(stage0_format())
    corpus.extend(stage1_atomic())
    corpus.extend(stage2_compose())
    corpus.extend(stage3_dialogues())
    corpus.extend(expand_templates(rng))
    corpus = augment_variants(corpus, rng)
    return corpus


def main() -> None:
    out_path = Path(__file__).parent / "ace_corpus.jsonl"
    corpus = generate_corpus()
    with out_path.open("w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(corpus)} examples to {out_path}")


if __name__ == "__main__":
    main()
