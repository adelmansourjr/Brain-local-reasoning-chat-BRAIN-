from dataclasses import dataclass
from typing import List, Tuple
import re

from mlx_lm import generate, sample_utils

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
FINAL_OPEN = "<final>"
FINAL_CLOSE = "</final>"

@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: Tuple[str, ...] = (FINAL_CLOSE,)

@dataclass
class ReasoningConfig:
    k: int = 3
    reflect: bool = True
    max_review_tokens: int = 160
    verifier_temperature: float = 0.0

REASON_PROMPT = f"""You are a careful problem solver.
Produce a hidden scratchpad between {THINK_OPEN} and {THINK_CLOSE}, then a short final answer between {FINAL_OPEN} and {FINAL_CLOSE}.
Keep the final answer concise and self-contained.

Question:
{{q}}

Format:
{THINK_OPEN} hidden notes {THINK_CLOSE}
{FINAL_OPEN} final answer {FINAL_CLOSE}
"""

REFLECT_PROMPT_TMPL = f"""You previously answered the question.
Review the final answer. If there is a clear mistake, provide a corrected final answer.

Question:
{{q}}

Your earlier answer:
{FINAL_OPEN}{{a}}{FINAL_CLOSE}

Respond ONLY with one of:
- If correct: {FINAL_OPEN}{{a}}{FINAL_CLOSE}
- If incorrect: {FINAL_OPEN}CORRECTED: <new short answer>{FINAL_CLOSE}
"""

VERIFY_PROMPT_TMPL = f"""You are a strict verifier.
Score the candidate FINAL answer for correctness and relevance to the Question with a number from 0 to 1.

Question:
{{q}}

Candidate:
{FINAL_OPEN}{{a}}{FINAL_CLOSE}

Respond with ONLY a number between 0 and 1.
"""

def _extract_final(text: str) -> str:
    i = text.find(FINAL_OPEN)
    j = text.find(FINAL_CLOSE, i + len(FINAL_OPEN)) if i >= 0 else -1
    if i >= 0 and j > i:
        return text[i + len(FINAL_OPEN):j].strip()
    # fallback: remove any think tags and return remainder
    return text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "").strip()

def _strip_think(text: str) -> str:
    out, start = [], 0
    while True:
        i = text.find(THINK_OPEN, start)
        if i == -1:
            out.append(text[start:])
            break
        out.append(text[start:i])
        j = text.find(THINK_CLOSE, i + len(THINK_OPEN))
        if j == -1:
            break
        start = j + len(THINK_CLOSE)
    return "".join(out)

def generate_once(model, tokenizer, prompt: str, gen_cfg: GenConfig) -> str:
    # Build a sampler (mlx-lm 0.28.x) from temperature/top_p
    sampler = sample_utils.make_sampler(temp=gen_cfg.temperature, top_p=gen_cfg.top_p)
    # Note: mlx-lm generate returns ONLY generated text. We'll parse for FINAL tags later.
    return generate(
        model,
        tokenizer,
        prompt,
        max_tokens=gen_cfg.max_new_tokens,
        sampler=sampler,
        verbose=False,
    )

def score_with_verifier(model, tokenizer, question: str, final_answer: str) -> float:
    prompt = VERIFY_PROMPT_TMPL.format(q=question, a=final_answer)
    text = generate_once(model, tokenizer, prompt, GenConfig(max_new_tokens=8, temperature=0.0, top_p=1.0, stop=()))
    try:
        return max(0.0, min(1.0, float(text.strip().split()[0])))
    except Exception:
        return 0.0

def reason(model, tokenizer, question: str,
           gen_cfg: GenConfig = GenConfig(),
           r_cfg: ReasoningConfig = ReasoningConfig()) -> str:
    base_prompt = REASON_PROMPT.format(q=question)
    candidates: List[str] = []
    for _ in range(r_cfg.k):
        raw = generate_once(model, tokenizer, base_prompt, gen_cfg)
        raw = _strip_think(raw)
        fin = _extract_final(raw)
        candidates.append(fin if fin else raw.strip())

    if r_cfg.reflect:
        improved = []
        for fin in candidates:
            rp = REFLECT_PROMPT_TMPL.format(q=question, a=fin)
            reviewed = generate_once(model, tokenizer, rp, GenConfig(max_new_tokens=r_cfg.max_review_tokens, temperature=0.3, top_p=0.9, stop=(FINAL_CLOSE,)))
            reviewed = _extract_final(reviewed) or fin
            improved.append(reviewed.strip())
        candidates = improved

    scored = [(c, score_with_verifier(model, tokenizer, question, c)) for c in candidates]
    best = max(scored, key=lambda x: x[1])[0]
    return best.strip()

# Expose tokens so callers can set stop correctly
FINAL_CLOSE_TOKEN = FINAL_CLOSE
