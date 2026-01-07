#!/usr/bin/env python3
import asyncio, re, textwrap, os, argparse
from typing import List, Tuple
import httpx, trafilatura

from mlx_lm import load
from reasoning_decoder import reason, GenConfig, FINAL_CLOSE_TOKEN

# --- Search backend (support both new 'ddgs' and legacy 'duckduckgo_search') ---
try:
    from ddgs import DDGS  # pip install ddgs
    DDGS_IMPL = "ddgs"
except Exception:
    try:
        from duckduckgo_search import DDGS  # pip install duckduckgo_search
        DDGS_IMPL = "duckduckgo_search"
    except Exception:
        DDGS = None
        DDGS_IMPL = None

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 Safari/605.1.15"}

def web_search(query: str, k: int = 5) -> List[Tuple[str, str]]:
    """Return [(title, url)] using DuckDuckGo. Works for both ddgs libs."""
    if DDGS is None:
        raise RuntimeError("Search backend missing. Install: pip install ddgs  (or)  pip install duckduckgo_search")
    out: List[Tuple[str, str]] = []
    # Both libs expose .text(); often returns an iterator/generator
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=k, region="uk-en", safesearch="moderate")
        for r in results or []:
            # Different keys across versions:
            url = r.get("href") or r.get("link") or r.get("url")
            title = r.get("title") or r.get("heading") or r.get("source") or (url or "result")
            if url:
                out.append((title, url))
    return out

# --- Fetch & extract clean text ---
async def fetch_one(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, headers=UA, timeout=12)
        r.raise_for_status()
        txt = trafilatura.extract(
            r.text,
            include_comments=False,
            include_tables=False,
            favor_precision=True
        ) or ""
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return ""

async def fetch_all(urls: List[str]) -> List[str]:
    async with httpx.AsyncClient(follow_redirects=True, http2=True, timeout=15) as client:
        return await asyncio.gather(*(fetch_one(client, u) for u in urls))

def build_context(docs: List[Tuple[str, str, str]], max_chars: int = 6000) -> Tuple[str, List[Tuple[str, str]]]:
    packed, used, keep = [], 0, []
    for title, url, text in docs:
        if not text:
            continue
        snippet = text[:1200]
        block = f"[{title}]({url}): {snippet}\n\n"
        if used + len(block) > max_chars:
            break
        packed.append(block)
        keep.append((title, url))
        used += len(block)
    return "".join(packed), keep

def augment_question(question: str, context: str) -> str:
    if not context.strip():
        return question  # no context available; ask directly
    return textwrap.dedent(f"""
    Use the CONTEXT below to answer the QUESTION. If the context doesn't contain the answer, say you don't know.
    Cite sources by name inline when useful.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """).strip()

def resolve_model(default_repo="meta-llama/Llama-3.2-1B-Instruct") -> tuple[str, str]:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+", help="Your query")
    p.add_argument("--model", help="HF repo id OR local path (defaults to model_path.txt if present)")
    args = p.parse_args()
    q = " ".join(args.question)

    # priority: --model > $MODEL_PATH > model_path.txt > default_repo
    if args.model:
        m = args.model
    elif os.environ.get("MODEL_PATH"):
        m = os.environ["MODEL_PATH"]
    elif os.path.exists("model_path.txt"):
        m = open("model_path.txt").read().strip()
    else:
        m = default_repo
    return m, q

def answer_with_search(model_id_or_path: str, question: str) -> str:
    hits = web_search(question, k=6)
    if not hits:
        # fall back to model-only answer
        model, tok = load(model_id_or_path)
        return reason(
            model,
            tok,
            question,
            gen_cfg=GenConfig(max_new_tokens=320, temperature=0.5, top_p=0.95, stop=(FINAL_CLOSE_TOKEN,)),
        )

    urls = [u for _, u in hits]
    texts = asyncio.run(fetch_all(urls))
    docs = [(t, u, x) for (t, u), x in zip(hits, texts)]
    context, kept = build_context(docs, max_chars=6000)

    model, tok = load(model_id_or_path)
    augmented = augment_question(question, context)
    final = reason(
        model,
        tok,
        augmented,
        gen_cfg=GenConfig(max_new_tokens=320, temperature=0.4, top_p=0.95, stop=(FINAL_CLOSE_TOKEN,)),
    )
    sources = "\n".join(f"- {title} — {url}" for title, url in kept[:5])
    return f"{final}\n\nSources:\n{sources}" if kept else final

if __name__ == "__main__":
    model_id, q = resolve_model()
    print(f"Using model: {model_id}")
    print(f"Search backend: {DDGS_IMPL or 'NONE'}")
    print(answer_with_search(model_id, q))