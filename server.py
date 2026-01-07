#!/usr/bin/env python3
# Local ChatGPT-like UI for MLX + reasoning + optional web search (WebSocket “streaming”)
import os, re, asyncio
from typing import List, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx, trafilatura

from mlx_lm import load
from reasoning_decoder import reason, GenConfig, FINAL_CLOSE_TOKEN, ReasoningConfig

# ---- Search backend (ddgs or duckduckgo_search) --------------------------------
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

def resolve_model_path() -> str:
    env = os.environ.get("MODEL_PATH")
    if env: return env
    if os.path.exists("model_path.txt"):
        return open("model_path.txt").read().strip()
    return "mlx-community/Qwen2.5-7B-Instruct-4bit"

MODEL_ID = resolve_model_path()
MODEL, TOK = load(MODEL_ID)

# ---- Web text fetch/search ------------------------------------------------------
async def fetch_one(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        txt = trafilatura.extract(r.text, include_comments=False, include_tables=False, favor_precision=True) or ""
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return ""

async def fetch_all(urls: List[str]) -> List[str]:
    async with httpx.AsyncClient(follow_redirects=True, http2=True, timeout=20) as client:
        return await asyncio.gather(*(fetch_one(client, u) for u in urls))

def web_search(query: str, k: int = 6, region: str = "uk-en") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if DDGS is None:
        return out
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=k, region=region, safesearch="moderate")
        for r in results or []:
            url = r.get("href") or r.get("link") or r.get("url")
            title = r.get("title") or r.get("heading") or r.get("source") or (url or "result")
            if url: out.append((title, url))
    return out

def build_context(docs: List[Tuple[str, str, str]], max_chars: int = 4800):
    packed, used, kept = [], 0, []
    for title, url, text in docs[:4]:  # top 4 docs to keep it snappy
        if not text: continue
        snippet = text[:900]
        block = f"[{title}]({url}): {snippet}\n\n"
        if used + len(block) > max_chars: break
        packed.append(block); kept.append((title, url)); used += len(block)
    return "".join(packed), kept

def augment_question(q: str, context: str) -> str:
    if not context.strip(): return q
    return (
        "Use the CONTEXT below to answer the QUESTION. "
        "If the context doesn't contain the answer, say you don't know.\n"
        "Cite sources inline when useful.\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{q}"
    ).strip()

# ---- FastAPI app ---------------------------------------------------------------
app = FastAPI(title="Local Reasoning UI", version="0.4")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/model_path.txt")
def model_path_file():
    if os.path.exists("model_path.txt"):
        return FileResponse("model_path.txt", media_type="text/plain")
    return JSONResponse({"detail": "Not found"}, status_code=404)

class ChatReq(BaseModel):
    message: str
    use_web: bool = True
    use_reasoning: bool = True
    temperature: float = 0.5
    top_p: float = 0.95
    max_new_tokens: int = 320
    # reasoning knobs (when enabled)
    k: int = 3
    reflect: bool = True
    region: str = "uk-en"

def run_reasoning(prompt: str, *, use_reasoning: bool, max_new: int, temp: float, top_p: float, k: int, reflect: bool) -> str:
    gen = GenConfig(max_new_tokens=max_new, temperature=temp, top_p=top_p, stop=(FINAL_CLOSE_TOKEN,))
    r = ReasoningConfig(k=k, reflect=reflect) if use_reasoning else ReasoningConfig(k=1, reflect=False)
    return reason(MODEL, TOK, prompt, gen_cfg=gen, r_cfg=r)

@app.post("/api/chat")
def api_chat(req: ChatReq):
    """Non-streaming endpoint (kept for quick tests)."""
    try:
        sources = []
        context = ""
        if req.use_web and DDGS is not None:
            hits = web_search(req.message, k=6, region=req.region)
            texts = asyncio.run(fetch_all([u for _, u in hits]))
            docs = [(t, u, x) for (t, u), x in zip(hits, texts)]
            context, kept = build_context(docs)
            sources = [{"title": t, "url": u} for t, u in kept[:5]]
        prompt = augment_question(req.message, context)
        answer = run_reasoning(prompt,
                               use_reasoning=req.use_reasoning,
                               max_new=req.max_new_tokens,
                               temp=req.temperature,
                               top_p=req.top_p,
                               k=req.k,
                               reflect=req.reflect)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return JSONResponse({"answer": f"Error: {e}", "sources": []}, status_code=500)

# ---- WebSocket for streaming-like UX ------------------------------------------
@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        payload = await ws.receive_json()
        msg = (payload.get("message") or "").strip()
        use_web = bool(payload.get("use_web", True))
        use_reasoning = bool(payload.get("use_reasoning", True))
        temperature = float(payload.get("temperature", 0.5))
        top_p = float(payload.get("top_p", 0.95))
        max_new = int(payload.get("max_new_tokens", 320))
        region = payload.get("region", "uk-en")
        k = int(payload.get("k", 3))
        reflect = bool(payload.get("reflect", True))

        if not msg:
            await ws.send_json({"type": "error", "data": "Empty message"})
            await ws.close(); return

        # Optional search
        sources = []; context = ""
        if use_web and DDGS is not None:
            await ws.send_json({"type": "status", "data": "Searching…"})
            hits = await asyncio.to_thread(web_search, msg, 6, region)
            await ws.send_json({"type": "status", "data": "Fetching sources…"})
            texts = await fetch_all([u for _, u in hits])
            docs = [(t, u, x) for (t, u), x in zip(hits, texts)]
            context, kept = build_context(docs)
            sources = [{"title": t, "url": u} for t, u in kept[:5] if t and u]

        # Reason/generate (computed in a thread)
        await ws.send_json({"type": "status", "data": "Reasoning…" if use_reasoning else "Generating…"})
        prompt = augment_question(msg, context)
        answer = await asyncio.to_thread(
            run_reasoning, prompt,
            use_reasoning=use_reasoning, max_new=max_new, temp=temperature, top_p=top_p, k=k, reflect=reflect
        )

        # Chunk out the final answer for a “streaming” feel
        await ws.send_json({"type": "start"})
        chunk = 36
        for i in range(0, len(answer), chunk):
            await ws.send_json({"type": "answer", "delta": answer[i:i+chunk]})
            await asyncio.sleep(0.01)
        await ws.send_json({"type": "done", "sources": sources})
    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_json({"type": "error", "data": f"{e}"})
    finally:
        await ws.close()