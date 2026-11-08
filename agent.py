import json
import re
import sys
import time

import ollama

import config
from tools import STOPWORDS, TOOLS, TOOL_FUNCS, clean_id, slim

MODEL = config.MODEL
OPTIONS = {"temperature": 0.2, "num_ctx": 4096}

SYSTEM = (
    "You are a research assistant. For any question about papers, call a tool; "
    "never answer about papers from memory. "
    "search_arxiv: preprints on a topic (2-4 core keywords only). "
    "search_openalex: published literature with citation counts; use for 'highly cited', "
    "'influential', surveys or venues. "
    "get_paper_details: full abstract/authors of ONE paper given an arXiv ID or link. "
    "For follow-ups about papers already listed, use the ids shown in the conversation. "
    "If no tool is needed (greeting, general question), answer briefly."
)

WRITE_SYSTEM = (
    "You are a research assistant. The user already sees the full list of papers as cards. "
    "Write a SHORT overview in at most 4 sentences (under 90 words): the main themes, and "
    "which 1-2 papers look most relevant and why. Do NOT list or re-describe every paper and "
    "do not write links. Mention only papers in the results. If the match type is relaxed or "
    "broad, or many papers have relevance 'partial' or 'low', say so plainly."
)

FILLER = STOPWORDS | {
    "find", "search", "show", "give", "me", "list", "get", "please", "can", "you", "use",
    "sort", "date", "related", "about", "some", "all", "top", "highly", "cited",
    "influential", "seminal", "classic", "most", "best", "important", "is", "are",
    "newest", "latest", "recently", "newer", "up", "date",
}
RECENT_RE = re.compile(r"\b(recent|recently|newest|latest|new|202[4-9])\b", re.I)
CITED_RE = re.compile(r"\b(cited|influential|seminal|classic|survey|review|journal|conference|published)\b", re.I)
STRONG_CITED_RE = re.compile(r"\b(cited|influential|seminal|classic)\b", re.I)
FOLLOWUP_RE = re.compile(r"\b(first|second|third|fourth|fifth|last|former|latter|these|those|them|it|its|compare)\b", re.I)
CONNECT_RE = re.compile(r"\b(?:with|and|for|in|using|via|on)\b")



def route(question, history, source, recent):
    """Rule-based tool choice for fast mode. Returns [(tool, args)] or None (needs the model)."""
    q = question.strip()
    pid = clean_id(q)
    if pid:
        return [("get_paper_details", {"arxiv_id": pid})]
    if history and FOLLOWUP_RE.search(q):
        return None

    # split on connector words so "A with B" becomes two concepts: "A | B"
    segs = []
    for seg in CONNECT_RE.split(q.lower()):
        ws = [w for w in re.findall(r"[a-z0-9\-]+", seg) if w not in FILLER]
        if ws:
            segs.append(" ".join(ws[:4]))
    segs = segs[:2]
    if not segs:
        return None
    query = " | ".join(segs)

    sort = "date" if (recent or RECENT_RE.search(q)) else "relevance"
    use_oa = source == "openalex" or (source == "auto" and CITED_RE.search(q))
    if use_oa:
        if sort == "relevance" and STRONG_CITED_RE.search(q):
            sort = "cited"
        return [("search_openalex", {"query": query, "sort": sort})]
    return [("search_arxiv", {"query": query, "sort": sort})]

def _stats(total_start, t_model, t_tools, mode):
    return {"type": "stats", "total": round(time.time() - total_start, 1),
            "model": round(t_model, 1), "tools": round(t_tools, 1), "mode": mode}


def _err(e):
    hint = " Is Ollama running? Open the Ollama app." if "connect" in str(e).lower() else ""
    return {"type": "error", "text": f"{type(e).__name__}: {e}.{hint}"}


def run_agent_stream(question, history=None, model=None, mode="fast", recent=False, source="auto"):
    """Run one question and yield events (status, tool_call, tool_result, token, stats, answer, error)."""
    model = model or MODEL
    history = [{"role": h["role"], "content": h["content"]} for h in (history or [])][-8:]
    t_start = time.time()
    t_model = t_tools = 0.0
    used_mode = mode

    calls = route(question, history, source, recent) if mode == "fast" else None

    if calls is None:  # agent mode: the model chooses the tool
        used_mode = "agent"
        hint = ""
        if recent:
            hint += " Prefer recent papers (sort='date')."
        if source == "arxiv":
            hint += " Use search_arxiv."
        elif source == "openalex":
            hint += " Use search_openalex."
        yield {"type": "status", "text": "Choosing a tool…"}
        msgs = [{"role": "system", "content": SYSTEM}] + history + [
            {"role": "user", "content": question + hint}]
        t0 = time.time()
        try:
            resp = ollama.chat(model=model, messages=msgs, tools=TOOLS,
                               options=OPTIONS, keep_alive=config.KEEP_ALIVE)
        except Exception as e:
            yield _err(e)
            return
        t_model += time.time() - t0
        msg = resp.message
        if not msg.tool_calls:
            text = msg.content or ""
            yield _stats(t_start, t_model, t_tools, used_mode)
            yield {"type": "answer", "text": text, "memory": text}
            return
        calls = [(c.function.name, dict(c.function.arguments)) for c in msg.tool_calls][:3]

    # ---- run the tools
    collected = []
    for name, args in calls:
        yield {"type": "tool_call", "name": name, "args": args}
        yield {"type": "status", "text": "Searching…"}
        t0 = time.time()
        fn = TOOL_FUNCS.get(name)
        if fn is None:
            out = {"error": f"Unknown tool: {name}"}
        else:
            try:
                out = fn(**args)
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
        dt = time.time() - t0
        t_tools += dt
        yield {"type": "tool_result", "name": name, "result": out, "seconds": round(dt, 1)}
        collected.append((name, out))

    good = [(n, o) for n, o in collected if isinstance(o, dict) and "error" not in o]
    if not good:
        errs = "; ".join(str(o.get("error")) for _, o in collected if isinstance(o, dict)) or "no results"
        text = f"I could not get results: {errs}"
        yield _stats(t_start, t_model, t_tools, used_mode)
        yield {"type": "answer", "text": text, "memory": text}
        return

    # ---- short streamed overview (the cards already show the full list)
    payload = json.dumps([{"tool": n, "result": slim(o)} for n, o in good], ensure_ascii=False)
    ctx = history[-2:] if used_mode == "agent" else []
    msgs = [{"role": "system", "content": WRITE_SYSTEM}] + ctx + [{
        "role": "user",
        "content": f"Question: {question}\n\nTool results (JSON):\n{payload}\n\nWrite the short overview now."}]
    yield {"type": "status", "text": "Writing overview…"}
    text = ""
    t0 = time.time()
    try:
        for chunk in ollama.chat(model=model, messages=msgs, stream=True,
                                 options={**OPTIONS, "num_predict": 200},
                                 keep_alive=config.KEEP_ALIVE):
            piece = chunk.message.content or ""
            if piece:
                text += piece
                yield {"type": "token", "text": piece}
    except Exception as e:
        yield _err(e)
        return
    t_model += time.time() - t0

    # what the model should remember next turn (so "the second paper" works)
    lines = []
    for _, o in good:
        if "papers" in o:
            for i, p in enumerate(o["papers"][:8], 1):
                lines.append(f"{i}. {p['title']} [{p.get('id') or p.get('link')}] ({p.get('published', '')})")
        elif "title" in o:
            lines.append(f"Details shown: {o['title']} [{o.get('id')}]")
    memory = text + ("\n\nPapers shown to the user:\n" + "\n".join(lines) if lines else "")

    yield _stats(t_start, t_model, t_tools, used_mode)
    yield {"type": "answer", "text": text, "memory": memory}


# ------------------------------------------------------------ command line
def _print_result(out):
    if not isinstance(out, dict):
        return
    if "error" in out:
        print("   error:", out["error"])
    elif "papers" in out:
        print(f"   [{out['match_type']}]")
        for i, p in enumerate(out["papers"], 1):
            print(f"   {i}. {p['title']} ({p['published']})\n      {p['link']}")
    elif "title" in out:
        print(f"   {out['title']}\n   {', '.join(out.get('authors', []))}\n   {out['link']}")


def ask(question, history=None, mode="fast"):
    memory = ""
    for ev in run_agent_stream(question, history, mode=mode):
        t = ev["type"]
        if t == "tool_call":
            print(f"  [tool] {ev['name']}({ev['args']})")
        elif t == "tool_result":
            _print_result(ev["result"])
            print("\nOverview: ", end="", flush=True)
        elif t == "token":
            print(ev["text"], end="", flush=True)
        elif t == "stats":
            print(f"\n  ({ev['total']}s total, model {ev['model']}s, search {ev['tools']}s, {ev['mode']})")
        elif t == "answer":
            memory = ev["memory"]
        elif t == "error":
            print("Error:", ev["text"])
    return memory


def main():
    args = [a for a in sys.argv[1:] if a != "--agent"]
    mode = "agent" if "--agent" in sys.argv else "fast"
    if args:
        ask(" ".join(args), mode=mode)
        return
    history = []
    print(f"Research agent ready ({mode} mode). Type a question (or 'quit').")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            m = ask(q, history, mode)
            if m:
                history += [{"role": "user", "content": q}, {"role": "assistant", "content": m}]
                history = history[-10:]


if __name__ == "__main__":
    main()