"""Related-work drafter (command line + engine for the web app).

    python related_work.py "noisy knowledge graph refinement with LLMs"
"""
import json
import re
import sys
import time

import ollama

import config
from tools import _arxiv_search, _keywords, search_arxiv

MODEL = config.MODEL
OPTS = {"temperature": 0.2, "num_ctx": 4096}
MAX_PAPERS = 10
RANK = {"high": 0, "partial": 1, "low": 2}


def get_queries(text, model):
    prompt = (
        "Give 3 arXiv search phrases (2-4 words each) that would find related work "
        "for the text below. Spell out abbreviations: write 'knowledge graph', not "
        "'KG'; 'large language models', not 'LLMs'. Make the 3 phrases different from "
        'each other. Reply as JSON: {"queries": ["...", "...", "..."]}\n\nText:\n' + text[:1500])
    try:
        r = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                        format="json", options=OPTS, keep_alive=config.KEEP_ALIVE)
        qs = json.loads(r.message.content).get("queries", [])
        qs = [str(q).strip() for q in qs if str(q).strip()][:3]
    except Exception:
        qs = []
    return qs or [" ".join(text.split()[:5])]


def _add(res, label, papers, seen, log):
    if not isinstance(res, dict):
        log.append(f"{label}: unexpected result")
        return
    if "error" in res:
        log.append(f"{label}: {res['error']}")
        return
    n = 0
    for p in res.get("papers", []):
        key = p.get("id") or p["title"]
        if key not in seen:
            seen.add(key)
            papers.append(p)
            n += 1
    log.append(f"{label}: {n} new papers ({res.get('match_type', '?')} match)")


def references(papers):
    lines = []
    for i, p in enumerate(papers, 1):
        au = ", ".join(p.get("authors", [])[:3]) + (" et al." if p.get("authors_total", 0) > 3 else "")
        lines.append(f"[{i}] {au} ({p['published'][:4]}). {p['title']}. {p['link']}")
    return "\n".join(lines)


def write_stream(text, papers, model):
    listing = "\n".join(
        f"[{i}] {p['title']} ({p['published'][:4]}): {p['summary'][:250]}"
        for i, p in enumerate(papers, 1))
    system = (
        "You write the related-work section of a research paper. Use ONLY the numbered "
        "papers given. Group them by theme, cite them as [1], [2], and base every claim "
        "on the abstract snippet. Do not invent papers, results or numbers. Write "
        "150-200 words of plain prose, no headings.")
    user = f"My work:\n{text[:1500]}\n\nPapers:\n{listing}\n\nWrite the related-work paragraph."
    for chunk in ollama.chat(model=model, stream=True, keep_alive=config.KEEP_ALIVE,
                             options={**OPTS, "num_predict": 450},
                             messages=[{"role": "system", "content": system},
                                       {"role": "user", "content": user}]):
        piece = chunk.message.content or ""
        if piece:
            yield piece


def run_draft(text, model=None, papers=None):
    """Yield events: status, queries, papers, token, done, error."""
    model = model or MODEL
    t0 = time.time()
    log = []

    if papers is None:
        yield {"type": "status", "text": "Planning searches…"}
        try:
            queries = get_queries(text, model)
        except Exception as e:
            yield {"type": "error", "text": f"{type(e).__name__}: {e}. Is Ollama running?"}
            return
        yield {"type": "queries", "queries": queries}

        papers, seen = [], set()
        for q in queries:
            yield {"type": "status", "text": f"Searching arXiv: {q}"}
            _add(search_arxiv(q, 5, "relevance"), q, papers, seen, log)

        if not papers:  # fallback: keywords from the user's own text
            words = _keywords(text, 4)[:3]
            yield {"type": "status", "text": "Trying a broader search…"}
            try:
                found, _ = _arxiv_search(" AND ".join(f"abs:{w}" for w in words), 8, "relevance", words)
                _add({"papers": found, "match_type": "fallback"}, "keywords: " + " ".join(words), papers, seen, log)
            except Exception as e:
                log.append(f"fallback failed: {e}")

        papers.sort(key=lambda p: RANK.get(p.get("relevance"), 1))
        strong = [p for p in papers if p.get("relevance") != "low"]
        if len(strong) >= 4:
            papers = strong
        papers = papers[:MAX_PAPERS]
    else:
        papers = papers[:15]

    yield {"type": "papers", "papers": papers, "log": log}
    if not papers:
        yield {"type": "error", "text": "No papers found. See the search log above, or try different wording."}
        return

    yield {"type": "status", "text": f"Writing from {len(papers)} papers…"}
    section = ""
    try:
        for piece in write_stream(text, papers, model):
            section += piece
            yield {"type": "token", "text": piece}
    except Exception as e:
        yield {"type": "error", "text": f"{type(e).__name__}: {e}"}
        return

    section = section.strip()
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", section)}
    yield {"type": "done", "section": section,
           "bad": sorted(n for n in cited if n < 1 or n > len(papers)),
           "unused": [i for i in range(1, len(papers) + 1) if i not in cited],
           "references": references(papers),
           "seconds": round(time.time() - t0, 1)}


def main():
    text = " ".join(sys.argv[1:]).strip() or input("Paste a topic, title or abstract:\n> ").strip()
    if not text:
        return
    for ev in run_draft(text):
        t = ev["type"]
        if t == "status":
            print(" ", ev["text"], flush=True)
        elif t == "papers":
            for line in ev["log"]:
                print("   log:", line)
        elif t == "token":
            print(ev["text"], end="", flush=True)
        elif t == "error":
            print("Error:", ev["text"])
        elif t == "done":
            print("\n\nReferences:\n" + ev["references"])
            if ev["bad"]:
                print("WARNING: cites missing references:", ev["bad"])
            if ev["unused"]:
                print("Found but not cited:", ev["unused"])
            fname = f"related_work_{time.strftime('%Y%m%d_%H%M')}.md"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"## Related Work (draft)\n\n{ev['section']}\n\n### References\n\n{ev['references']}\n")
            print(f"Saved to {fname} ({ev['seconds']}s)")


if __name__ == "__main__":
    main()