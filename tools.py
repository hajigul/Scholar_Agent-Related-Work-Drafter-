import hashlib
import json
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
import urllib.error


import config

NS = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_API = "https://export.arxiv.org/api/query?"
OPENALEX_API = "https://api.openalex.org/works?"
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

STOPWORDS = {
    "a", "an", "the", "of", "for", "with", "on", "in", "and", "or", "to",
    "using", "based", "recent", "new", "latest", "paper", "papers", "work",
}

# new-style (2405.17249) and old-style (cs.LG/0701001) arXiv IDs, version stripped
ID_RE = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?")

MATCH_TEXT = {
    "exact": "exact phrase in title or abstract",
    "relaxed": "RELAXED match: the last keyword was dropped because the exact phrase had no results",
    "broad": "BROAD keyword match; results may be only loosely relevant",
    "openalex": "OpenAlex relevance search",
    "fallback": "FALLBACK: OpenAlex was unavailable, so these are arXiv preprints without citation counts",
}

_lock = threading.Lock()
_last_arxiv = 0.0


def clean_id(text):
    m = ID_RE.search(str(text))
    return m.group(1) if m else None



def http_get(url, pace=False):
    """Cached GET. Returns (bytes, was_cached). arXiv calls are paced 3s apart.
    Retries up to 2 times on HTTP 429 (rate limit)."""
    global _last_arxiv
    f = CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".bin")
    if f.exists() and time.time() - f.stat().st_mtime < config.CACHE_TTL:
        return f.read_bytes(), True
    if pace:
        with _lock:
            wait = 3.0 - (time.time() - _last_arxiv)
            if wait > 0:
                time.sleep(wait)
            _last_arxiv = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "ScholarAgent/2.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                try:
                    delay = min(float(e.headers.get("Retry-After", 2)), 5)
                except (TypeError, ValueError):
                    delay = 2
                time.sleep(delay)
                continue
            raise
    f.write_bytes(data)
    return data, False


def _keywords(query, limit=4):
    words = re.findall(r"[a-z0-9\-]+", str(query).lower())
    return [w for w in words if w not in STOPWORDS][:limit]


def _stem(w):
    return w[: max(4, len(w) - 2)]


def _relevance(title, summary, words):
    text = f"{title} {summary}".lower()
    hit = sum(_stem(w) in text for w in words)
    ratio = hit / max(len(words), 1)
    return "high" if ratio == 1 else "partial" if ratio >= 0.5 else "low"


def _norm_sort(sort):
    s = str(sort).lower()
    if s in ("date", "recent", "newest", "latest"):
        return "date"
    if s in ("cited", "citations", "most_cited"):
        return "cited"
    return "relevance"


def _n(max_results):
    try:
        return max(1, min(int(max_results), 8))
    except (TypeError, ValueError):
        return 5


def _pack(kind, papers, cached):
    return {
        "match_type": kind,
        "match": MATCH_TEXT[kind],
        "cached": cached,
        "note": "Order is final. Judge fit using each paper's 'relevance' field.",
        "papers": papers,
    }


# ------------------------------------------------------------------ arXiv
def _arxiv_paper(e, words=None, chars=350):
    link = e.findtext("a:id", default="", namespaces=NS).strip()
    title = " ".join(e.findtext("a:title", default="", namespaces=NS).split())
    full = " ".join(e.findtext("a:summary", default="", namespaces=NS).split())
    authors = [
        " ".join(a.findtext("a:name", default="", namespaces=NS).split())
        for a in e.findall("a:author", NS)
    ]
    p = {
        "source": "arxiv",
        "id": clean_id(link),
        "title": title,
        "summary": full[:chars],
        "published": e.findtext("a:published", default="", namespaces=NS)[:10],
        "link": link,
        "authors": authors[:10],
        "authors_total": len(authors),
    }
    if words:
        p["relevance"] = _relevance(title, full, words)
    return p


def _arxiv_search(search, n, sort_by, words):
    url = (
        ARXIV_API + "search_query=" + urllib.parse.quote(search)
        + f"&max_results={n}&sortBy={sort_by}&sortOrder=descending"
    )
    data, cached = http_get(url, pace=True)
    root = ET.fromstring(data)
    return [_arxiv_paper(e, words) for e in root.findall("a:entry", NS)], cached




def search_arxiv(query="", max_results=5, sort="relevance", **_):
    """Search arXiv. 'A | B' means both concepts must appear; falls back step by step."""
    n = _n(max_results)
    sort_by = "submittedDate" if _norm_sort(sort) == "date" else "relevance"
    phrases = [" ".join(_keywords(p, 4)) for p in str(query).split("|")]
    phrases = [p for p in phrases if p][:2]
    words = _keywords(str(query).replace("|", " "), 5)
    if not words:
        return {"error": "Empty query. Give a short topic phrase."}

    attempts = []
    if len(phrases) == 2:  # both concepts as exact phrases
        both = " AND ".join(f'(ti:"{p}" OR abs:"{p}")' for p in phrases)
        attempts.append(("exact", both))
    flat = " ".join(words)
    attempts.append(("exact", f'(ti:"{flat}" OR abs:"{flat}")'))
    if len(words) >= 3:
        rel = " ".join(words[:-1])
        attempts.append(("relaxed", f'(ti:"{rel}" OR abs:"{rel}")'))
    attempts.append(("broad", " AND ".join(f"abs:{w}" for w in words)))

    try:
        for kind, search in attempts:
            papers, cached = _arxiv_search(search, n, sort_by, words)
            if papers:
                return _pack(kind, papers, cached)
    except Exception as e:
        return {"error": f"arXiv request failed: {e}"}
    return {"error": "No papers found. Try a different topic phrase."}

def get_paper_details(arxiv_id="", id="", url="", **_):
    """Full abstract, authors and categories for ONE arXiv paper."""
    pid = clean_id(arxiv_id or id or url)
    if not pid:
        return {"error": "No arXiv ID found. Give an ID like 2405.17249 or an arxiv.org link."}
    try:
        data, cached = http_get(ARXIV_API + "id_list=" + urllib.parse.quote(pid), pace=True)
        root = ET.fromstring(data)
    except Exception as e:
        return {"error": f"arXiv request failed: {e}"}

    entries = root.findall("a:entry", NS)
    if not entries:
        return {"error": f"No arXiv paper found for ID {pid}."}
    e = entries[0]
    p = _arxiv_paper(e, chars=2000)
    if not p["title"] or p["title"].lower() == "error":
        return {"error": f"No arXiv paper found for ID {pid}."}

    authors = [
        " ".join(a.findtext("a:name", default="", namespaces=NS).split())
        for a in e.findall("a:author", NS)
    ]
    pdf = ""
    for l in e.findall("a:link", NS):
        if l.get("title") == "pdf":
            pdf = l.get("href", "")
    p.update({
        "authors": authors[:12],
        "authors_total": len(authors),
        "categories": [c.get("term") for c in e.findall("a:category", NS)],
        "updated": e.findtext("a:updated", default="", namespaces=NS)[:10],
        "pdf_link": pdf,
        "cached": cached,
    })
    return p


# --------------------------------------------------------------- OpenAlex
def _abstract(inv):
    """OpenAlex stores abstracts as an inverted index; rebuild the text."""
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def _oa_paper(w, words):
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    doi = w.get("doi") or ""
    authors = [
        (a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])
    ]
    title = w.get("display_name") or ""
    full = _abstract(w.get("abstract_inverted_index"))
    return {
        "source": "openalex",
        "id": doi.replace("https://doi.org/", "") or w.get("id", ""),
        "title": title,
        "summary": full[:350],
        "published": w.get("publication_date") or str(w.get("publication_year") or ""),
        "link": doi or loc.get("landing_page_url") or w.get("id", ""),
        "authors": authors[:10],
        "authors_total": len(authors),
        "venue": src.get("display_name") or "",
        "citations": w.get("cited_by_count", 0),
        "pdf_link": (w.get("open_access") or {}).get("oa_url") or "",
        "relevance": _relevance(title, full, words),
    }



def search_openalex(query="", max_results=5, sort="relevance", **_):
    """Search published literature (all venues) with citation counts.
    For sort='cited': fetch a relevance-ranked pool, keep papers that really
    contain the keywords, then sort that pool by citations in Python.
    Falls back to arXiv if OpenAlex is unavailable."""
    n = _n(max_results)
    s = _norm_sort(sort)
    words = _keywords(str(query).replace("|", " "), 6)
    if not words:
        return {"error": "Empty query. Give a short topic phrase."}

    params = {
        "search": " ".join(words),
        "per-page": 50 if s == "cited" else n,
        "select": "id,doi,display_name,publication_year,publication_date,cited_by_count,"
                  "authorships,primary_location,open_access,abstract_inverted_index",
    }
    if s == "date":  # recent = last 2 years, still ranked by relevance
        params["filter"] = "from_publication_date:" + (date.today() - timedelta(days=730)).isoformat()
    if config.CONTACT_EMAIL:
        params["mailto"] = config.CONTACT_EMAIL

    try:
        data, cached = http_get(OPENALEX_API + urllib.parse.urlencode(params))
        papers = [_oa_paper(w, words) for w in json.loads(data).get("results", [])]
        if s == "cited":
            papers = [p for p in papers if p.get("relevance") == "high"]
            papers.sort(key=lambda p: p.get("citations") or 0, reverse=True)
            papers = papers[:n]
        if papers:
            return _pack("openalex", papers, cached)
        return {"error": "No papers found. Try a different topic phrase."}
    except Exception as e:
        fb = search_arxiv(query, max_results, "date" if s == "date" else "relevance")
        if isinstance(fb, dict) and "papers" in fb:
            fb["match_type"] = "fallback"
            fb["match"] = MATCH_TEXT["fallback"] + f" ({type(e).__name__}: {e})"
            return fb
        return {"error": f"OpenAlex request failed: {e}"}


# ------------------------------------------- compact view sent to the model
def slim(result):
    """Short version of a tool result: smaller prompt means a faster model."""
    if isinstance(result, dict) and "papers" in result:
        out = dict(result)
        out["papers"] = [
            {k: p[k] for k in ("title", "published", "relevance", "citations", "venue")
             if p.get(k) not in (None, "")}
            | {"summary": p.get("summary", "")[:220]}
            for p in result["papers"]
        ]
        return out
    return result


# ------------------------------------------------------ what the model sees
TOOLS = [
    {"type": "function", "function": {
        "name": "search_arxiv",
        "description": (
            "Search arXiv preprints on a topic. Query = the core topic phrase only "
            "(2-4 words), e.g. 'knowledge graph completion'. If the topic combines two "
            "concepts, separate them with ' | ', e.g. 'retrieval augmented generation | knowledge graph'. "
            "No filler words."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Core topic phrase, 2-4 words"},
            "max_results": {"type": "integer", "description": "Number of papers (default 5)"},
            "sort": {"type": "string", "enum": ["relevance", "date"],
                     "description": "'date' for recent or newest work"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "search_openalex",
        "description": (
            "Search published scholarly literature across all venues, with citation counts. "
            "Use for 'highly cited', 'influential', surveys, or papers not only on arXiv."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Core topic phrase, 2-4 words"},
            "max_results": {"type": "integer", "description": "Number of papers (default 5)"},
            "sort": {"type": "string", "enum": ["relevance", "date", "cited"],
                     "description": "'date' = last 2 years, 'cited' = most cited first"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "get_paper_details",
        "description": (
            "Full abstract, authors, categories and PDF link of ONE arXiv paper. Use when the "
            "user gives an arXiv ID or link, or asks about one paper from an earlier list."),
        "parameters": {"type": "object", "properties": {
            "arxiv_id": {"type": "string", "description": "arXiv ID like 2405.17249, or an arxiv.org link"},
        }, "required": ["arxiv_id"]},
    }},
]

TOOL_FUNCS = {
    "search_arxiv": search_arxiv,
    "search_openalex": search_openalex,
    "get_paper_details": get_paper_details,
}