"""Evaluate ScholarAgent.

    python eval.py                        # fast mode
    python eval.py --mode agent           # model picks the tools
    python eval.py --mode both            # compare fast vs agent
    python eval.py --model llama3.1:8b    # another model
    python eval.py --only S1,D1           # selected cases
    python eval.py --limit 4              # first 4 cases (quick test)
    python eval.py --cache                # allow cached searches (timings less honest)
    python eval.py --score FILE.csv       # score the relevance labels you filled in
"""
import argparse
import csv
import json
import re
import statistics
import time

import config
from agent import MODEL, run_agent_stream

ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
PRECISION_PASS = 0.6

# groups: a paper matches if EVERY group has at least one alternative in its title+abstract snippet
CASES = [
    {"id": "S1", "kind": "search", "recent": True,
     "question": "Find recent papers on knowledge graph completion.",
     "groups": [["knowledge graph"], ["completion"]]},
    {"id": "S2", "kind": "search", "recent": False,
     "question": "Find papers on link prediction in knowledge graphs.",
     "groups": [["link prediction"], ["knowledge graph"]]},
    {"id": "S3", "kind": "search", "recent": True,
     "question": "Find the newest papers on retrieval augmented generation with knowledge graphs.",
     "groups": [["retrieval"], ["knowledge graph", "graph"]]},
    {"id": "S4", "kind": "search", "recent": False,
     "question": "Find papers on knowledge graph embedding.",
     "groups": [["knowledge graph"], ["embedding"]]},
    {"id": "S5", "kind": "search", "recent": False,
     "question": "Find papers on protein-protein interaction prediction.",
     "groups": [["protein"], ["interaction"]]},
    {"id": "S6", "kind": "search", "recent": False,
     "question": "Find papers on drug target prediction.",
     "groups": [["drug"], ["target"]], "expect": ["mucos"]},
    {"id": "S7", "kind": "search", "recent": True,
     "question": "Find recent papers on noisy knowledge graph refinement.",
     "groups": [["knowledge graph"], ["noise", "noisy", "error", "quality", "refine"]]},
    {"id": "S8", "kind": "search", "recent": False,
     "question": "Find papers on community detection in complex networks.",
     "groups": [["community detection"], ["network"]]},
    {"id": "S9", "kind": "search", "recent": False, "tool": "search_openalex",
     "question": "Highly cited papers on link prediction.",
     "groups": [["link prediction", "link-prediction"]]},
    {"id": "D1", "kind": "detail",
     "question": "Give me the full abstract and authors of arXiv paper 1706.03762.",
     "groups": [["attention"]]},
    {"id": "D2", "kind": "detail",
     "question": "What are the authors and categories of arXiv:1810.04805?",
     "groups": [["bert", "bidirectional"]]},
    {"id": "D3", "kind": "detail",
     "question": "Summarize the paper at https://arxiv.org/abs/2405.17249",
     "groups": [["completion"]]},
    {"id": "D4", "kind": "detail",
     "question": "Who wrote arXiv 2306.08302 and what is it about?",
     "groups": [["knowledge graph"]]},
]
DEFAULT_TOOL = {"search": "search_arxiv", "detail": "get_paper_details"}


def matches(text, groups):
    text = text.lower()
    return all(any(alt in text for alt in g) for g in groups)


def paper_matches(p, groups):
    return matches(p.get("title", "") + " " + p.get("summary", ""), groups)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def run_case(case, model, mode):
    t0 = time.time()
    t_first = None
    calls, results, answer, error, stats = [], [], "", "", {}
    for ev in run_agent_stream(case["question"], model=model, mode=mode,
                               recent=case.get("recent", False)):
        t = ev["type"]
        if t == "tool_call":
            calls.append(ev)
        elif t == "tool_result":
            results.append(ev)
            if t_first is None:
                t_first = round(time.time() - t0, 1)
        elif t == "answer":
            answer = ev["text"]
        elif t == "error":
            error = ev["text"]
        elif t == "stats":
            stats = ev
    latency = round(time.time() - t0, 1)

    papers, tool_error = [], False
    for r in results:
        res = r["result"]
        if not isinstance(res, dict):
            continue
        if "error" in res:
            tool_error = True
        elif "papers" in res:
            papers += res["papers"]
        elif "title" in res:
            papers.append(res)
    ids = {p["id"] for p in papers if p.get("id")}

    answer_ids = set(ID_RE.findall(answer))
    hallucinated = sorted(answer_ids - ids)
    want_tool = case.get("tool") or DEFAULT_TOOL[case["kind"]]
    called = any(c["name"] == want_tool for c in calls)

    row = {
        "id": case["id"], "mode": mode, "kind": case["kind"], "question": case["question"],
        "called_tool": called, "n_tool_calls": len(calls), "tool_error": tool_error,
        "n_papers": len(papers), "fake_ids": len(hallucinated),
        "first_result_s": t_first, "latency_s": latency,
        "model_s": stats.get("model"), "tools_s": stats.get("tools"),
        "error": error, "tool_precision": None, "sort_ok": None, "found_expected": None,
        "correct_paper": None,
    }

    if case["kind"] == "search":
        if papers:
            row["tool_precision"] = round(sum(paper_matches(p, case["groups"]) for p in papers) / len(papers), 3)
        sorts = [str(c["args"].get("sort", "")).lower() for c in calls]
        row["sort_ok"] = (not case.get("recent")) or any(s in ("date", "recent", "newest") for s in sorts)
        if case.get("expect"):
            row["found_expected"] = any(any(s in p.get("title", "").lower() for s in case["expect"]) for p in papers)
        row["passed"] = bool(
            called and not error and row["tool_precision"] is not None
            and row["tool_precision"] >= PRECISION_PASS and row["sort_ok"]
            and row["found_expected"] is not False and not hallucinated)
    else:
        row["correct_paper"] = any(paper_matches(p, case["groups"]) for p in papers)
        row["passed"] = bool(called and not error and not tool_error
                             and row["correct_paper"] and not hallucinated)

    row["_answer"] = answer
    row["_calls"] = [{"name": c["name"], "args": c["args"]} for c in calls]
    row["_papers"] = [{"title": p.get("title", ""), "link": p.get("link", "")} for p in papers]
    return row


def summarize(rows, model, mode):
    s = [r for r in rows if r["kind"] == "search"]
    n = len(rows)
    lat = [r["latency_s"] for r in rows]
    fr = [r["first_result_s"] for r in rows if r["first_result_s"] is not None]
    exp = [r["found_expected"] for r in rows if r["found_expected"] is not None]
    print("\n" + "=" * 62)
    print(f"RESULTS  model={model}  mode={mode}  cases={n}")
    print("=" * 62)
    print(f"Pass rate:              {sum(r['passed'] for r in rows)}/{n}")
    print(f"Right tool called:      {sum(r['called_tool'] for r in rows)}/{n}")
    print(f"Cases with fake IDs:    {sum(r['fake_ids'] > 0 for r in rows)}/{n}")
    print(f"Mean tool precision:    {mean(r['tool_precision'] for r in s)}   (search cases, keyword proxy)")
    print(f"Sort=date when needed:  {sum(bool(r['sort_ok']) for r in s)}/{len(s)}")
    if exp:
        print(f"Expected paper found:   {sum(exp)}/{len(exp)}")
    print(f"First results (mean):   {round(statistics.mean(fr), 1) if fr else None}s")
    print(f"Total latency mean/med: {round(statistics.mean(lat), 1)}s / {round(statistics.median(lat), 1)}s")
    print("=" * 62)


def write_labels(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "mode", "rank", "title", "link", "relevant"])
        for r in rows:
            if r["kind"] != "search":
                continue
            for i, p in enumerate(r["_papers"], 1):
                w.writerow([r["id"], r["mode"], i, p["title"], p["link"], ""])


def score_labels(path):
    groups = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r.get("relevant") or "").strip()
            if v in ("0", "1"):
                groups.setdefault((r["case_id"], r["mode"]), []).append(int(v))
    if not groups:
        print("No labels found. Put 1 (relevant) or 0 (not relevant) in the 'relevant' column.")
        return
    print(f"{'case':<6}{'mode':<7}{'P@k':>6}   n")
    vals = {}
    for (cid, mode), v in sorted(groups.items()):
        p = sum(v) / len(v)
        vals.setdefault(mode, []).append(p)
        print(f"{cid:<6}{mode:<7}{p:>6.2f}   {len(v)}")
    for mode, v in vals.items():
        print(f"Mean precision@k ({mode}): {statistics.mean(v):.3f} over {len(v)} cases")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--mode", default="fast", choices=["fast", "agent", "both"])
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--score", default="")
    args = ap.parse_args()

    if args.score:
        score_labels(args.score)
        return
    if not args.cache:
        config.CACHE_TTL = 0  # honest timings: always hit the network

    cases = CASES
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[:args.limit]

    modes = ["fast", "agent"] if args.mode == "both" else [args.mode]
    stamp = time.strftime("%Y%m%d_%H%M")
    tag = args.model.replace(":", "_").replace("/", "_")
    all_rows = []

    for mode in modes:
        rows = []
        for c in cases:
            print(f"[{mode}:{c['id']}] {c['question']}", flush=True)
            r = run_case(c, args.model, mode)
            rows.append(r)
            print(f"     {'PASS' if r['passed'] else 'FAIL'}  first={r['first_result_s']}s  total={r['latency_s']}s  "
                  f"prec={r['tool_precision']}  fake_ids={r['fake_ids']}", flush=True)
        summarize(rows, args.model, mode)
        all_rows += rows

    base = f"eval_{tag}_{stamp}"
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    fields = [k for k in all_rows[0] if not k.startswith("_")]
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    write_labels(all_rows, base + "_labels.csv")
    print(f"Saved: {base}.csv / .json (full overviews) and {base}_labels.csv (label these by hand)")


if __name__ == "__main__":
    main()