#!/usr/bin/env python3
"""
Run the Question Bank (or any question list) against the LIVE backend and dump
everything needed to eyeball the results.

Hits Flask's /endpoint directly with show_chunks=true -- no frontend, no
browser, no RAG re-implementation. Whatever the running container serves is
exactly what gets measured.

WHY THIS EXISTS
    Every ad-hoc "run these N questions and look at the chunks" script written
    during debugging got thrown away and rewritten slightly differently the
    next time. This is that script, once, kept.

USAGE
    # whole bank against the docker-compose backend
    python3 evals/manual_run/run_question_bank.py

    # smoke test
    python3 evals/manual_run/run_question_bank.py --limit 5

    # a different bank, or a CSV export of any sheet
    python3 evals/manual_run/run_question_bank.py --input evals/RAGAS/manual_questions.csv

    # multi-turn protocol cases (history carried between messages)
    python3 evals/manual_run/run_question_bank.py --script evals/manual_run/protocol_cases.json

    # both in one run, tagged
    python3 evals/manual_run/run_question_bank.py --input "Manual Questions.csv" \
        --script evals/manual_run/protocol_cases.json --tag pre-release

INPUT FORMATS
    CSV   : one question per row. The question column is auto-detected
            (first header containing "question", else "prompt"/"query", else
            column 0). Override with --question-column.
    JSON  : any of
              ["question one", "question two"]
              [{"name": "...", "message": "..."}]                  single turn
              [{"name": "...", "messages": ["turn 1", "turn 2"]}]   multi-turn
            Multi-turn cases carry conversation history between calls, which
            single-turn fire-and-forget cannot do -- required for the SMART
            goal / EV gating / tangent-and-return protocol checks.

OUTPUT (evals/manual_run/results/, already covered by .gitignore's
         `back-end/evals/**/results/`)
    <tag>_<timestamp>.json  full fidelity: every chunk, distance, rag_debug
    <tag>_<timestamp>.csv   one row per turn, flat, for sorting in a sheet

NOTES
    * /endpoint is rate limited to 20 requests / 60s per IP and returns 429
      over that, so --delay defaults to 3.1s. 429s and 5xx are retried with
      backoff rather than silently recorded as failures.
    * rag_debug is only returned when the server has FLASK_DEBUG=1. If it is
      missing the script says so loudly instead of quietly writing empty
      chunk columns.
    * stdlib only, so it runs under any venv in this repo (back-end/.venv is
      3.9, evals/TruLens/.venv is 3.10).
"""

import argparse
import csv
import datetime
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_HERE))  # back-end/
DEFAULT_INPUT = os.path.join(_BACKEND_DIR, "Manual Questions.csv")
DEFAULT_OUT = os.path.join(_HERE, "results")

# Scope-refusal detector. Same pattern used throughout the grounding work:
# matches the bot declining a question as out of scope, NOT "I don't have your
# location" style missing-data replies, which are correct behaviour.
REFUSE = re.compile(r"outside what i can help|i'?m focused on (?:your|my)\b", re.I)

QUESTION_HEADER_HINTS = ("question", "prompt", "query")


# --------------------------------------------------------------------------
# input loading
# --------------------------------------------------------------------------
def load_csv(path, column=None):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"{path}: no header row")
        col = column
        if col is None:
            for hint in QUESTION_HEADER_HINTS:
                match = [h for h in reader.fieldnames if h and hint in h.lower()]
                if match:
                    col = match[0]
                    break
        if col is None:
            col = reader.fieldnames[0]
        if col not in reader.fieldnames:
            raise SystemExit(
                f"{path}: column {col!r} not found. Available: {reader.fieldnames}")
        print(f"  question column: {col!r}")
        # Per-row city. /endpoint reads the city from the request body only --
        # it never parses one out of the message text -- so a row whose policy
        # rule tests a geocoding or weather-coverage branch reaches that branch
        # ONLY if its city is sent. A single global --city cannot express
        # "row60 needs Zzqxonvale, row61 needs London, row21 needs none", which
        # is why six Weather rows were silently exercising the no-city fallback
        # instead of the branch they claim to test.
        city_col = next((h for h in reader.fieldnames
                         if h and h.strip().lower() == "city"), None)
        if city_col:
            print(f"  per-row city column: {city_col!r}")

        cases = []
        for i, row in enumerate(reader, 1):
            q = (row.get(col) or "").strip()
            if not q:
                continue
            cases.append({
                "name": f"row{i}",
                "messages": [q],
                "city": (row.get(city_col) or "").strip() if city_col else "",
                "meta": {k: v for k, v in row.items() if k != col and v},
            })
        return cases


def load_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases = []
    for i, item in enumerate(data, 1):
        if isinstance(item, str):
            cases.append({"name": f"case{i}", "messages": [item], "city": "", "meta": {}})
        elif isinstance(item, dict):
            msgs = item.get("messages")
            if msgs is None:
                msgs = [item["message"]] if item.get("message") else []
            if not msgs:
                continue
            cases.append({"name": item.get("name", f"case{i}"),
                          "messages": list(msgs),
                          "city": (item.get("city") or "").strip(),
                          "meta": {k: v for k, v in item.items()
                                   if k not in ("name", "message", "messages", "city")}})
    return cases


def load_input(path, column=None):
    if path.lower().endswith(".json"):
        return load_json(path)
    return load_csv(path, column)


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------
def post(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), json.load(resp)


def ask(url, message, history, city, le8, timeout, retries):
    """One turn. Returns (status, data, error_string)."""
    payload = {"message": message, "history": history, "show_chunks": True}
    if city:
        payload["city"] = city
    if le8:
        payload["le8_data"] = le8
    attempt = 0
    while True:
        try:
            return post(url, payload, timeout) + (None,)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            retryable = e.code == 429 or 500 <= e.code < 600
            if retryable and attempt < retries:
                wait = 5 * (2 ** attempt) if e.code == 429 else 2 * (2 ** attempt)
                print(f"      HTTP {e.code}; retry in {wait}s", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            return e.code, None, "HTTP {}: {}".format(e.code, detail)
        except Exception as e:  # timeout, connection refused, bad JSON
            if attempt < retries:
                wait = 2 * (2 ** attempt)
                print(f"      {type(e).__name__}; retry in {wait}s", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            return None, None, "{}: {}".format(type(e).__name__, e)


# --------------------------------------------------------------------------
# per-turn extraction
# --------------------------------------------------------------------------
def summarize_turn(data):
    """Pull the reporting fields out of one /endpoint response."""
    dbg = (data or {}).get("rag_debug") or {}
    chunks = dbg.get("chunks") or []
    used = [c for c in chunks if c.get("used_in_context")]
    top = chunks[0] if chunks else {}
    return {
        "reply": (data or {}).get("reply", ""),
        "has_rag_debug": bool(dbg),
        "rag_query": dbg.get("rag_query", ""),
        "chunks_used": dbg.get("context_chunks_count", len(used)),
        "total_candidates": dbg.get("total_candidates", len(chunks)),
        "distance_threshold": dbg.get("distance_threshold"),
        "rag_error": dbg.get("error"),
        "top1_distance": top.get("distance"),
        "top1_source": (top.get("metadata") or {}).get("source", ""),
        "animations": len((data or {}).get("animations") or []),
        "videos": len((data or {}).get("exercise_videos") or []),
        "chunks": [{
            "rank": i,
            "distance": c.get("distance"),
            "used_in_context": c.get("used_in_context"),
            "source": (c.get("metadata") or {}).get("source", ""),
            "text": c.get("text", ""),
        } for i, c in enumerate(chunks, 1)],
    }


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=None,
                   help="CSV or JSON of questions (default: back-end/Manual Questions.csv)")
    p.add_argument("--script", default=None,
                   help="JSON of multi-turn cases; may be combined with --input")
    p.add_argument("--question-column", default=None, help="override CSV column detection")
    p.add_argument("--url", default="http://localhost:5001",
                   help="backend base URL (default: docker-compose's mapped port)")
    p.add_argument("--limit", type=int, default=None, help="only the first N cases")
    p.add_argument("--start", type=int, default=0, help="skip the first N cases (resume)")
    p.add_argument("--rows", default=None,
                   help="only these case names/numbers, comma separated (e.g. 25,32,60,61). "
                        "Applied after naming, so row numbering matches the source sheet.")
    p.add_argument("--delay", type=float, default=3.1,
                   help="seconds between turns; /endpoint allows 20/60s (default 3.1)")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=2, help="retries on 429/5xx/timeout")
    p.add_argument("--city", default=None,
                   help="fallback city for rows with no City value (default: none, which "
                        "exercises the no-city path). A per-row City column always wins.")
    p.add_argument("--le8", default=None, help="path to a JSON file of le8_data")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--tag", default="run", help="filename prefix for this run")
    args = p.parse_args()

    if not args.input and not args.script:
        args.input = DEFAULT_INPUT

    cases = []
    if args.input:
        print(f"Loading {args.input}")
        cases += load_input(args.input, args.question_column)
    if args.script:
        print(f"Loading {args.script}")
        cases += load_json(args.script)

    if args.rows:
        want = {r.strip() for r in args.rows.split(",") if r.strip()}
        want |= {f"row{r}" for r in want}
        cases = [c for c in cases if c["name"] in want]
        if not cases:
            raise SystemExit(f"--rows {args.rows} matched no cases")
        print(f"  --rows filter: {[c['name'] for c in cases]}")

    cases = cases[args.start:]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases to run.")

    le8 = None
    if args.le8:
        with open(args.le8, encoding="utf-8") as f:
            le8 = json.load(f)

    endpoint = args.url.rstrip("/") + "/endpoint"
    health = args.url.rstrip("/") + "/health"
    try:
        _, h = post_get(health)
        print(f"Backend: {health} -> {h}")
    except Exception as e:
        print(f"WARNING: health check failed ({e}). Continuing anyway.")

    n_turns = sum(len(c["messages"]) for c in cases)
    print(f"{len(cases)} case(s), {n_turns} turn(s), ~{n_turns * args.delay / 60:.1f} min of pacing\n")

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out_dir, f"{args.tag}_{stamp}.json")
    csv_path = os.path.join(args.out_dir, f"{args.tag}_{stamp}.csv")

    records = []
    errors = []
    no_debug = 0
    for ci, case in enumerate(cases, 1):
        history = []
        multi = len(case["messages"]) > 1
        for ti, msg in enumerate(case["messages"], 1):
            label = f"[{ci}/{len(cases)}]" + (f" turn {ti}/{len(case['messages'])}" if multi else "")
            print(f"{label} {msg[:72]}", flush=True)
            row_city = case.get("city") or args.city
            status, data, err = ask(endpoint, msg, history, row_city, le8,
                                    args.timeout, args.retries)
            rec = {"case": case["name"], "turn": ti, "multi_turn": multi,
                   "question": msg, "city_sent": row_city or "",
                   "http_status": status, "error": err,
                   "meta": case.get("meta", {})}
            if data is None:
                rec.update({"reply": "", "is_refusal": None, "chunks": []})
                errors.append((case["name"], ti, err))
                print(f"      ERROR: {err}", flush=True)
            else:
                s = summarize_turn(data)
                rec.update(s)
                rec["is_refusal"] = bool(REFUSE.search(s["reply"]))
                if not s["has_rag_debug"]:
                    no_debug += 1
                history = data.get("history", history)
                print(f"      {s['chunks_used']}/{s['total_candidates']} chunks"
                      f"  top1={s['top1_distance']}"
                      f"{'  REFUSAL' if rec['is_refusal'] else ''}"
                      f"{'  videos=%d' % s['videos'] if s['videos'] else ''}", flush=True)
            records.append(rec)
            if args.delay:
                time.sleep(args.delay)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"run": {"timestamp": stamp, "endpoint": endpoint,
                           "input": args.input, "script": args.script,
                           "cases": len(cases), "turns": len(records)},
                   "records": records}, f, indent=2)

    cols = ["case", "turn", "multi_turn", "question", "city_sent", "reply", "is_refusal",
            "chunks_used", "total_candidates", "top1_distance", "top1_source",
            "distance_threshold", "rag_query", "rag_error", "animations",
            "videos", "http_status", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    # ---------------- summary ----------------
    dists = [r["top1_distance"] for r in records
             if isinstance(r.get("top1_distance"), (int, float))]
    refusals = [r for r in records if r.get("is_refusal")]
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  cases                 : {len(cases)}")
    print(f"  turns run             : {len(records)}")
    print(f"  refusals              : {len(refusals)}"
          f" ({100.0 * len(refusals) / len(records):.1f}%)" if records else "")
    if dists:
        print(f"  top-1 distance        : mean {statistics.mean(dists):.4f} | "
              f"median {statistics.median(dists):.4f} | "
              f"min {min(dists):.4f} | max {max(dists):.4f}")
    else:
        print("  top-1 distance        : n/a (no chunks captured)")
    print(f"  errors / non-200      : {len(errors)}")
    for name, turn, err in errors[:10]:
        print(f"      {name} turn {turn}: {err}")
    if len(errors) > 10:
        print(f"      ... and {len(errors) - 10} more")
    if no_debug:
        print(f"  WARNING: {no_debug} turn(s) returned no rag_debug — the server is"
              f" running with FLASK_DEBUG != 1, so chunk/distance columns are empty.")
    if refusals:
        print("\n  refused:")
        for r in refusals[:15]:
            print(f"      [{r['case']}] {r['question'][:64]}")
        if len(refusals) > 15:
            print(f"      ... and {len(refusals) - 15} more")
    print(f"\n  JSON: {json_path}")
    print(f"  CSV : {csv_path}")


def post_get(url):
    """GET helper for the health check."""
    with urllib.request.urlopen(url, timeout=15) as resp:
        return resp.getcode(), json.load(resp)


if __name__ == "__main__":
    main()
