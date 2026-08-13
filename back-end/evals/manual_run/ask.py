#!/usr/bin/env python3
"""
Ad-hoc spot-check against the LIVE backend. Read the answer right here in the
terminal -- no CSV to edit, no results file to open.

run_question_bank.py is the batch instrument: fixed input file, paced run,
JSON/CSV artifacts. This is the opposite end -- one or two questions, typed
straight into the shell, answer printed immediately. Everything it shares with
the bank (transport, retry/backoff, per-turn city, the refusal detector, the
FLASK_DEBUG check) behaves the same way, so a finding here reproduces there.

USAGE
    # one question
    python3 evals/manual_run/ask.py "how do I connect my Fitbit?"

    # several independent questions (fresh conversation each)
    python3 evals/manual_run/ask.py "question one" "question two"

    # ONE conversation, turn by turn -- walk a SMART goal / EV intake / crisis flow
    python3 evals/manual_run/ask.py --history \
        "I want to be more active" "physical activity" "twice a week"

    # from a throwaway list (one question per line, # comments ok)
    python3 evals/manual_run/ask.py --file questions.txt

    # from stdin
    echo "what's my LE8 score?" | python3 evals/manual_run/ask.py

    # per-turn city (weather/geocoding branches read the request body ONLY)
    python3 evals/manual_run/ask.py --city London "what's the weather?"

    # keep a copy as well as printing
    python3 evals/manual_run/ask.py --save "question"

    # a JSON file gets you per-case city + multi-turn, same shape as
    # protocol_cases.json:  [{"name": "...", "messages": [...], "city": "..."}]
    python3 evals/manual_run/ask.py --file evals/manual_run/protocol_cases.json

PRE-FLIGHT (automatic, no flag needed)
    Before spending a single model call this checks /health and refuses to run
    on an EMPTY Chroma collection -- a backend serving 0 chunks answers every
    question from the system prompt alone and looks perfectly fine while
    testing nothing, which is the exact failure mode worth catching early.

    FLASK_DEBUG is checked on the first reply rather than up front, because
    rag_debug only rides along on a real answer and probing for it separately
    would cost an extra model call every run. If it is missing you get a loud
    warning immediately, not a footnote after the run. --strict turns that
    warning into an abort.

STDLIB ONLY, so it runs under any venv in this repo.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(_HERE, "results")

# ---------------------------------------------------------------------------
# Refusal detector.
#
# Superset of run_question_bank.py's REFUSE, fixing two gaps found while
# testing the SCOPE BOUNDARY carve-out:
#
#   1. APOSTROPHE. The bank's pattern spells the contraction i'?m with an
#      ASCII quote, but the model emits U+2019 ("I’m"). A decline phrased only
#      as "I’m focused on your LE8 scores..." -- no "outside what I can help
#      with" opener -- was therefore scored as an ANSWER. Observed in the
#      wild, not hypothetical.
#   2. BARE THIRD-PARTY REFUSALS. "I can't share another patient's medical
#      information." is a decline carrying neither stock phrase.
#
# The third clause is keyed on a third-party PERSON NOUN rather than a nearby
# "other", because proximity alone false-positives on legitimate first-person
# declines: "I can't provide advice, but other people find..." is not a scope
# refusal. Verified against both real captured replies and boundary cases.
# ---------------------------------------------------------------------------
REFUSE = re.compile(
    r"outside what i can help|i['’]?m focused on (?:your|my)\b|"
    r"(?:can['’]?t|cannot|not able to)\s+(?:\w+\s+){0,3}?"
    r"(?:access|share|provide|give|look up)\b[^.]{0,60}?"
    r"\b(?:another|other|someone else['’]?s?)\s+(?:\w+\s+){0,2}?"
    r"(?:patient|participant|member|user|client)s?['’]?s?\b",
    re.I,
)


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------
def load_file(path):
    """A .json case list (protocol_cases.json shape) or one question per line."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cases = []
        for i, item in enumerate(data, 1):
            if isinstance(item, str):
                cases.append({"name": f"case{i}", "messages": [item], "city": ""})
            elif isinstance(item, dict):
                msgs = item.get("messages")
                if msgs is None:
                    msgs = [item["message"]] if item.get("message") else []
                if msgs:
                    cases.append({"name": item.get("name", f"case{i}"),
                                  "messages": list(msgs),
                                  "city": (item.get("city") or "").strip()})
        return cases
    with open(path, encoding="utf-8") as f:
        qs = [ln.strip() for ln in f
              if ln.strip() and not ln.lstrip().startswith("#")]
    return [{"name": f"q{i}", "messages": [q], "city": ""}
            for i, q in enumerate(qs, 1)]


# ---------------------------------------------------------------------------
# transport (same retry/backoff contract as run_question_bank.ask)
# ---------------------------------------------------------------------------
def post(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ask_once(url, message, history, city, le8, timeout, retries):
    payload = {"message": message, "history": history, "show_chunks": True}
    if city:
        payload["city"] = city
    if le8:
        payload["le8_data"] = le8
    attempt = 0
    while True:
        try:
            return post(url, payload, timeout), None
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            if (e.code == 429 or 500 <= e.code < 600) and attempt < retries:
                wait = 5 * (2 ** attempt) if e.code == 429 else 2 * (2 ** attempt)
                print(f"    HTTP {e.code}; retry in {wait}s", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            return None, "HTTP {}: {}".format(e.code, detail)
        except Exception as e:
            if attempt < retries:
                wait = 2 * (2 ** attempt)
                print(f"    {type(e).__name__}; retry in {wait}s", flush=True)
                time.sleep(wait)
                attempt += 1
                continue
            return None, "{}: {}".format(type(e).__name__, e)


def preflight(base, timeout=15):
    """Fail fast on a backend that would make the whole run meaningless."""
    health = base + "/health"
    try:
        with urllib.request.urlopen(health, timeout=timeout) as resp:
            h = json.load(resp)
    except Exception as e:
        raise SystemExit(
            f"PRE-FLIGHT FAILED: {health} unreachable ({e}).\n"
            f"  Is the backend up?  docker compose up -d backend")
    chunks = h.get("chroma_chunks")
    if h.get("chroma") == "error":
        raise SystemExit(f"PRE-FLIGHT FAILED: Chroma error: {h.get('chroma_error')}")
    if chunks == 0:
        raise SystemExit(
            "PRE-FLIGHT FAILED: Chroma collection is EMPTY (0 chunks).\n"
            "  Every answer would come from the system prompt alone and RAG\n"
            "  behaviour would be untested. Ingest first, or point --url at a\n"
            "  backend with a populated volume.")
    print(f"pre-flight: {base} ok — {chunks} chunks indexed")
    return h


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
RULE = "─" * 78


def show_turn(label, question, city, data, elapsed, no_color=False):
    """Print one turn readably. Returns the summary dict."""
    dbg = (data or {}).get("rag_debug") or {}
    chunks = dbg.get("chunks") or []
    top = chunks[0] if chunks else {}
    reply = (data or {}).get("reply", "")
    refused = bool(REFUSE.search(reply))

    print(RULE)
    head = f"{label}"
    if city:
        head += f"   city={city!r}"
    print(head)
    print(f"Q: {question}")
    print(RULE)
    print(reply if reply else "(empty reply)")
    print(RULE)

    used = dbg.get("context_chunks_count")
    total = dbg.get("total_candidates", len(chunks))
    bits = []
    if dbg:
        bits.append(f"chunks {used}/{total} used")
        if top:
            src = (top.get("metadata") or {}).get("source", "")
            bits.append(f"top-1 dist {top.get('distance')} ({src[:34]})")
        else:
            bits.append("top-1 dist n/a")
    else:
        bits.append("no rag_debug (FLASK_DEBUG != 1)")
    bits.append(f"animations {len((data or {}).get('animations') or [])}")
    bits.append(f"videos {len((data or {}).get('exercise_videos') or [])}")
    bits.append(f"{elapsed:.1f}s")
    print("  " + " · ".join(bits))
    print(f"  refusal: {'YES — scope decline' if refused else 'no'}")
    if dbg.get("error"):
        print(f"  RAG ERROR: {dbg['error']}")
    for v in (data or {}).get("exercise_videos") or []:
        print(f"    video: [{v.get('difficulty')}] {v.get('title', '')[:60]}")
    for a in (data or {}).get("animations") or []:
        print(f"    animation: {a.get('title', '')[:60]}")
    print()
    return {
        "question": question, "city": city, "reply": reply,
        "refused": refused, "has_rag_debug": bool(dbg),
        "chunks_used": used, "total_candidates": total,
        "top1_distance": top.get("distance"),
        "top1_source": (top.get("metadata") or {}).get("source", ""),
        "animations": len((data or {}).get("animations") or []),
        "videos": len((data or {}).get("exercise_videos") or []),
        "elapsed_s": round(elapsed, 2),
        "rag_query": dbg.get("rag_query", ""),
    }


def main():
    p = argparse.ArgumentParser(
        description="Ad-hoc spot-check against the live backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("questions", nargs="*", help="one or more questions")
    p.add_argument("--file", default=None,
                   help="questions from a .txt (one per line) or .json case list")
    p.add_argument("--history", action="store_true",
                   help="chain all questions as ONE conversation (multi-turn) "
                        "instead of asking each with a fresh history")
    p.add_argument("--city", default=None,
                   help="city sent with every turn; /endpoint reads city from "
                        "the request body only, never from the message text")
    p.add_argument("--le8", default=None, help="path to a JSON file of le8_data")
    p.add_argument("--url", default="http://localhost:5001",
                   help="backend base URL (default: docker-compose's mapped port)")
    p.add_argument("--delay", type=float, default=3.1,
                   help="seconds between turns (rate limit is 20/60s per IP)")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--save", action="store_true",
                   help="also write a JSON transcript to results/")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--tag", default="ask", help="filename prefix when --save")
    p.add_argument("--strict", action="store_true",
                   help="abort if the backend returns no rag_debug")
    args = p.parse_args()

    # ---- gather questions -------------------------------------------------
    cases = []
    if args.file:
        cases += load_file(args.file)
    if args.questions:
        cases += [{"name": f"q{i}", "messages": [q], "city": ""}
                  for i, q in enumerate(args.questions, 1)]
    if not cases and not sys.stdin.isatty():
        qs = [ln.strip() for ln in sys.stdin
              if ln.strip() and not ln.lstrip().startswith("#")]
        cases += [{"name": f"q{i}", "messages": [q], "city": ""}
                  for i, q in enumerate(qs, 1)]
    if not cases:
        p.error("no questions given (pass them as arguments, --file, or stdin)")

    # --history collapses everything into a single conversation.
    if args.history:
        msgs = [m for c in cases for m in c["messages"]]
        city = next((c["city"] for c in cases if c["city"]), "")
        cases = [{"name": "conversation", "messages": msgs, "city": city}]

    le8 = None
    if args.le8:
        with open(args.le8, encoding="utf-8") as f:
            le8 = json.load(f)

    base = args.url.rstrip("/")
    preflight(base)
    endpoint = base + "/endpoint"

    n_turns = sum(len(c["messages"]) for c in cases)
    print(f"{len(cases)} case(s), {n_turns} turn(s)\n")

    records = []
    warned_no_debug = False
    first = True
    for ci, case in enumerate(cases, 1):
        history = []
        multi = len(case["messages"]) > 1
        for ti, msg in enumerate(case["messages"], 1):
            if not first:
                time.sleep(args.delay)
            first = False
            label = f"[{case['name']}]"
            if multi:
                label += f" turn {ti}/{len(case['messages'])}"
            city = case.get("city") or args.city or ""
            t0 = time.time()
            data, err = ask_once(endpoint, msg, history, city, le8,
                                 args.timeout, args.retries)
            elapsed = time.time() - t0
            if err:
                print(RULE)
                print(f"{label}\nQ: {msg}")
                print(f"  REQUEST FAILED: {err}")
                print()
                records.append({"question": msg, "error": err})
                continue
            rec = show_turn(label, msg, city, data, elapsed)
            rec["case"] = case["name"]
            rec["turn"] = ti
            records.append(rec)

            if not rec["has_rag_debug"] and not warned_no_debug:
                warned_no_debug = True
                msg_ = ("!! No rag_debug in the response — the backend is running\n"
                        "   with FLASK_DEBUG != 1, so chunk counts and distances\n"
                        "   are unavailable for this whole run.")
                if args.strict:
                    raise SystemExit(msg_ + "\n   (--strict: aborting)")
                print(msg_ + "\n")

            # Only chain history when asked; otherwise each question is fresh.
            if args.history or multi:
                history = data.get("history", history)

    # ---- summary ----------------------------------------------------------
    ok = [r for r in records if "error" not in r]
    refused = [r for r in ok if r["refused"]]
    print(RULE)
    print(f"{len(ok)} turn(s) · {len(refused)} refusal(s) · "
          f"{len(records) - len(ok)} error(s)")
    for r in refused:
        print(f"    refused: {r['question'][:66]}")

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(args.out_dir, f"{args.tag}_{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  saved: {path}")


if __name__ == "__main__":
    main()
