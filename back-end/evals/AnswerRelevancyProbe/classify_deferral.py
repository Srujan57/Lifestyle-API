"""Step 2b: label every baseline response COMMITS vs DEFERS with an LLM classifier.

Background
----------
56 of the 163 rows in evals/RAGAS/RAGAS_Baseline.csv scored exactly 0.0 on
ragas_answer_relevancy. In ragas 0.4.3 that can only happen when all 3 generated
questions are flagged noncommittal, because _calculate_score() computes
`cosine_sim.mean() * int(not all_noncommittal)`.

The first hypothesis -- that the app's Motivational-Interviewing habit of ending
on a follow-up question triggers the flag -- is dead: 68 rows end in a question
and scored fine. The replacement hypothesis is CLINICAL DEFERRAL: the flag fires
on responses that decline to commit to a substantive answer and hand the question
to the care team instead. This script produces the independent variable for that
test.

Why a different model than the judge
------------------------------------
The judge under investigation is gpt-4o-mini. Using gpt-4o-mini to also produce
the deferral label would risk measuring "gpt-4o-mini finds this evasive" twice
and manufacturing a correlation out of a shared blind spot. The classifier
therefore runs on a different, stronger model (CLASSIFIER_MODEL below) at
temperature 0. Labels are hand-verified at the Step 2b gate regardless.

Read-only with respect to everything outside this folder: reads the baseline CSV,
writes only into evals/AnswerRelevancyProbe/. Does not import app.py.

Usage:
    .venv/bin/python back-end/evals/AnswerRelevancyProbe/classify_deferral.py
    .venv/bin/python back-end/evals/AnswerRelevancyProbe/classify_deferral.py --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASELINE_CSV = REPO / "back-end" / "evals" / "RAGAS" / "RAGAS_Baseline.csv"
ENV_FILE = REPO / "back-end" / ".env"
OUT_JSON = HERE / "deferral_labels.json"

# Deliberately NOT gpt-4o-mini -- see module docstring.
CLASSIFIER_MODEL = "gpt-4o"
# This org's gpt-4o quota is 30k tokens/min and each row costs ~550 tokens, so
# the ceiling is ~55 requests/min. Two workers sit just under that; raising this
# trades a few seconds of wall clock for a storm of 429s.
MAX_WORKERS = 2
MAX_RETRIES = 8

INSTRUCTION = """\
You are auditing responses from a health-coaching chatbot that serves cancer \
survivors. For each response, decide whether it actually answers the user's \
question or whether it hands the question off to a clinician instead.

Assign exactly one label:

COMMITS - The response gives a substantive answer to the question asked. It \
states information, gives guidance, explains a mechanism, or makes a \
recommendation the user can act on. A response is still COMMITS if it ALSO \
advises consulting a doctor or care team, as long as real substantive content \
is present alongside that advice.

DEFERS - The response declines to give a substantive answer and redirects to a \
clinician or care team as the answer itself. The referral is doing the work of \
the answer. Generic encouragement, restating the question back, empathy without \
content, or "everyone is different, ask your team" all count as DEFERS when no \
actionable substance is present.

The single discriminating test: strip out the empathy and the referral. Is there \
still a real answer to the question left over? If yes -> COMMITS. If no -> DEFERS.

Judge only whether substance is present, not whether it is correct, complete, or \
well-sourced. Ending with a follow-up question to the user is irrelevant to this \
label - ignore conversational structure entirely.

Give a one-sentence reason naming the specific substantive content you found, or \
naming what was missing."""

SCHEMA = {
    "name": "deferral_label",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["COMMITS", "DEFERS"]},
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    },
}

_print_lock = threading.Lock()


def classify_one(client, idx: int, question: str, response: str) -> dict:
    """Label a single response, retrying transient API failures with backoff."""
    user_msg = (
        f"USER QUESTION:\n{question}\n\n"
        f"CHATBOT RESPONSE:\n{response}"
    )
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=CLASSIFIER_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_schema", "json_schema": SCHEMA},
            )
            payload = json.loads(resp.choices[0].message.content)
            usage = resp.usage
            return {
                "row": idx,
                "label": payload["label"],
                "reason": payload["reason"],
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            last_err = exc
            if attempt == MAX_RETRIES - 1:
                break
            # 429s here are token-per-minute quota, not per-request throttling,
            # so back off past the top of the current minute window rather than
            # retrying into the same exhausted budget.
            if "429" in str(exc) or "rate_limit" in str(exc):
                time.sleep(min(20.0, 5.0 * (attempt + 1)) + random.random())
            else:
                time.sleep((2 ** attempt) + random.random())

    # Stop the run rather than log-and-continue: a silently mislabelled row would
    # corrupt the independent variable for every downstream step.
    raise RuntimeError(f"row {idx} failed after {MAX_RETRIES} attempts: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="classify only the first N rows (smoke test)")
    args = ap.parse_args()

    env = dotenv_values(ENV_FILE)
    api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"ERROR: no OPENAI_API_KEY in {ENV_FILE}", file=sys.stderr)
        return 1

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(BASELINE_CSV)
    if args.limit:
        df = df.head(args.limit)
    print(f"classifying {len(df)} responses with {CLASSIFIER_MODEL} (temperature=0)")

    # Resume support: keep any rows already labelled by a previous run.
    done = {}
    if OUT_JSON.exists() and not args.limit:
        done = {r["row"]: r for r in json.loads(OUT_JSON.read_text())}
        if done:
            print(f"  resuming: {len(done)} rows already labelled")

    todo = [i for i in df.index if i not in done]
    if not todo:
        print("  nothing to do")
    else:
        completed = [0]

        def checkpoint():
            """Persist everything labelled so far. Caller must hold _print_lock."""
            OUT_JSON.write_text(json.dumps([done[i] for i in sorted(done)], indent=2))

        def work(i):
            rec = classify_one(client, int(i), str(df.at[i, "Question"]),
                               str(df.at[i, "Bot Answer"]))
            # Checkpoint every row rather than once at the end: an unrecoverable
            # failure partway through must not throw away the rows that already
            # succeeded and were already paid for.
            with _print_lock:
                done[rec["row"]] = rec
                completed[0] += 1
                checkpoint()
                print(f"  [{completed[0]}/{len(todo)}] row {i:3d} -> {rec['label']}")
            return rec

        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                list(pool.map(work, todo))
        except Exception:
            with _print_lock:
                checkpoint()
            print(f"\nrun failed; {len(done)} labelled rows checkpointed to "
                  f"{OUT_JSON}. Re-run to resume.", file=sys.stderr)
            raise

    records = [done[i] for i in sorted(done)]
    OUT_JSON.write_text(json.dumps(records, indent=2))

    pt = sum(r["prompt_tokens"] for r in records)
    ct = sum(r["completion_tokens"] for r in records)
    # gpt-4o list pricing at time of writing: $2.50/1M in, $10.00/1M out.
    cost = pt / 1e6 * 2.50 + ct / 1e6 * 10.00
    n_def = sum(1 for r in records if r["label"] == "DEFERS")

    print(f"\nwrote {OUT_JSON}")
    print(f"  labelled : {len(records)}")
    print(f"  DEFERS   : {n_def}")
    print(f"  COMMITS  : {len(records) - n_def}")
    print(f"  tokens   : {pt} in / {ct} out   (actual cost ~${cost:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
