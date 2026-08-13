"""Step 3: instrument ragas 0.4.3's answer_relevancy noncommittal flag.

Why
---
56 of 163 baseline rows scored exactly 0.0 on ragas_answer_relevancy. In ragas
0.4.3 that is only reachable when all 3 generated questions are flagged
noncommittal, because _calculate_score() computes:

    score = cosine_sim.mean() * int(not all_noncommittal)

Two hypotheses for the cause are already dead: trailing questions (68 rows end
in '?' and scored fine) and clinical deferral as a binary (an independent gpt-4o
classifier labelled all 163 rows COMMITS -- this corpus never substitutes a
referral for an answer). This probe opens the metric up and logs what the judge
actually does, per row and per pass.

Fidelity
--------
The judge is reconstructed exactly as the baseline built it in
evals/RAGAS/run_ragas_eval.py:430-431 --

    LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"), bypass_n=True)
    LangchainEmbeddingsWrapper(OpenAIEmbeddings())      # text-embedding-ada-002
    strictness = 3

and temperature is left as None so LangchainLLMWrapper applies
get_temperature(n=3) -> 0.3, exactly as in the baseline run. Rather than
reimplement the scoring maths, this drives ragas's own
`question_generation.generate_multiple`, `calculate_similarity`, and
`_calculate_score`, so the numbers cannot drift from what the metric does.

The one intentional deviation: ChatOpenAI gets max_retries raised from the
default 2, because this org's quota is tight and a 429 mid-run would otherwise
abort a pass. Retry count affects resilience, not sampling.

Because the judge runs at temperature 0.3 it is stochastic, so every response is
scored over --passes independent passes. That is the point: the per-row spread
across passes IS the noise floor, and without it a single 0.0 cannot be
distinguished from a reproducible one.

Read-only outside this folder. Does not import app.py.

Usage:
    .venv/bin/python back-end/evals/AnswerRelevancyProbe/probe_noncommittal.py
    .venv/bin/python back-end/evals/AnswerRelevancyProbe/probe_noncommittal.py --limit 3 --passes 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASELINE_CSV = REPO / "back-end" / "evals" / "RAGAS" / "RAGAS_Baseline.csv"
ENV_FILE = REPO / "back-end" / ".env"
LABELS_JSON = HERE / "deferral_labels.json"
OUT_CSV = HERE / "probe_results.csv"
CKPT_JSON = HERE / "probe_checkpoint.json"

JUDGE_MODEL = "gpt-4o-mini"
STRICTNESS = 3
MAX_RETRIES = 6

# Phrases that hand a question to a clinician. Used for the descriptive
# per-row features only -- nothing here labels or filters a row.
REFERRAL_PATTERNS = [
    r"care team", r"healthcare team", r"health care team", r"medical team",
    r"oncolog\w*", r"your doctor", r"your provider", r"your physician",
    r"consult\w*", r"speak (?:with|to)", r"talk (?:with|to) your",
    r"check with", r"reach out to", r"discuss (?:this )?with",
]
REFERRAL_RE = re.compile("|".join(REFERRAL_PATTERNS), re.IGNORECASE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def static_features(response: str) -> dict:
    """Per-row descriptive features. No API calls."""
    text = str(response)
    n = len(text)
    matches = list(REFERRAL_RE.finditer(text))
    feats = {
        "n_referral_phrases": len(matches),
        "referral_density_per_1k": len(matches) / (n / 1000.0) if n else np.nan,
        "response_len": n,
    }
    if matches:
        first = matches[0].start()
        feats["first_referral_pos_frac"] = first / n
        # "Substantive claim precedes deferral": is there a real sentence before
        # the first referral phrase? Heuristic -- a preceding sentence of >=40
        # chars counts as substantive. Bare empathy openers are typically shorter.
        prefix = text[:first]
        prior_sents = [s for s in SENT_SPLIT_RE.split(prefix) if s.strip()]
        feats["substantive_precedes_referral"] = any(
            len(s.strip()) >= 40 for s in prior_sents)
    else:
        feats["first_referral_pos_frac"] = np.nan
        feats["substantive_precedes_referral"] = None
    return feats


def build_judge(api_key: str):
    """Reconstruct the baseline judge. See module docstring for fidelity notes."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import ResponseRelevancy

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model=JUDGE_MODEL, api_key=api_key, max_retries=MAX_RETRIES),
        bypass_n=True,
    )
    judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key))
    scorer = ResponseRelevancy(llm=judge_llm, embeddings=judge_emb)
    assert scorer.strictness == STRICTNESS, "strictness drifted from baseline"
    return judge_llm, scorer


async def probe_one(scorer, judge_llm, sem, loop, idx: int,
                    question: str, response: str, pass_no: int) -> dict:
    """One row, one pass: generate 3 questions, flag them, score them."""
    from ragas.metrics._answer_relevance import ResponseRelevanceInput

    async with sem:
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                # temperature=None -> wrapper applies get_temperature(3) = 0.3
                outs = await scorer.question_generation.generate_multiple(
                    llm=judge_llm,
                    data=ResponseRelevanceInput(response=response),
                    n=STRICTNESS,
                )
                gen_qs = [o.question for o in outs]
                flags = [int(o.noncommittal) for o in outs]

                # Embeddings call is blocking; keep it off the event loop.
                cos = await loop.run_in_executor(
                    None, scorer.calculate_similarity, question, gen_qs)

                # Use ragas's own scorer so the final number cannot diverge.
                score = scorer._calculate_score(outs, {"user_input": question})

                return {
                    "row": idx,
                    "pass": pass_no,
                    "gen_q1": gen_qs[0] if len(gen_qs) > 0 else None,
                    "gen_q2": gen_qs[1] if len(gen_qs) > 1 else None,
                    "gen_q3": gen_qs[2] if len(gen_qs) > 2 else None,
                    "flag1": flags[0] if len(flags) > 0 else None,
                    "flag2": flags[1] if len(flags) > 1 else None,
                    "flag3": flags[2] if len(flags) > 2 else None,
                    "cos1": float(cos[0]) if len(cos) > 0 else np.nan,
                    "cos2": float(cos[1]) if len(cos) > 1 else np.nan,
                    "cos3": float(cos[2]) if len(cos) > 2 else np.nan,
                    "mean_cosine": float(np.mean(cos)),
                    "n_flagged": sum(flags),
                    "all_noncommittal": bool(np.all(flags)),
                    "score": float(score),
                }
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt == MAX_RETRIES - 1:
                    break
                if "429" in str(exc) or "rate_limit" in str(exc):
                    await asyncio.sleep(min(20.0, 5.0 * (attempt + 1)) + random.random())
                else:
                    await asyncio.sleep((2 ** attempt) + random.random())
        raise RuntimeError(f"row {idx} pass {pass_no} failed: {last_err}")


async def run(args) -> int:
    env = dotenv_values(ENV_FILE)
    api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"ERROR: no OPENAI_API_KEY in {ENV_FILE}", file=sys.stderr)
        return 1

    df = pd.read_csv(BASELINE_CSV)
    if args.limit:
        df = df.head(args.limit)

    labels = {}
    if LABELS_JSON.exists():
        labels = {r["row"]: r["label"] for r in json.loads(LABELS_JSON.read_text())}

    judge_llm, scorer = build_judge(api_key)
    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(args.concurrency)

    # Resume: keep any (row, pass) already completed.
    done = {}
    if CKPT_JSON.exists() and not args.limit:
        for rec in json.loads(CKPT_JSON.read_text()):
            done[(rec["row"], rec["pass"])] = rec
        if done:
            print(f"resuming: {len(done)} row-passes already done")

    def checkpoint():
        CKPT_JSON.write_text(json.dumps(
            [done[k] for k in sorted(done)], indent=2))

    print(f"probing {len(df)} rows x {args.passes} passes "
          f"({len(df) * args.passes * STRICTNESS} judge calls) "
          f"with {JUDGE_MODEL} @ temperature 0.3, concurrency {args.concurrency}")

    for pass_no in range(1, args.passes + 1):
        todo = [i for i in df.index if (int(i), pass_no) not in done]
        if not todo:
            print(f"pass {pass_no}: already complete")
            continue
        print(f"\npass {pass_no}/{args.passes}: {len(todo)} rows")
        t0 = time.time()

        tasks = [
            probe_one(scorer, judge_llm, sem, loop, int(i),
                      str(df.at[i, "Question"]), str(df.at[i, "Bot Answer"]), pass_no)
            for i in todo
        ]
        try:
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                done[(rec["row"], rec["pass"])] = rec
                if len(done) % 20 == 0:
                    checkpoint()
        except Exception:
            checkpoint()
            print(f"\npass {pass_no} failed; {len(done)} row-passes checkpointed. "
                  f"Re-run to resume.", file=sys.stderr)
            raise

        # Checkpoint after each pass, as specified.
        checkpoint()
        n_zero = sum(1 for k, v in done.items()
                     if k[1] == pass_no and v["score"] == 0.0)
        print(f"pass {pass_no} done in {time.time() - t0:.0f}s -- "
              f"{n_zero}/{len(df)} rows scored exactly 0.0")

    # Join per-pass results with static per-row features and baseline values.
    rows = []
    for (idx, pass_no), rec in sorted(done.items()):
        base = df.loc[idx]
        out = dict(rec)
        out["baseline_score"] = base["ragas_answer_relevancy"]
        out["baseline_zero"] = bool(base["ragas_answer_relevancy"] == 0.0)
        out["deepeval_score"] = base["deepeval_answer_relevancy"]
        out["deferral_label"] = labels.get(idx)
        out["question"] = base["Question"]
        out["category"] = base["Category"]
        out["source"] = base["Source"]
        out["ends_in_question"] = str(base["Bot Answer"]).rstrip().endswith("?")
        out.update(static_features(base["Bot Answer"]))
        rows.append(out)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(rows)} row-pass records)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    return asyncio.get_event_loop().run_until_complete(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
