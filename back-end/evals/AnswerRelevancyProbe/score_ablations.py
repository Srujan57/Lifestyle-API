"""Step 4: score the five ablation arms.

Arms are scored in a SINGLE run so every arm meets the same judge under the same
conditions, and arm order is randomised within each row so that any drift in the
API over the run cannot align with a particular arm.

Every arm is compared against the CONTROL ARM'S RERUN, never against the stored
baseline score -- the judge is stochastic at temperature 0.3, so a control rerun
is the only fair reference.

Unablatable rows are skipped for that arm (their text would be byte-identical to
control, so scoring them would burn API budget and pad the paired test with fake
zero-difference pairs). They are reported separately.

The judge is imported from probe_noncommittal so it is constructed by exactly the
same code path as Step 3.

Usage:
    .venv/bin/python back-end/evals/AnswerRelevancyProbe/score_ablations.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from probe_noncommittal import (  # noqa: E402
    ENV_FILE, STRICTNESS, build_judge, probe_one,
)

ABLATIONS_JSON = HERE / "step4_ablations.json"
OUT_CSV = HERE / "step4_results.csv"
CKPT_JSON = HERE / "step4_checkpoint.json"
ARMS = ["control", "-Q", "-D", "-Q-D", "-H"]
SEED = 20260813


async def run(args) -> int:
    env = dotenv_values(ENV_FILE)
    api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"ERROR: no OPENAI_API_KEY in {ENV_FILE}", file=sys.stderr)
        return 1

    records = json.loads(ABLATIONS_JSON.read_text())
    judge_llm, scorer = build_judge(api_key)
    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(args.concurrency)

    # Build the work list: (row, arm, pass) for every ablatable arm.
    # Arm order is shuffled per row per pass with a fixed seed.
    rng = random.Random(SEED)
    jobs = []
    for pass_no in range(1, args.passes + 1):
        for rec in records:
            arms = [a for a in ARMS if not rec["arms"][a]["unablatable"]]
            rng.shuffle(arms)
            for arm_pos, arm in enumerate(arms):
                jobs.append({
                    "row": rec["row"], "arm": arm, "pass": pass_no,
                    "arm_position": arm_pos,
                    "question": rec["question"],
                    "text": rec["arms"][arm]["text"],
                })

    done = {}
    if CKPT_JSON.exists():
        for r in json.loads(CKPT_JSON.read_text()):
            done[(r["row"], r["arm"], r["pass"])] = r
        if done:
            print(f"resuming: {len(done)} row-arm-passes already scored")

    todo = [j for j in jobs if (j["row"], j["arm"], j["pass"]) not in done]
    print(f"scoring {len(todo)} row-arm-passes "
          f"({len(todo) * STRICTNESS} judge calls), concurrency {args.concurrency}")

    def checkpoint():
        CKPT_JSON.write_text(json.dumps([done[k] for k in sorted(done)], indent=2))

    async def work(job):
        rec = await probe_one(scorer, judge_llm, sem, loop, job["row"],
                              job["question"], job["text"], job["pass"])
        rec["arm"] = job["arm"]
        rec["arm_position"] = job["arm_position"]
        return rec

    t0 = time.time()
    n_done = 0
    try:
        for fut in asyncio.as_completed([work(j) for j in todo]):
            rec = await fut
            done[(rec["row"], rec["arm"], rec["pass"])] = rec
            n_done += 1
            if n_done % 50 == 0:
                checkpoint()
                print(f"  {n_done}/{len(todo)}  ({time.time() - t0:.0f}s)")
    except Exception:
        checkpoint()
        print(f"\nfailed; {len(done)} scored records checkpointed. Re-run to resume.",
              file=sys.stderr)
        raise
    checkpoint()

    # Join with the per-arm text features computed at build time.
    feat_by = {(r["row"], a): r["arms"][a]["features"] for r in records for a in ARMS}
    unabl = {(r["row"], a): r["arms"][a]["unablatable"] for r in records for a in ARMS}
    rows = []
    for k in sorted(done):
        rec = dict(done[k])
        f = feat_by[(rec["row"], rec["arm"])]
        rec["arm_contrast_per_1k"] = f["contrast_per_1k"]
        rec["arm_empathy_per_1k"] = f["empathy_per_1k"]
        rec["arm_referral_per_1k"] = f["referral_per_1k"]
        rec["arm_len"] = f["len"]
        rec["unablatable"] = unabl[(rec["row"], rec["arm"])]
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(rows)} records) in {time.time() - t0:.0f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    return asyncio.get_event_loop().run_until_complete(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
