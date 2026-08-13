"""Resolve the ambiguous -D arm by taking it from 3 passes to 9.

At 3 passes -D gave delta -0.173, raw p 0.036 -- clears BH (0.048) but fails
Bonferroni (0.143), and is null on the 14-row common subset (p 0.221). That is
genuinely ambiguous rather than cleanly null, so this adds passes 4-9 for the
-D arm AND its paired control rows (the same 25 rows in the control arm), giving
9 paired passes per row.

Scope is deliberately limited to those two arms on those 25 rows. Passes 1-3 are
already in step4_results.csv and are reused, not rescored.

Writes step4_extension_results.csv (passes 4-9 only).

Usage:
    .venv/bin/python back-end/evals/AnswerRelevancyProbe/extend_d_passes.py
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

from probe_noncommittal import ENV_FILE, STRICTNESS, build_judge, probe_one  # noqa: E402

ABLATIONS_JSON = HERE / "step4_ablations.json"
OUT_CSV = HERE / "step4_extension_results.csv"
CKPT_JSON = HERE / "step4_extension_checkpoint.json"
ARMS = ["control", "-D"]
SEED = 20260813


async def run(args) -> int:
    env = dotenv_values(ENV_FILE)
    api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"ERROR: no OPENAI_API_KEY in {ENV_FILE}", file=sys.stderr)
        return 1

    records = json.loads(ABLATIONS_JSON.read_text())
    # Only rows where -D is actually ablatable; control is scored on the SAME rows
    # so every comparison stays paired.
    rows = [r for r in records if not r["arms"]["-D"]["unablatable"]]
    print(f"-D ablatable rows: {len(rows)}")

    judge_llm, scorer = build_judge(api_key)
    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(args.concurrency)

    rng = random.Random(SEED + 1)
    jobs = []
    for pass_no in range(args.pass_start, args.pass_end + 1):
        for rec in rows:
            arms = list(ARMS)
            rng.shuffle(arms)
            for pos, arm in enumerate(arms):
                jobs.append({"row": rec["row"], "arm": arm, "pass": pass_no,
                             "arm_position": pos, "question": rec["question"],
                             "text": rec["arms"][arm]["text"]})

    done = {}
    if CKPT_JSON.exists():
        for r in json.loads(CKPT_JSON.read_text()):
            done[(r["row"], r["arm"], r["pass"])] = r
        if done:
            print(f"resuming: {len(done)} already scored")

    todo = [j for j in jobs if (j["row"], j["arm"], j["pass"]) not in done]
    print(f"scoring {len(todo)} row-arm-passes ({len(todo) * STRICTNESS} judge calls)")

    def checkpoint():
        CKPT_JSON.write_text(json.dumps([done[k] for k in sorted(done)], indent=2))

    async def work(job):
        rec = await probe_one(scorer, judge_llm, sem, loop, job["row"],
                              job["question"], job["text"], job["pass"])
        rec["arm"] = job["arm"]
        rec["arm_position"] = job["arm_position"]
        return rec

    t0, n = time.time(), 0
    try:
        for fut in asyncio.as_completed([work(j) for j in todo]):
            rec = await fut
            done[(rec["row"], rec["arm"], rec["pass"])] = rec
            n += 1
            if n % 50 == 0:
                checkpoint()
                print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)")
    except Exception:
        checkpoint()
        print(f"\nfailed; {len(done)} checkpointed. Re-run to resume.", file=sys.stderr)
        raise
    checkpoint()

    feat = {(r["row"], a): r["arms"][a]["features"] for r in records for a in ARMS}
    out = []
    for k in sorted(done):
        rec = dict(done[k])
        f = feat[(rec["row"], rec["arm"])]
        rec["arm_contrast_per_1k"] = f["contrast_per_1k"]
        rec["arm_empathy_per_1k"] = f["empathy_per_1k"]
        rec["arm_referral_per_1k"] = f["referral_per_1k"]
        rec["arm_len"] = f["len"]
        rec["unablatable"] = None
        out.append(rec)
    pd.DataFrame(out).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(out)} records) in {time.time() - t0:.0f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-start", type=int, default=4)
    ap.add_argument("--pass-end", type=int, default=9)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    return asyncio.get_event_loop().run_until_complete(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
