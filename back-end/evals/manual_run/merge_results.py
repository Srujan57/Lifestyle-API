#!/usr/bin/env python3
"""Pair a run_question_bank.py result set back onto its source sheet.

Emits one row per question with the ORIGINAL columns (expected rule, category,
prior Bot Response, prior Pass/Fail, Notes) sitting next to the NEW response
and its retrieval stats, so nothing has to be re-matched by hand.

    python3 evals/manual_run/merge_results.py results/<run>.json [-o out.csv]

The source columns come from the `meta` dict that run_question_bank.py carries
through from the input CSV, so this works for any sheet, not just this one.
"""
import argparse
import csv
import json
import os

p = argparse.ArgumentParser()
p.add_argument("results_json")
p.add_argument("-o", "--out", default=None)
a = p.parse_args()

with open(a.results_json, encoding="utf-8") as f:
    data = json.load(f)
records = data["records"]

# Preserve the source sheet's own column order, then append the new columns.
meta_cols = []
for r in records:
    for k in (r.get("meta") or {}):
        if k not in meta_cols:
            meta_cols.append(k)

NEW = ["NEW Bot Response", "NEW is_refusal", "NEW chunks_used",
       "NEW total_candidates", "NEW top1_distance", "NEW top1_source",
       "NEW animations", "NEW videos", "NEW http_status", "NEW error"]
cols = ["row", "Question"] + meta_cols + NEW

out = a.out or os.path.splitext(a.results_json)[0] + "_MERGED.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in records:
        row = {"row": r["case"], "Question": r["question"]}
        row.update(r.get("meta") or {})
        row.update({
            "NEW Bot Response": r.get("reply", ""),
            "NEW is_refusal": r.get("is_refusal"),
            "NEW chunks_used": r.get("chunks_used"),
            "NEW total_candidates": r.get("total_candidates"),
            "NEW top1_distance": r.get("top1_distance"),
            "NEW top1_source": r.get("top1_source"),
            "NEW animations": r.get("animations"),
            "NEW videos": r.get("videos"),
            "NEW http_status": r.get("http_status"),
            "NEW error": r.get("error"),
        })
        w.writerow(row)

print(f"wrote {out}  ({len(records)} rows, {len(cols)} columns)")
print("columns:", ", ".join(cols))
