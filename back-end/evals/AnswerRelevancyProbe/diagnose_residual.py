"""Diagnostic: within the -Q arm, what distinguishes rows that stayed flagged
from rows that flipped?

Both groups had their trailing question removed, so the trailing question is held
constant and cannot explain the difference. This is descriptive only -- no new
arms, no new hypotheses, no API calls. Everything is computed from
step4_ablations.json (post-ablation text) and step4_per_row.csv (outcomes).

Groups, over the 3 scored passes of the -Q arm:
    still_flagged : P(zero) > 0.5  (majority of passes still score exactly 0.0)
    flipped       : P(zero) < 0.5

Features are measured on the POST-ABLATION text, i.e. the -Q arm's own text.

Writes results/table8_residual_diagnostic.csv,
results/step4_residual_per_row.csv and step4_residual_still_flagged.txt.
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT_TXT = HERE / "step4_residual_still_flagged.txt"

AUTONOMY = [
    r"you might", r"you could", r"up to you", r"some people find",
    r"if you'?d like", r"whatever feels", r"when you'?re ready",
]
OPTIONS = [r"\bor\b", r"\balternatively\b", r"\banother option\b"]
REFLECTION = [
    r"it sounds like", r"what i'?m hearing", r"you mentioned", r"it seems like",
]
SENT_SPLIT = re.compile(r"(?<=[.!?])(?<!\d\.)\s+")


def count(pats, t):
    return sum(len(re.findall(p, t, re.IGNORECASE)) for p in pats)


def featurize(t: str) -> dict:
    n = max(len(t), 1)
    per1k = 1000.0 / n
    sents = [s for s in SENT_SPLIT.split(t) if s.strip()]
    a, o, r = count(AUTONOMY, t), count(OPTIONS, t), count(REFLECTION, t)
    return {
        "question_marks_total": t.count("?"),
        "has_any_question_mark": int(t.count("?") > 0),
        "autonomy_count": a,
        "autonomy_per_1k": a * per1k,
        "options_count": o,
        "options_per_1k": o * per1k,
        "reflection_count": r,
        "reflection_per_1k": r * per1k,
        "len_chars": len(t),
        "n_sentences": len(sents),
    }


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    s = np.minimum(p[o] * m / np.arange(1, m + 1), 1.0)
    s = np.minimum.accumulate(s[::-1])[::-1]
    out = np.empty_like(s)
    out[o] = s
    return out


def main() -> int:
    records = {r["row"]: r for r in json.loads((HERE / "step4_ablations.json").read_text())}
    per = pd.read_csv(RESULTS / "step4_per_row.csv")
    q = per[per.arm == "-Q"].set_index("row")

    still = sorted(q.index[q.p_zero > 0.5])
    flip = sorted(q.index[q.p_zero < 0.5])
    print(f"-Q arm rows: {len(q)}   still_flagged: {len(still)}   flipped: {len(flip)}")
    assert not set(still) & set(flip)

    rows = []
    for grp, ids in (("still_flagged", still), ("flipped", flip)):
        for i in ids:
            f = featurize(records[i]["arms"]["-Q"]["text"])
            f.update({"row": i, "group": grp, "p_zero_minus_Q": q.at[i, "p_zero"],
                      "mean_cosine_minus_Q": q.at[i, "mean_cosine"]})
            rows.append(f)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "step4_residual_per_row.csv", index=False)

    feats = ["question_marks_total", "has_any_question_mark",
             "autonomy_count", "autonomy_per_1k",
             "options_count", "options_per_1k",
             "reflection_count", "reflection_per_1k",
             "len_chars", "n_sentences"]
    res = []
    A = df[df.group == "still_flagged"]
    B = df[df.group == "flipped"]
    for c in feats:
        a, b = A[c].dropna(), B[c].dropna()
        try:
            u, p = mannwhitneyu(a, b)
            auc = u / (len(a) * len(b))
        except ValueError:
            u, p, auc = np.nan, 1.0, 0.5
        res.append({
            "feature": c,
            "n_still_flagged": len(a), "n_flipped": len(b),
            "mean_still_flagged": a.mean(), "mean_flipped": b.mean(),
            "median_still_flagged": a.median(), "median_flipped": b.median(),
            "AUC_still_flagged_vs_flipped": auc,
            "separation": abs(auc - 0.5),
            "p_raw": p,
        })
    t = pd.DataFrame(res).sort_values("separation", ascending=False)
    t["p_benjamini_hochberg"] = bh(t["p_raw"].values)
    t["n_tests"] = len(t)
    t["population"] = (f"-Q arm only (trailing question removed in BOTH groups); "
                       f"still_flagged = P(zero)>0.5 (n={len(still)}), "
                       f"flipped = P(zero)<0.5 (n={len(flip)})")
    t["measured_on"] = "post-ablation text of the -Q arm"
    t.to_csv(RESULTS / "table8_residual_diagnostic.csv", index=False)

    print()
    print(t[["feature", "mean_still_flagged", "mean_flipped",
             "AUC_still_flagged_vs_flipped", "p_raw",
             "p_benjamini_hochberg"]].to_string(index=False))

    # ---- primary deliverable: the actual text ----
    lines = []
    w = lines.append
    w("=" * 100)
    w("POST-ABLATION TEXT OF EVERY -Q ROW THAT REMAINED FLAGGED")
    w("=" * 100)
    w("")
    w("These are the responses AFTER the trailing question was removed, which")
    w("still scored exactly 0.0 in a majority of the 3 passes.")
    w(f"n = {len(still)} of {len(q)} -Q-ablatable rows.")
    w("")
    w("For contrast, the same ablation flipped %d rows to non-zero." % len(flip))
    w("")
    for i in still:
        rec = records[i]
        f = featurize(rec["arms"]["-Q"]["text"])
        w("=" * 100)
        w(f"ROW {i}   P(zero) after -Q = {q.at[i,'p_zero']:.2f}   "
          f"mean_cosine = {q.at[i,'mean_cosine']:.4f}")
        w(f"  question_marks={f['question_marks_total']}  "
          f"autonomy={f['autonomy_count']}  options={f['options_count']}  "
          f"reflection={f['reflection_count']}  "
          f"len={f['len_chars']}  sentences={f['n_sentences']}")
        w(f"  USER QUESTION: {rec['question']}")
        if rec["arms"]["-Q"]["removed"]:
            w("  REMOVED BY -Q:")
            for x in rec["arms"]["-Q"]["removed"]:
                w(f"    - {x!r}")
        w("-" * 100)
        for para in rec["arms"]["-Q"]["text"].split("\n"):
            if para.strip():
                w(textwrap.fill(para, 96))
            else:
                w("")
        w("")
    OUT_TXT.write_text("\n".join(lines))
    print(f"\nwrote {OUT_TXT}")
    print(f"wrote {RESULTS/'table8_residual_diagnostic.csv'}")
    print(f"wrote {RESULTS/'step4_residual_per_row.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
