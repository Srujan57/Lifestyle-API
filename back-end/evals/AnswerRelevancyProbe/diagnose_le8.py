"""Diagnostic: cross-reference LE8 content against the answer-relevancy zeros
and, separately, against faithfulness.

Two LE8 signals are computed and deliberately NOT merged:
  (a) Category column is an LE8 domain
  (b) Bot Answer text carries an LE8 score / tier / category reference

They measure different things -- (a) is how the question was catalogued, (b) is
what the answer actually said -- so they are reported side by side with their
disagreements enumerated.

The relevancy analysis (section 2) and the faithfulness analysis (section 3) are
kept visually separate: they are different metrics and different failure modes.

No API calls. Reads RAGAS_Baseline.csv only.
Writes results/table9_le8_crosstab.csv and results/step5_le8_per_row.csv.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASELINE = REPO / "back-end" / "evals" / "RAGAS" / "RAGAS_Baseline.csv"
RESULTS = HERE / "results"
REPORT = HERE / "step5_le8_report.txt"

# ---- signal (a): Category column ----
LE8_CATEGORIES = {
    "BMI", "Blood Pressure", "Blood Lipids", "Blood Sugar", "Sleep",
    "LE8 Composite", "Physical Activity", "Diet",
}

# ---- signal (b): text-level LE8 score / tier / category reference ----
LE8_DOMAIN_TERMS = (
    r"BMI|body mass index|blood pressure|systolic|diastolic|mmHg"
    r"|blood lipid\w*|cholesterol|non-HDL|LDL|HDL"
    r"|blood sugar|glucose|HbA1c|A1C"
    r"|sleep|physical activity|steps|diet|MEPA"
    r"|nicotine|smoking|LE8|Life'?s Essential 8"
)
DOMAIN_RE = re.compile(LE8_DOMAIN_TERMS, re.IGNORECASE)
NUM_RE = re.compile(r"\d")
# b2: explicit score/tier vocabulary
SCORE_TIER_RE = re.compile(
    r"your score|score of|scores? (?:is|are|of)|\bpoints?\b|\btier\b"
    r"|\bideal\b|\bintermediate\b", re.IGNORECASE)
# b3: the bare qualifiers the brief named -- broad, tracked separately
BARE_QUALIFIER_RE = re.compile(r"\bmoderate\b|\bhigh\b", re.IGNORECASE)
PROXIMITY = 40  # chars between a domain term and a digit to count as "attached"


def number_attached_to_domain(text: str, window: int = PROXIMITY) -> bool:
    """True if a digit appears within `window` chars of an LE8 domain term."""
    t = str(text)
    for m in DOMAIN_RE.finditer(t):
        lo = max(0, m.start() - window)
        hi = min(len(t), m.end() + window)
        if NUM_RE.search(t[lo:hi]):
            return True
    return False


def fisher_block(title, a, b, c, d, rowlab, w):
    """2x2 with Fisher exact. a,b = positive row; c,d = negative row."""
    w(f"  {title}")
    w(f"    {'':<26s} {'baseline_zero':>14s} {'baseline_nonzero':>18s}")
    w(f"    {rowlab[0]:<26s} {a:>14d} {b:>18d}")
    w(f"    {rowlab[1]:<26s} {c:>14d} {d:>18d}")
    if min(a + b, c + d) == 0 or min(a + c, b + d) == 0:
        w("    Fisher exact: UNDEFINED (a margin is zero)")
        return
    odds, p = fisher_exact([[a, b], [c, d]])
    r1 = a / (a + b) if (a + b) else float("nan")
    r2 = c / (c + d) if (c + d) else float("nan")
    w(f"    Fisher exact: odds ratio = {odds:.3f}, p = {p:.5g}")
    w(f"    zero rate, {rowlab[0]:<24s} = {r1:.3f}  ({a}/{a+b})")
    w(f"    zero rate, {rowlab[1]:<24s} = {r2:.3f}  ({c}/{c+d})")


def main() -> int:
    df = pd.read_csv(BASELINE)
    df["baseline_zero"] = df["ragas_answer_relevancy"] == 0.0

    # signal (a)
    df["le8_by_category"] = df["Category"].isin(LE8_CATEGORIES)

    # signal (b) and its components
    df["b1_number_near_domain"] = df["Bot Answer"].map(number_attached_to_domain)
    df["b2_score_tier_wording"] = df["Bot Answer"].map(
        lambda t: bool(SCORE_TIER_RE.search(str(t))))
    df["b3_bare_qualifier"] = df["Bot Answer"].map(
        lambda t: bool(BARE_QUALIFIER_RE.search(str(t))))
    df["le8_by_text"] = (df.b1_number_near_domain | df.b2_score_tier_wording
                         | df.b3_bare_qualifier)

    out, w = [], None
    lines = []
    w = lines.append

    w("=" * 100)
    w("LE8 CONTENT vs ANSWER-RELEVANCY ZEROS  and  (separately) vs FAITHFULNESS")
    w("=" * 100)
    w(f"source: {BASELINE}")
    w(f"rows  : {len(df)}")
    w("")

    # ---------------------------------------------------------------- 1
    w("#" * 100)
    w("# SECTION 1 -- IDENTIFYING LE8 CONTENT (two independent signals, NOT merged)")
    w("#" * 100)
    w("")
    w(f"(a) le8_by_category : Category in {sorted(LE8_CATEGORIES)}")
    w(f"    -> {int(df.le8_by_category.sum())} rows")
    w("")
    w("    Category breakdown:")
    for cat, n in df["Category"].value_counts().items():
        mark = "LE8" if cat in LE8_CATEGORIES else "   "
        w(f"      [{mark}] {cat:<34s} {n:>3d}")
    w("")
    w("(b) le8_by_text : Bot Answer carries an LE8 score/tier/category reference")
    w(f"    -> {int(df.le8_by_text.sum())} rows")
    w("")
    w("    Components (a row can trip more than one):")
    w(f"      b1 digit within {PROXIMITY} chars of an LE8 domain term : "
      f"{int(df.b1_number_near_domain.sum()):>3d}")
    w(f"      b2 explicit score/tier wording                     : "
      f"{int(df.b2_score_tier_wording.sum()):>3d}")
    w(f"      b3 bare 'moderate'/'high'                          : "
      f"{int(df.b3_bare_qualifier.sum()):>3d}")
    w("")
    w("    CAVEAT: b3 is the qualifier wording named in the brief, but 'moderate'")
    w("    and 'high' are also ordinary exercise vocabulary in this corpus")
    w("    ('moderate-intensity aerobic activity', 'high-impact'), so b3 is a")
    w("    weak indicator of LE8 scoring content. Counts with and without b3")
    w("    are reported below so its effect is visible.")
    b_no3 = df.b1_number_near_domain | df.b2_score_tier_wording
    w(f"      le8_by_text WITHOUT b3 : {int(b_no3.sum())} rows "
      f"(vs {int(df.le8_by_text.sum())} with it)")
    df["le8_by_text_strict"] = b_no3
    w("")

    # disagreements
    w("-" * 100)
    w("WHERE (a) AND (b) DISAGREE")
    w("-" * 100)
    both = df.le8_by_category & df.le8_by_text
    a_only = df.le8_by_category & ~df.le8_by_text
    b_only = ~df.le8_by_category & df.le8_by_text
    neither = ~df.le8_by_category & ~df.le8_by_text
    w("")
    w(f"    {'':<22s} {'text=YES':>10s} {'text=NO':>10s}")
    w(f"    {'category=YES':<22s} {int(both.sum()):>10d} {int(a_only.sum()):>10d}")
    w(f"    {'category=NO':<22s} {int(b_only.sum()):>10d} {int(neither.sum()):>10d}")
    w("")
    agree = int((both | neither).sum())
    w(f"    agreement: {agree}/{len(df)} = {100*agree/len(df):.1f}%")
    w(f"    Cohen-style disagreement: {int(a_only.sum())} category-only, "
      f"{int(b_only.sum())} text-only")
    w("")
    if a_only.sum():
        w(f"    CATEGORY=LE8 but TEXT shows no LE8 reference ({int(a_only.sum())} rows):")
        for i in df.index[a_only][:12]:
            w(f"      row {i:3d}  [{df.at[i,'Category']}]  "
              f"{str(df.at[i,'Question'])[:78]}")
        w("")
    if b_only.sum():
        w(f"    TEXT shows LE8 reference but CATEGORY is not LE8 "
          f"({int(b_only.sum())} rows) -- first 12:")
        for i in df.index[b_only][:12]:
            trig = [n for n, f in (("b1", df.at[i, "b1_number_near_domain"]),
                                   ("b2", df.at[i, "b2_score_tier_wording"]),
                                   ("b3", df.at[i, "b3_bare_qualifier"])) if f]
            w(f"      row {i:3d}  [{df.at[i,'Category']}]  trips={'+'.join(trig)}  "
              f"{str(df.at[i,'Question'])[:60]}")
        w("")

    # ---------------------------------------------------------------- 2
    w("")
    w("#" * 100)
    w("# SECTION 2 -- LE8 CONTENT vs ANSWER-RELEVANCY ZEROS")
    w("#   metric: ragas_answer_relevancy == 0.0   (the noncommittal-flag artifact)")
    w("#" * 100)
    w("")
    for lbl, col, rowlab in (
        ("(a) BY CATEGORY", "le8_by_category", ("LE8 category", "non-LE8 category")),
        ("(b) BY TEXT", "le8_by_text", ("LE8 text ref", "no LE8 text ref")),
        ("(b-strict) BY TEXT, no b3", "le8_by_text_strict",
         ("LE8 text (strict)", "no LE8 text (strict)")),
    ):
        m = df[col]
        a = int((m & df.baseline_zero).sum())
        b = int((m & ~df.baseline_zero).sum())
        c = int((~m & df.baseline_zero).sum())
        d = int((~m & ~df.baseline_zero).sum())
        fisher_block(lbl, a, b, c, d, rowlab, w)
        w("")

    # ---------------------------------------------------------------- 3
    w("")
    w("#" * 100)
    w("# SECTION 3 -- LE8 CONTENT vs FAITHFULNESS   [DIFFERENT METRIC, DIFFERENT ARTIFACT]")
    w("#   metric: ragas_faithfulness    (the LE8-injection artifact)")
    w("#   This is NOT the same failure mode as Section 2. Do not read across.")
    w("#" * 100)
    w("")
    f = df["ragas_faithfulness"]
    w(f"  faithfulness overall: n={f.notna().sum()}, NaN={f.isna().sum()}, "
      f"mean={f.mean():.4f}, median={f.median():.4f}")
    w("")
    for lbl, col in (("(a) BY CATEGORY", "le8_by_category"),
                     ("(b) BY TEXT", "le8_by_text"),
                     ("(b-strict) BY TEXT, no b3", "le8_by_text_strict")):
        m = df[col]
        fa, fb = df.loc[m, "ragas_faithfulness"], df.loc[~m, "ragas_faithfulness"]
        w(f"  {lbl}")
        w(f"    LE8     : n={len(fa):>3d}  mean faithfulness = {fa.mean():.4f}  "
          f"median = {fa.median():.4f}")
        w(f"    non-LE8 : n={len(fb):>3d}  mean faithfulness = {fb.mean():.4f}  "
          f"median = {fb.median():.4f}")
        w(f"    difference (LE8 - non-LE8) = {fa.mean() - fb.mean():+.4f}")
        try:
            u, p = mannwhitneyu(fa.dropna(), fb.dropna())
            w(f"    Mann-Whitney U: p = {p:.5g}, AUC = {u/(len(fa.dropna())*len(fb.dropna())):.3f}")
        except ValueError:
            w("    Mann-Whitney U: undefined")
        lo_le8 = int((fa < 0.4).sum())
        lo_non = int((fb < 0.4).sum())
        w(f"    LE8 rows below 0.4     : {lo_le8}/{len(fa)} ({100*lo_le8/max(len(fa),1):.1f}%)")
        w(f"    non-LE8 rows below 0.4 : {lo_non}/{len(fb)} ({100*lo_non/max(len(fb),1):.1f}%)")
        w("")

    # ---------------------------------------------------------------- 4
    w("")
    w("#" * 100)
    w("# SECTION 4 -- OVERLAP: are the two failure modes the same rows?")
    w("#" * 100)
    w("")
    lowf = df["ragas_faithfulness"] < 0.4
    z = df.baseline_zero
    a = int((z & lowf).sum())
    b = int((z & ~lowf).sum())
    c = int((~z & lowf).sum())
    d = int((~z & ~lowf).sum())
    w(f"    {'':<26s} {'faithfulness<0.4':>18s} {'faithfulness>=0.4':>19s}")
    w(f"    {'answer_relevancy == 0':<26s} {a:>18d} {b:>19d}")
    w(f"    {'answer_relevancy > 0':<26s} {c:>18d} {d:>19d}")
    w("")
    odds, p = fisher_exact([[a, b], [c, d]])
    w(f"    Fisher exact: odds ratio = {odds:.3f}, p = {p:.5g}")
    w(f"    low-faithfulness rate among relevancy zeros    : "
      f"{a/(a+b):.3f}  ({a}/{a+b})")
    w(f"    low-faithfulness rate among relevancy non-zeros: "
      f"{c/(c+d):.3f}  ({c}/{c+d})")
    w("")
    rz = df.loc[z, "ragas_faithfulness"]
    rn = df.loc[~z, "ragas_faithfulness"]
    w(f"    mean faithfulness, relevancy zeros    : {rz.mean():.4f}")
    w(f"    mean faithfulness, relevancy non-zeros: {rn.mean():.4f}")
    try:
        u, pp = mannwhitneyu(rz.dropna(), rn.dropna())
        w(f"    Mann-Whitney U on faithfulness: p = {pp:.5g}, "
          f"AUC = {u/(len(rz.dropna())*len(rn.dropna())):.3f}")
    except ValueError:
        pass
    w("")
    w(f"    Rows that are BOTH relevancy-zero AND faithfulness<0.4: {a}")
    if a:
        w(f"      row ids: {sorted(df.index[z & lowf])}")
    w("")

    REPORT.write_text("\n".join(lines))

    keep = ["Question", "Category", "Source", "ragas_answer_relevancy",
            "ragas_faithfulness", "deepeval_answer_relevancy", "baseline_zero",
            "le8_by_category", "le8_by_text", "le8_by_text_strict",
            "b1_number_near_domain", "b2_score_tier_wording", "b3_bare_qualifier"]
    df[keep].to_csv(RESULTS / "step5_le8_per_row.csv", index=True, index_label="row")

    ct = []
    for metric, mask_col in (("answer_relevancy_zero", "baseline_zero"),
                             ("faithfulness_below_0.4", None)):
        for sig in ("le8_by_category", "le8_by_text", "le8_by_text_strict"):
            m = df[sig]
            target = df.baseline_zero if mask_col else (df["ragas_faithfulness"] < 0.4)
            a_, b_ = int((m & target).sum()), int((m & ~target).sum())
            c_, d_ = int((~m & target).sum()), int((~m & ~target).sum())
            odds_, p_ = fisher_exact([[a_, b_], [c_, d_]])
            ct.append({"metric": metric, "le8_signal": sig,
                       "le8_and_target": a_, "le8_not_target": b_,
                       "nonle8_and_target": c_, "nonle8_not_target": d_,
                       "rate_le8": a_ / max(a_ + b_, 1),
                       "rate_nonle8": c_ / max(c_ + d_, 1),
                       "fisher_odds_ratio": odds_, "fisher_p": p_})
    pd.DataFrame(ct).to_csv(RESULTS / "table9_le8_crosstab.csv", index=False)

    print("\n".join(lines))
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
