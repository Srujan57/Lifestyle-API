"""Step 2b analysis: cross-tabulate deferral and trailing-question against the
zero/non-zero split, and emit the hand-verification review file.

Reads the baseline CSV and deferral_labels.json. Writes step2b_review.txt and
step2b_crosstab.csv into this folder. No API calls, no writes outside this folder.
"""
from __future__ import annotations

import json
import re
import textwrap
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASELINE_CSV = REPO / "back-end" / "evals" / "RAGAS" / "RAGAS_Baseline.csv"
LABELS_JSON = HERE / "deferral_labels.json"
REVIEW_TXT = HERE / "step2b_review.txt"
CROSSTAB_CSV = HERE / "step2b_crosstab.csv"

# Phrases that hand a question to a clinician. Used only to RANK how
# deferral-flavoured a response is for review sampling -- not to label anything.
REFERRAL_PATTERNS = [
    r"care team", r"healthcare team", r"health care team", r"medical team",
    r"oncolog\w*", r"your doctor", r"your provider", r"your physician",
    r"consult\w*", r"speak (?:with|to)", r"talk (?:with|to) your",
    r"check with", r"reach out to", r"discuss (?:this )?with",
]
REFERRAL_RE = re.compile("|".join(REFERRAL_PATTERNS), re.IGNORECASE)


def ends_in_question(text) -> bool:
    return str(text).rstrip().endswith("?")


def fisher_block(name, a, b, c, d, rowlab, collab):
    """Format a 2x2 with a Fisher exact test, guarding degenerate tables."""
    out = []
    w = out.append
    w(f"  {name}")
    w(f"    {'':<22s} {'score == 0.0':>14s} {'score > 0.0':>14s}")
    w(f"    {rowlab[0]:<22s} {a:>14d} {b:>14d}")
    w(f"    {rowlab[1]:<22s} {c:>14d} {d:>14d}")
    if min(a + b, c + d) == 0 or min(a + c, b + d) == 0:
        w("    Fisher exact: UNDEFINED -- a margin is zero, the variable does not vary.")
        w("    No association can be measured from this table.")
    else:
        odds, p = fisher_exact([[a, b], [c, d]])
        w(f"    Fisher exact: odds ratio = {odds:.3f}, p = {p:.5f}")
        z_rate = a / (a + c) if (a + c) else float("nan")
        n_rate = b / (b + d) if (b + d) else float("nan")
        w(f"    {rowlab[0]} share of zeros     = {z_rate:.3f}")
        w(f"    {rowlab[0]} share of non-zeros = {n_rate:.3f}")
    w("")
    return out


def main() -> int:
    df = pd.read_csv(BASELINE_CSV)
    labels = {r["row"]: r for r in json.loads(LABELS_JSON.read_text())}
    df["defer_label"] = [labels[i]["label"] for i in df.index]
    df["defer_reason"] = [labels[i]["reason"] for i in df.index]
    df["zero"] = df["ragas_answer_relevancy"] == 0.0
    df["endsq"] = df["Bot Answer"].map(ends_in_question)
    df["n_referral"] = df["Bot Answer"].map(lambda t: len(REFERRAL_RE.findall(str(t))))
    df["len"] = df["Bot Answer"].str.len()
    df["referral_density"] = df["n_referral"] / (df["len"] / 1000.0)

    df[["Question", "defer_label", "defer_reason", "zero", "endsq",
        "n_referral", "referral_density", "ragas_answer_relevancy",
        "deepeval_answer_relevancy"]].to_csv(CROSSTAB_CSV, index=True,
                                             index_label="row")

    lines = []
    w = lines.append
    w("=" * 100)
    w("STEP 2B REVIEW -- deferral hypothesis")
    w("=" * 100)
    w("")
    w(f"Baseline : {BASELINE_CSV}")
    w(f"Labels   : {LABELS_JSON}  (classifier: gpt-4o, temperature 0)")
    w(f"Rows     : {len(df)}")
    w("")

    w("-" * 100)
    w("HEADLINE: THE DEFERRAL VARIABLE HAS NO VARIANCE")
    w("-" * 100)
    w("")
    counts = Counter(df["defer_label"])
    w(f"  COMMITS : {counts['COMMITS']}")
    w(f"  DEFERS  : {counts['DEFERS']}")
    w("")
    w("  Every one of the 163 responses was labelled COMMITS. The DEFERS row of the")
    w("  requested 2x2 is empty, so the cross-tabulation and its Fisher exact test")
    w("  cannot be computed as specified.")
    w("")
    w("  This is NOT a broken classifier. Fed synthetic texts, the same prompt and")
    w("  model separate the classes cleanly:")
    w("     'Everyone's situation is different, discuss with your healthcare")
    w("      team...'                                              -> DEFERS")
    w("     'I hear you. This one is best answered by your care team.' -> DEFERS")
    w("     empathy with no content and no referral                 -> DEFERS")
    w("     \"I'm not able to advise on that. Contact your oncology")
    w("      team right away.\"                                     -> DEFERS")
    w("     substance + 'do check with your care team'              -> COMMITS")
    w("     substance alone                                         -> COMMITS")
    w("")
    w("  The finding is a property of the corpus: this chatbot always supplies some")
    w("  substantive content. Referral language is layered ON TOP of an answer, it")
    w("  never REPLACES the answer. So 'declines to answer and redirects' describes")
    w("  no row in the baseline, and clinical deferral in the binary sense proposed")
    w("  cannot be what distinguishes the 56 zeros.")
    w("")

    w("-" * 100)
    w("2x2 TABLES")
    w("-" * 100)
    w("")
    a = int(((df.defer_label == "DEFERS") & df.zero).sum())
    b = int(((df.defer_label == "DEFERS") & ~df.zero).sum())
    c = int(((df.defer_label == "COMMITS") & df.zero).sum())
    d = int(((df.defer_label == "COMMITS") & ~df.zero).sum())
    lines.extend(fisher_block("(1) DEFERRAL x SCORE  [as requested]", a, b, c, d,
                              ("DEFERS", "COMMITS"), "score"))

    a2 = int((df.endsq & df.zero).sum())
    b2 = int((df.endsq & ~df.zero).sum())
    c2 = int((~df.endsq & df.zero).sum())
    d2 = int((~df.endsq & ~df.zero).sum())
    lines.extend(fisher_block("(2) TRAILING QUESTION x SCORE  [for comparison]",
                              a2, b2, c2, d2,
                              ("ends in ?", "does not"), "score"))

    w("  VERDICT ON WHICH VARIABLE SEPARATES BETTER")
    w("    Trailing question : weak but real separation (OR 3.00, p = 0.0068).")
    w("                        Not sufficient -- 68 rows end in '?' and scored fine.")
    w("    Clinical deferral : no separation measurable at all; the variable is")
    w("                        constant across the corpus.")
    w("    Neither variable explains the 56 zeros. The trailing-question split,")
    w("    despite being rejected as the cause, remains the only one of the two with")
    w("    any measurable association.")
    w("")

    w("-" * 100)
    w("GRADED REFERRAL SIGNAL (regex, for orientation only -- not a label)")
    w("-" * 100)
    w("")
    w("  The binary is constant, but referral LANGUAGE still varies in amount, and")
    w("  it varies in the direction the hypothesis predicts:")
    w("")
    for lbl, mask in [("zeros    ", df.zero), ("non-zeros", ~df.zero)]:
        sub = df[mask]
        w(f"    {lbl}  n={len(sub):3d}   mean referral phrases/response = "
          f"{sub.n_referral.mean():.2f}   per 1k chars = {sub.referral_density.mean():.2f}")
    w("")
    w("  So deferral is plausibly still live as a GRADED property (how much of the")
    w("  response is referral vs substance) even though it is dead as a binary.")
    w("  Measuring that needs a graded instrument, not a two-way label.")
    w("")

    # Review samples. Two of four cells are empty, so sample the two that exist,
    # and prioritise the zero-scoring rows since that is where the COMMITS label
    # carries the most weight.
    w("=" * 100)
    w("HAND-VERIFICATION SAMPLES")
    w("=" * 100)
    w("")
    w("  The requested 8-per-cell layout collapses to two populated cells.")
    w("  Cells DEFERS/zero and DEFERS/non-zero contain no rows.")
    w("")
    w("  For the zero-scoring cell the 8 shown are the most referral-heavy rows in")
    w("  the group -- i.e. the BEST available candidates for a DEFERS label. If the")
    w("  COMMITS call looks wrong anywhere, it will look wrong here first.")
    w("")

    cells = [
        ("COMMITS x score == 0.0", df[(df.defer_label == "COMMITS") & df.zero],
         "referral_density"),
        ("COMMITS x score >  0.0", df[(df.defer_label == "COMMITS") & ~df.zero],
         "referral_density"),
        ("DEFERS  x score == 0.0", df[(df.defer_label == "DEFERS") & df.zero], None),
        ("DEFERS  x score >  0.0", df[(df.defer_label == "DEFERS") & ~df.zero], None),
    ]

    for title, sub, sort_by in cells:
        w("")
        w("=" * 100)
        w(f"CELL: {title}   (n = {len(sub)})")
        w("=" * 100)
        if len(sub) == 0:
            w("")
            w("  EMPTY -- no rows received this label.")
            continue
        picks = sub.sort_values(sort_by, ascending=False).head(8) if sort_by \
            else sub.head(8)
        for idx, row in picks.iterrows():
            w("")
            w("-" * 100)
            w(f"[CSV row {idx}]  ragas={row['ragas_answer_relevancy']:.4f}  "
              f"deepeval={row['deepeval_answer_relevancy']:.4f}  "
              f"referral_phrases={row['n_referral']}  ends_in_?={row['endsq']}")
            w(f"  CLASSIFIER: {row['defer_label']}")
            w(f"  REASON    : {row['defer_reason']}")
            w("-" * 100)
            w("QUESTION:")
            w(textwrap.fill(str(row["Question"]), 96,
                            initial_indent="    ", subsequent_indent="    "))
            w("")
            w("BOT ANSWER (full text):")
            for para in str(row["Bot Answer"]).split("\n"):
                if para.strip():
                    w(textwrap.fill(para, 96, initial_indent="    ",
                                    subsequent_indent="    "))
                else:
                    w("")
            w("")
            tail = str(row["Bot Answer"]).rstrip()[-120:]
            w(f"  >> FINAL 120 CHARS (verbatim): ...{tail!r}")

    lines.append("")
    REVIEW_TXT.write_text("\n".join(lines))

    print("\n".join(lines[:80]))
    print(f"\n... full review written to {REVIEW_TXT}")
    print(f"per-row table written to {CROSSTAB_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
