"""Diagnostic: why are TruLens groundedness scores low?

Summarisation over text that already exists. Zero API calls. TruLens is NOT
re-run and nothing under evals/TruLens/ is written to -- both files are opened
read-only.

Inputs (read-only):
  evals/TruLens/trulens_scored_results.csv  -- scores + per-claim reasoning
  evals/TruLens/flagged_records.json        -- same 59 records WITH retrieved_context

The reasoning column is a repeated block of
    Criteria: <claim extracted from the bot answer>
    Supporting Evidence: <quote from context | NOTHING FOUND>
one block per (claim x context chunk), so criteria repeat. They are deduped,
keeping the best evidence seen for each claim.

Two classifications are produced and kept separate:
  * FAILURE TYPE  -- what kind of text the unsupported claim is. Derived from the
    corpus by inspection, then encoded as rules.
  * MECHANISM     -- why it went unsupported, using the retrieved_context.

Both are rule-based, not hand-adjudicated. Rows the rules cannot separate are
reported as unclassified rather than forced into a bucket.

Writes results/table10_groundedness_failure_types.csv,
results/step6_groundedness_claims.csv, and step6_groundedness_report.txt.
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TL = REPO / "back-end" / "evals" / "TruLens"
SCORED = TL / "trulens_scored_results.csv"
FLAGGED = TL / "flagged_records.json"
RESULTS = HERE / "results"
REPORT = HERE / "step6_groundedness_report.txt"

PAIR_RE = re.compile(
    r"Criteria:\s*(.*?)\s*\nSupporting Evidence:\s*(.*?)(?=\nCriteria:|\Z)", re.S)

STOP = set("""a an the and or but if of to in on for with without at by from as is are was
were be been being it its this that these those you your yours they them their we our us i
me my can could may might will would should shall do does did have has had not no so than
then there here what which who whom when where why how all any both each few more most other
some such only own same too very s t just also into out up down over under again further""".split())


def content_words(t: str) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", str(t).lower()) if w not in STOP}


# ---------------------------------------------------------------------------
# FAILURE TYPES -- derived by reading the unsupported criteria, then encoded.
# Order matters: first match wins, most structural first.
# ---------------------------------------------------------------------------
GENERIC_WELLNESS = re.compile(
    r"\bhydrat|drink (?:plenty|enough|water)|\bstress\b|\bsleep\b|\broutine\b"
    r"|celebrate|small victories|workout buddy|support group|\bjoin(?:ing)? a group"
    r"|track your progress|\bmindful|meditation|deep breathing|be patient"
    r"|listen to your body|set realistic|stay motivated|\bconsistency\b", re.I)
NUTRITION_ENUM = re.compile(
    r"lean protein|whole grain|healthy fats|\bfruits? and vegetables?\b|\bfiber\b"
    r"|\bavocado|\bnuts\b|\bseeds\b|olive oil|quinoa|brown rice|\blegumes?\b"
    r"|\bbeans\b|omega|\bsalmon\b|\bchicken\b|\bturkey\b|portion", re.I)
EXERCISE_MODALITY = re.compile(
    r"\bwalking\b|\bswimming\b|\bcycling\b|\byoga\b|\bpilates\b|stationary bike"
    r"|resistance band|bodyweight|\bsquats?\b|\blunges?\b|push-?ups?|\bstretching\b"
    r"|\btai chi\b|water aerobics|\bjogging\b|\belliptical\b", re.I)
REFERRAL = re.compile(
    r"care team|healthcare team|health care team|\bdoctor\b|\bphysician\b|oncolog"
    r"|\bdietitian\b|consult|\bprovider\b|speak (?:with|to)|talk (?:with|to)"
    r"|professional guidance|\btrainer\b", re.I)
CANCER_SPECIFIC = re.compile(
    r"\bcancer\b|\bsurvivor|\btreatment\b|\bchemo|\bradiation\b|\btumou?r\b"
    r"|\bmetasta|\blymphedema\b|\bneuropathy\b|\brecurrence\b|hormone therapy"
    r"|\boncolog", re.I)
HEADER_LIKE = re.compile(r"^\**[A-Z][\w\s/&-]{1,34}\**\s*:\s*$")
# Added after a first pass left 31% in "other" -- these were read out of that
# bucket and encoded, same derive-then-encode method as the categories above.
OFF_DOMAIN = re.compile(
    r"breastfeed|breast milk|\binfant|\bbaby\b|\bbabies\b|pediatric|\bchild(?:ren)?'?s?\b"
    r"|formula feeding|\bDNA\b|\bgenetic|genomic|\bnutrigenom", re.I)
NAMED_AUTHORITY = re.compile(
    r"world health organization|\bWHO\b|american heart association|\bAHA\b"
    r"|american cancer society|\bACS\b|\bCDC\b|guidelines? recommend"
    r"|\brecommended by\b|are encouraged to", re.I)
NUMERIC_PRESCRIPTION = re.compile(
    r"\d+\s*(?:-\s*\d+\s*)?(?:minutes?|hours?|days?|weeks?|sets?|reps?|repetitions?"
    r"|seconds?|cups?|servings?|times?|steps?|grams?|mg\b|%)"
    r"|\b(?:once|twice|three times)\s+(?:a|per)\s+week"
    r"|at least \d+|aim for \d+", re.I)
MECHANISM_CLAIM = re.compile(
    r"reduce[sd]? inflammation|lower(?:s|ing)? (?:blood pressure|cholesterol)"
    r"|strengthens? (?:your |the )?heart|improve[sd]? (?:circulation|cholesterol)"
    r"|\bplaques?\b|\barteries\b|blood flow|immune system|\bmetabolis|insulin"
    r"|bone density|\bendorphin|\bmuscle mass\b|\bcirculation\b", re.I)
PROGRESSION = re.compile(
    r"\bgradually\b|start (?:slow|small|with)|begin with|as you (?:feel|get|build)"
    r"|work(?:ing)? up to|increase (?:the )?(?:intensity|duration|frequency)"
    r"|build up|over time", re.I)

# Ordered most-specific first; first match wins. A claim can plausibly belong to
# more than one bucket, so the order encodes which label is most informative for
# groundedness (e.g. a specific number must be supported, so a numeric
# prescription is filed as such even when it also names an exercise).
def failure_type(c: str) -> str:
    t = c.strip()
    if t.endswith("?"):
        return "interrogative_intake_question"
    if HEADER_LIKE.match(t) or t.endswith(":"):
        return "list_scaffold_header"
    if len(content_words(t)) <= 2:
        return "fragment_no_content"
    if OFF_DOMAIN.search(t):
        return "off_domain_topic"
    if NAMED_AUTHORITY.search(t):
        return "named_authority_attribution"
    if NUMERIC_PRESCRIPTION.search(t):
        return "numeric_prescription"
    if MECHANISM_CLAIM.search(t):
        return "physiological_mechanism_claim"
    if REFERRAL.search(t):
        return "clinician_referral"
    if CANCER_SPECIFIC.search(t):
        return "cancer_specific_claim"
    if NUTRITION_ENUM.search(t):
        return "nutrition_enumeration"
    if EXERCISE_MODALITY.search(t):
        return "exercise_modality_description"
    if PROGRESSION.search(t):
        return "progression_pacing_guidance"
    if GENERIC_WELLNESS.search(t):
        return "generic_wellness_advice"
    return "other_unclassified"


NO_FACTUAL_CONTENT = {"interrogative_intake_question", "list_scaffold_header",
                      "fragment_no_content"}
# On-topic for this corpus (cancer + exercise literature). Used only to separate
# "the corpus should have covered this" from "the model volunteered outside
# knowledge" -- both look identical as low overlap.
ON_TOPIC = CANCER_SPECIFIC


def mechanism(claim: str, ftype: str, ctx_words: set) -> tuple:
    """Return (mechanism, overlap). Rule-based; see report for stated limits."""
    if ftype in NO_FACTUAL_CONTENT:
        return "no_factual_content_counted_as_claim", np.nan
    cw = content_words(claim)
    if not cw:
        return "no_factual_content_counted_as_claim", np.nan
    ov = len(cw & ctx_words) / len(cw)
    if ov >= 0.60:
        return "present_but_paraphrased_judge_strictness", ov
    if ov < 0.30:
        if ON_TOPIC.search(claim):
            return "absent_from_context_retrieval_problem", ov
        return "general_knowledge_added_grounding_problem", ov
    return "unclassified_intermediate_overlap", ov


def main() -> int:
    d = pd.read_csv(SCORED)
    flagged = {r["question"]: r for r in json.loads(FLAGGED.read_text())}

    claims = []
    for i, r in d.iterrows():
        best = {}
        for crit, ev in PAIR_RE.findall(str(r.groundedness_reasoning)):
            crit, ev = crit.strip(), ev.strip()
            if crit not in best or (best[crit].upper().startswith("NOTHING")
                                    and not ev.upper().startswith("NOTHING")):
                best[crit] = ev
        ctx = " ".join(flagged[r.question]["retrieved_context"])
        ctx_words = content_words(ctx)
        for crit, ev in best.items():
            uns = ev.upper().startswith("NOTHING FOUND")
            ft = failure_type(crit)
            mech, ov = mechanism(crit, ft, ctx_words) if uns else ("supported", np.nan)
            claims.append({
                "row": i, "question": r.question, "category": r.category,
                "source_pdf": r.source_pdf, "groundedness_score": r.groundedness_score,
                "claim": crit, "evidence": ev, "unsupported": uns,
                "failure_type": ft if uns else "", "mechanism": mech,
                "context_overlap": ov,
            })
    C = pd.DataFrame(claims)
    C.to_csv(RESULTS / "step6_groundedness_claims.csv", index=False)
    U = C[C.unsupported]

    L, w = [], None
    L = []
    w = L.append
    w("=" * 100)
    w("TRULENS GROUNDEDNESS -- WHY THE SCORES ARE LOW")
    w("=" * 100)
    w(f"source (read-only): {SCORED}")
    w(f"context joined from: {FLAGGED}")
    w("TruLens was NOT re-run. No API calls. Nothing under evals/TruLens/ written.")
    w("")

    # ---------------------------------------------------------------- 1
    w("#" * 100)
    w("# SECTION 1 -- n AND SCORE DISTRIBUTION")
    w("#" * 100)
    w("")
    s = d.groundedness_score
    w(f"  n = {len(d)} rows")
    w(f"  mean {s.mean():.4f}   median {s.median():.4f}   std {s.std():.4f}")
    w(f"  min  {s.min():.4f}   max {s.max():.4f}")
    w("")
    w("  distribution:")
    bins = pd.cut(s, [-.001, .2, .4, .6, .8, 1.0])
    for b, n in bins.value_counts().sort_index().items():
        w(f"    {str(b):<14s} {n:>3d}  {'#' * n}")
    w("")
    w("  *** SELECTION CAVEAT -- READ BEFORE QUOTING ANY NUMBER ABOVE ***")
    w("  These 59 rows come from flagged_records.json, i.e. they were SELECTED ON")
    w("  OUTCOME (records already flagged as problematic on ragas faithfulness /")
    w("  context recall). This is a DIAGNOSTIC sample, NOT a prevalence estimate.")
    w("  The mean of 0.284 is the mean among rows chosen for being bad; it says")
    w("  nothing about the groundedness of the pipeline overall, and the maximum")
    w(f"  observed score is {s.max():.2f} precisely because high scorers were never")
    w("  eligible for inclusion. Do not report these as 'groundedness is 0.28'.")
    w("")
    w(f"  Claims parsed: {len(C)} distinct (claim x row) pairs, "
      f"mean {len(C)/len(d):.1f} per row.")
    w(f"  Unsupported ('NOTHING FOUND'): {int(C.unsupported.sum())}/{len(C)} "
      f"= {100*C.unsupported.mean():.1f}%")
    w("")

    # ---------------------------------------------------------------- 2
    w("")
    w("#" * 100)
    w("# SECTION 2 -- FAILURE TYPES (derived from the reasoning text, not a preset list)")
    w("#" * 100)
    w("")
    w("  Categories were derived by reading the unsupported criteria, then encoded")
    w("  as rules. Counts are of UNSUPPORTED claims.")
    w("")
    ft = U.failure_type.value_counts()
    w(f"  {'failure_type':<38s} {'claims':>7s} {'%':>7s} {'rows':>6s}")
    for k, n in ft.items():
        nr = U[U.failure_type == k].row.nunique()
        w(f"  {k:<38s} {n:>7d} {100*n/len(U):>6.1f}% {nr:>6d}")
    w(f"  {'TOTAL':<38s} {len(U):>7d}")
    w("")
    for k in ft.index:
        ex = U[U.failure_type == k].iloc[0]
        w(f"  --- {k}  (n={ft[k]}) ---")
        w(f"      verbatim: {ex.claim[:300]!r}")
        w(f"      from row {ex.row}, score {ex.groundedness_score:.3f}")
        w("")

    # ---------------------------------------------------------------- 3
    w("")
    w("#" * 100)
    w("# SECTION 3 -- MECHANISM: why each unsupported claim went unsupported")
    w("#" * 100)
    w("")
    w("  Rule: content-word overlap between the claim and the row's retrieved_context.")
    w("    overlap >= 0.60            -> present but paraphrased (judge strictness)")
    w("    overlap <  0.30, on-topic  -> absent from context (retrieval problem)")
    w("    overlap <  0.30, off-topic -> general knowledge added (grounding problem)")
    w("    0.30 <= overlap < 0.60     -> UNCLASSIFIED (the rule cannot separate these)")
    w("    no factual content         -> counted as a claim despite asserting nothing")
    w("")
    w("  LIMIT: on-topic vs off-topic is keyed on cancer/treatment vocabulary, so a")
    w("  generic-wellness sentence that happens to name cancer is filed as retrieval.")
    w("  This split is a heuristic, not an adjudication.")
    w("")
    mc = U.mechanism.value_counts()
    w(f"  {'mechanism':<48s} {'claims':>7s} {'%':>7s}")
    for k, n in mc.items():
        w(f"  {k:<48s} {n:>7d} {100*n/len(U):>6.1f}%")
    w(f"  {'TOTAL':<48s} {len(U):>7d}")
    w("")
    w(f"  COULD NOT CLASSIFY: "
      f"{int(mc.get('unclassified_intermediate_overlap', 0))} claims with "
      f"intermediate overlap (0.30-0.60).")
    w("")
    w("  failure_type x mechanism:")
    ct = pd.crosstab(U.failure_type, U.mechanism)
    for line in ct.to_string().split("\n"):
        w("    " + line)
    w("")

    # ---------------------------------------------------------------- 4
    w("")
    w("#" * 100)
    w("# SECTION 4 -- SOURCE DOCUMENT AND QUESTION TYPE vs GROUNDEDNESS")
    w("#" * 100)
    w("")
    w(f"  source_pdf is NULL for {int(d.source_pdf.isna().sum())}/{len(d)} rows "
      f"(the deepeval-generated 'rag_grounded' set carries no source_pdf).")
    w("")
    w("  by source_pdf:")
    g = d.groupby(d.source_pdf.fillna("(none - rag_grounded)"))["groundedness_score"]
    for k, sub in g:
        w(f"    {str(k)[:58]:<60s} n={len(sub):>3d}  mean={sub.mean():.4f}  "
          f"min={sub.min():.4f}  max={sub.max():.4f}")
    w("")
    w("  by category:")
    for k, sub in d.groupby("category")["groundedness_score"]:
        w(f"    {k:<38s} n={len(sub):>3d}  mean={sub.mean():.4f}  "
          f"min={sub.min():.4f}  max={sub.max():.4f}")
    w("")
    if d.category.nunique() > 1:
        from scipy.stats import kruskal
        grps = [g.values for _, g in d.groupby("category")["groundedness_score"]
                if len(g) >= 2]
        if len(grps) >= 2:
            h, p = kruskal(*grps)
            w(f"  Kruskal-Wallis across categories (groups with n>=2): "
              f"H={h:.3f}, p={p:.4g}")
    w("")
    w("  Unsupported-claim RATE by category (claims, not rows):")
    for k, sub in C.groupby("category"):
        w(f"    {k:<38s} claims={len(sub):>4d}  unsupported={sub.unsupported.mean():.3f}")
    w("")

    # ---------------------------------------------------------------- 5
    w("")
    w("#" * 100)
    w("# SECTION 5 -- FULL REASONING TEXT, 10 LOWEST-SCORING ROWS")
    w("#" * 100)
    w("")
    low = d.nsmallest(10, "groundedness_score")
    for i, r in low.iterrows():
        w("=" * 100)
        w(f"ROW {i}   groundedness = {r.groundedness_score:.4f}   "
          f"context_relevance = {r.context_relevance_score:.4f}")
        w(f"  category   : {r.category}")
        w(f"  source_pdf : {r.source_pdf}")
        w(f"  QUESTION   : {r.question}")
        w("-" * 100)
        w("BOT ANSWER:")
        for p in str(r.bot_answer).split("\n"):
            w(textwrap.fill(p, 96) if p.strip() else "")
        w("-" * 100)
        w("GROUNDEDNESS REASONING (verbatim, deduped to distinct criteria):")
        best = {}
        for crit, ev in PAIR_RE.findall(str(r.groundedness_reasoning)):
            crit, ev = crit.strip(), ev.strip()
            if crit not in best or (best[crit].upper().startswith("NOTHING")
                                    and not ev.upper().startswith("NOTHING")):
                best[crit] = ev
        for crit, ev in best.items():
            w("")
            w("  Criteria: " + textwrap.fill(crit, 92, subsequent_indent="    "))
            w("  Supporting Evidence: " + textwrap.fill(
                ev, 92, subsequent_indent="    "))
        w("")
    REPORT.write_text("\n".join(L))

    # ---- table10 ----
    t10 = []
    for k, n in ft.items():
        sub = U[U.failure_type == k]
        mm = sub.mechanism.value_counts()
        t10.append({
            "failure_type": k,
            "n_unsupported_claims": int(n),
            "pct_of_unsupported": 100 * n / len(U),
            "n_rows_affected": int(sub.row.nunique()),
            "mech_no_factual_content": int(mm.get("no_factual_content_counted_as_claim", 0)),
            "mech_absent_retrieval_problem": int(mm.get("absent_from_context_retrieval_problem", 0)),
            "mech_general_knowledge_grounding_problem": int(mm.get("general_knowledge_added_grounding_problem", 0)),
            "mech_paraphrased_judge_strictness": int(mm.get("present_but_paraphrased_judge_strictness", 0)),
            "mech_unclassified": int(mm.get("unclassified_intermediate_overlap", 0)),
            "mean_context_overlap": sub.context_overlap.mean(),
            "example_verbatim": sub.iloc[0].claim[:300],
            "selection_caveat": ("rows selected on outcome from flagged_records.json; "
                                 "diagnostic sample, NOT a prevalence estimate"),
        })
    pd.DataFrame(t10).to_csv(RESULTS / "table10_groundedness_failure_types.csv",
                             index=False)

    print("\n".join(L[:150]))
    print(f"\n... full report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
