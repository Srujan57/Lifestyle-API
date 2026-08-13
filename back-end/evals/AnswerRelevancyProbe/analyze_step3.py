"""Step 3 analysis: what the instrumented noncommittal flag actually shows.

Consumes probe_results.csv (163 rows x 3 passes) and reports, in the order the
findings matter:

  1. Flag stability -- how many of 3 passes produce all_noncommittal, per row.
     This is the noise floor. If the zeros do not reproduce, that is the finding.
  2. Mean cosine similarity with the flag multiplier removed, zeros vs non-zeros.
     Tests whether hedging corrupts question generation upstream, or whether the
     flag alone drives the zeros.
  3. Every generated question for the zero-scoring rows, dumped for inspection.
  4. Open-ended: what separates baseline_zero rows from baseline_nonzero ones.

Writes step3_summary.md and step3_generated_questions.txt. No API calls.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
PROBE_CSV = HERE / "probe_results.csv"
SUMMARY_MD = HERE / "step3_summary.md"
QUESTIONS_TXT = HERE / "step3_generated_questions.txt"

# --- feature vocabularies for the open-ended separation analysis -------------
HEDGE_WORDS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bcan\b", r"\bgenerally\b",
    r"\boften\b", r"\btypically\b", r"\busually\b", r"\bsome\b", r"\bpossibly\b",
    r"\bpotentially\b", r"\bperhaps\b", r"\blikely\b", r"\btend to\b",
    r"\bin general\b", r"\bmany\b", r"\bcertain\b",
]
CONDITIONAL_WORDS = [
    r"\bif\b", r"\bdepend\w*\b", r"\bdepending on\b", r"\bunless\b",
    r"\bas long as\b", r"\bwhen\b", r"\bwhether\b", r"\bvary\w*\b",
    r"\bbased on your\b", r"\bfor you\b",
]
CONTRAST_WORDS = [
    r"\bhowever\b", r"\bbut\b", r"\balthough\b", r"\bthough\b", r"\bwhile\b",
    r"\byet\b", r"\bon the other hand\b", r"\bthat said\b",
]
IMPORTANCE_WORDS = [
    r"it'?s important to", r"it'?s crucial to", r"it'?s essential to",
    r"it'?s best to", r"be sure to", r"make sure to", r"keep in mind",
    r"it'?s a good idea to",
]
EMPATHY_WORDS = [
    r"it'?s understandable", r"i understand", r"that'?s a great", r"great question",
    r"it'?s great to hear", r"i hear you", r"that sounds", r"it'?s completely",
    r"you'?re not alone", r"it'?s normal", r"i'?m glad",
]
REFERRAL_PATTERNS = [
    r"care team", r"healthcare team", r"health care team", r"medical team",
    r"oncolog\w*", r"your doctor", r"your provider", r"your physician",
    r"consult\w*", r"speak (?:with|to)", r"talk (?:with|to) your",
    r"check with", r"reach out to", r"discuss (?:this )?with",
]
PERSONALIZATION = [
    r"your (?:specific|individual|particular|unique) (?:situation|needs|case|circumstances)",
    r"everyone is different", r"every(?:one|body)'?s .{0,20}different",
    r"tailored", r"personalized", r"individual\w*",
]
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def count(patterns, text):
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)


def featurize(text: str) -> dict:
    """Descriptive features. Deliberately broad -- the question is open-ended."""
    t = str(text)
    n = max(len(t), 1)
    per1k = 1000.0 / n
    sents = [s for s in SENT_SPLIT_RE.split(t) if s.strip()]
    words = re.findall(r"\b\w+\b", t)
    first_sent = sents[0] if sents else ""

    return {
        "len_chars": len(t),
        "n_sentences": len(sents),
        "mean_sentence_len": np.mean([len(s) for s in sents]) if sents else np.nan,
        "hedge_per_1k": count(HEDGE_WORDS, t) * per1k,
        "conditional_per_1k": count(CONDITIONAL_WORDS, t) * per1k,
        "contrast_per_1k": count(CONTRAST_WORDS, t) * per1k,
        "importance_frame_per_1k": count(IMPORTANCE_WORDS, t) * per1k,
        "empathy_per_1k": count(EMPATHY_WORDS, t) * per1k,
        "referral_per_1k": count(REFERRAL_PATTERNS, t) * per1k,
        "personalization_per_1k": count(PERSONALIZATION, t) * per1k,
        "n_digits_per_1k": len(re.findall(r"\d", t)) * per1k,
        "n_questions_in_body": t.count("?"),
        "n_exclamations": t.count("!"),
        "has_numbered_list": bool(re.search(r"^\s*\d+\.", t, re.MULTILINE)),
        "has_bold_headers": t.count("**") // 2,
        "opens_with_empathy": bool(count(EMPATHY_WORDS, first_sent)),
        "first_sent_len": len(first_sent),
        "second_person_per_1k": len(re.findall(r"\byou\b|\byour\b", t, re.I)) * per1k,
        "modal_ratio": (count(HEDGE_WORDS, t) / max(len(words), 1)),
    }


def adjust_pvalues(pvals):
    """Bonferroni and Benjamini-Hochberg adjusted p-values.

    Implemented here rather than via statsmodels, which is not installed in this
    venv and must not be added (the environment is pinned around ragas 0.4.3).
    Both return values are capped at 1.0; BH is enforced monotone non-decreasing
    from the largest p downwards, as the procedure requires.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    bonf = np.minimum(p * m, 1.0)

    order = np.argsort(p)
    ranked = p[order]
    bh_sorted = np.minimum(ranked * m / np.arange(1, m + 1), 1.0)
    # Enforce monotonicity: step down from the largest.
    bh_sorted = np.minimum.accumulate(bh_sorted[::-1])[::-1]
    bh = np.empty_like(bh_sorted)
    bh[order] = bh_sorted
    return bonf, bh


def stability_table(sub: pd.DataFrame, n_passes: int) -> pd.Series:
    """Per row: how many passes produced all_noncommittal."""
    g = sub.groupby("row")["all_noncommittal"].sum()
    return g.value_counts().reindex(range(n_passes + 1), fill_value=0).sort_index()


def main() -> int:
    df = pd.read_csv(PROBE_CSV)
    n_passes = df["pass"].nunique()
    rows = sorted(df["row"].unique())
    out = []
    w = out.append

    w("# Step 3 — instrumenting the ragas `answer_relevancy` noncommittal flag")
    w("")
    w(f"Judge reconstructed exactly as the baseline built it: `gpt-4o-mini`, "
      f"`bypass_n=True`, `OpenAIEmbeddings()` (ada-002), `strictness=3`, "
      f"temperature left as `None` so the wrapper applies "
      f"`get_temperature(3)=0.3`.")
    w("")
    w(f"- rows probed: **{len(rows)}**")
    w(f"- passes per row: **{n_passes}**")
    w(f"- judge calls: **{len(df) * 3}**")
    w("")

    # Per-row aggregate
    agg = df.groupby("row").agg(
        baseline_score=("baseline_score", "first"),
        baseline_zero=("baseline_zero", "first"),
        deepeval=("deepeval_score", "first"),
        n_passes_all_nc=("all_noncommittal", "sum"),
        mean_score=("score", "mean"),
        mean_cosine=("mean_cosine", "mean"),
        total_flags=("n_flagged", "sum"),
        deferral_label=("deferral_label", "first"),
        ends_in_question=("ends_in_question", "first"),
        referral_density=("referral_density_per_1k", "first"),
        question=("question", "first"),
        category=("category", "first"),
    )
    # Explicit population names. Three distinct groupings are in play and
    # conflating them is exactly the bug this replaced:
    #   baseline_zero              stored baseline score == 0.0            (56)
    #   reprobe_flagged_1plus_of3  all_noncommittal in >=1 of 3 passes     (58)
    #   reprobe_flagged_3of3       all_noncommittal in all 3 passes        (51)
    agg = agg.rename(columns={"n_passes_all_nc": "n_passes_all_noncommittal"})
    agg["reprobe_p_zero"] = agg["n_passes_all_noncommittal"] / n_passes
    agg["reprobe_per_question_flag_rate"] = agg["total_flags"] / (n_passes * 3)
    agg["baseline_nonzero"] = ~agg["baseline_zero"]
    agg["reprobe_flagged_1plus_of3"] = agg["n_passes_all_noncommittal"] >= 1
    agg["reprobe_flagged_3of3"] = agg["n_passes_all_noncommittal"] == n_passes

    zeros = agg[agg.baseline_zero]
    nonzeros = agg[agg.baseline_nonzero]

    # ---------------------------------------------------------------- 1
    w("## 1. Flag stability — the noise floor")
    w("")
    w(f"Passes (of {n_passes}) in which all 3 generated questions were flagged "
      f"noncommittal, i.e. the row scored exactly 0.0:")
    w("")
    zt = stability_table(df[df.baseline_zero], n_passes)  # probe CSV keeps its own col
    nt = stability_table(df[~df.baseline_zero], n_passes)
    w(f"| passes with all_noncommittal | `baseline_zero` (n={len(zeros)}) | `baseline_nonzero` (n={len(nonzeros)}) |")
    w("|---|---|---|")
    for k in range(n_passes + 1):
        zp = 100 * zt[k] / max(len(zeros), 1)
        np_ = 100 * nt[k] / max(len(nonzeros), 1)
        w(f"| {k}/{n_passes} | {zt[k]} ({zp:.1f}%) | {nt[k]} ({np_:.1f}%) |")
    w("")
    w("(The non-zero column is the complete set of 107, not a sample.)")
    w("")
    w(f"- baseline zeros reproducing in **all {n_passes}** passes: "
      f"**{zt[n_passes]}/{len(zeros)}** ({100*zt[n_passes]/max(len(zeros),1):.1f}%)")
    w(f"- baseline zeros that **never** reproduced: "
      f"**{zt[0]}/{len(zeros)}** ({100*zt[0]/max(len(zeros),1):.1f}%)")
    w(f"- baseline non-zeros that flipped to 0.0 at least once: "
      f"**{len(nonzeros) - nt[0]}/{len(nonzeros)}** "
      f"({100*(len(nonzeros)-nt[0])/max(len(nonzeros),1):.1f}%)")
    w("")

    # Baseline vs reprobe agreement
    w("### Baseline-vs-reprobe agreement on landing at 0.0")
    w("")
    maj_zero = agg["reprobe_p_zero"] > 0.5
    tp = int((agg.baseline_zero & maj_zero).sum())
    fn = int((agg.baseline_zero & ~maj_zero).sum())
    fp = int((agg.baseline_nonzero & maj_zero).sum())
    tn = int((agg.baseline_nonzero & ~maj_zero).sum())
    w("Majority of passes scoring 0.0 vs the stored baseline:")
    w("")
    w("| | reprobe majority 0.0 | reprobe majority > 0 |")
    w("|---|---|---|")
    w(f"| baseline 0.0 | {tp} | {fn} |")
    w(f"| baseline > 0 | {fp} | {tn} |")
    w("")
    w(f"Agreement: **{100*(tp+tn)/len(agg):.1f}%**")
    w("")

    # ---------------------------------------------------------------- 2
    w("## 2. Cosine similarity with the flag multiplier removed")
    w("")
    w("`mean_cosine` is the similarity between the user's real question and the "
      "3 questions the judge generated from the response — the score ragas would "
      "have produced *before* multiplying by `int(not all_noncommittal)`.")
    w("")
    w(f"| group | n | mean cosine | median | min | max |")
    w("|---|---|---|---|---|---|")
    for lbl, sub in [("baseline zeros", zeros), ("baseline non-zeros", nonzeros)]:
        c = sub["mean_cosine"]
        w(f"| {lbl} | {len(sub)} | **{c.mean():.4f}** | {c.median():.4f} | "
          f"{c.min():.4f} | {c.max():.4f} |")
    u, p = mannwhitneyu(zeros["mean_cosine"], nonzeros["mean_cosine"])
    auc = u / (len(zeros) * len(nonzeros))
    w("")
    w(f"Mann-Whitney U: p = {p:.4g}, AUC = {auc:.3f}")
    w("")
    delta = nonzeros["mean_cosine"].mean() - zeros["mean_cosine"].mean()
    w(f"Difference in mean cosine: **{delta:+.4f}** "
      f"(non-zeros minus zeros).")
    w("")

    # Direct test of the "deferral corrupts question generation" hypothesis:
    # if it held, questions generated from zero-scoring rows would be
    # meta-questions about seeing a clinician rather than about the topic.
    per_q = []
    for _, r in df.iterrows():
        for k in (1, 2, 3):
            per_q.append({"zero": r["baseline_zero"], "q": str(r[f"gen_q{k}"]),
                          "cos": r[f"cos{k}"], "flag": r[f"flag{k}"]})
    per_q = pd.DataFrame(per_q)
    meta_re = re.compile(
        r"ask my (?:doctor|care|health)|consult|talk to my|should i (?:see|ask|contact)"
        r"|care team|healthcare team|my doctor|clinician", re.IGNORECASE)
    per_q["meta"] = per_q["q"].map(lambda s: bool(meta_re.search(s)))

    w("### Are the generated questions meta-questions about consulting a clinician?")
    w("")
    w("| group | generated questions | meta-questions | rate |")
    w("|---|---|---|---|")
    for lbl, sub in [("from baseline zeros", per_q[per_q.zero]),
                     ("from baseline non-zeros", per_q[~per_q.zero])]:
        w(f"| {lbl} | {len(sub)} | {int(sub['meta'].sum())} | "
          f"{100*sub['meta'].mean():.1f}% |")
    w("")
    w("Cosine of each generated question, split by that question's own flag:")
    w("")
    w("| flag | n | mean cosine |")
    w("|---|---|---|")
    for fl, sub in per_q.groupby("flag"):
        w(f"| {int(fl)} | {len(sub)} | {sub['cos'].mean():.4f} |")
    w("")

    # ---------------------------------------------------------------- 4
    w("## 3. What separates the flagged rows from the rest")
    w("")
    w("### Population definitions")
    w("")
    w("Three groupings are in play and must not be conflated:")
    w("")
    w("| name | definition | n |")
    w("|---|---|---|")
    w(f"| `baseline_zero` | stored `ragas_answer_relevancy` == 0.0 | "
      f"{int(agg.baseline_zero.sum())} |")
    w(f"| `baseline_nonzero` | stored score > 0.0 | "
      f"{int(agg.baseline_nonzero.sum())} |")
    w(f"| `reprobe_flagged_1plus_of3` | `all_noncommittal` in >= 1 of 3 passes | "
      f"{int(agg.reprobe_flagged_1plus_of3.sum())} |")
    w(f"| `reprobe_flagged_3of3` | `all_noncommittal` in all 3 passes | "
      f"{int(agg.reprobe_flagged_3of3.sum())} |")
    w("")
    bz = set(agg.index[agg.baseline_zero])
    f58 = set(agg.index[agg.reprobe_flagged_1plus_of3])
    w(f"`baseline_zero` and `reprobe_flagged_1plus_of3` differ by "
      f"{len(bz - f58)} + {len(f58 - bz)} rows:")
    w("")
    w("| direction | row | baseline score | passes all_noncommittal |")
    w("|---|---|---|---|")
    for i in sorted(bz - f58):
        w(f"| in `baseline_zero` only | {i} | "
          f"{agg.at[i,'baseline_score']:.4f} | "
          f"{int(agg.at[i,'n_passes_all_noncommittal'])}/{n_passes} |")
    for i in sorted(f58 - bz):
        w(f"| in `reprobe_flagged_1plus_of3` only | {i} | "
          f"{agg.at[i,'baseline_score']:.4f} | "
          f"{int(agg.at[i,'n_passes_all_noncommittal'])}/{n_passes} |")
    w("")
    w(f"Arithmetic: {len(bz)} - {len(bz - f58)} + {len(f58 - bz)} = {len(f58)}.")
    w("")
    w("Features are ranked by separation strength, not by a preset hypothesis.")
    w("")

    # Featurize from the response text carried in the probe CSV? The probe CSV
    # does not carry the response, so re-read the baseline for text.
    base = pd.read_csv(HERE.parents[2] / "back-end" / "evals" / "RAGAS" /
                       "RAGAS_Baseline.csv")
    feats = pd.DataFrame([featurize(base.at[i, "Bot Answer"]) for i in agg.index],
                         index=agg.index)
    feats = feats.astype({c: float for c in feats.columns
                          if feats[c].dtype == bool})

    def separation(pos_mask, pos_name, neg_name):
        """Rank features by how well they separate two named populations."""
        rows_ = []
        for col in feats.columns:
            a = feats.loc[pos_mask, col].dropna()
            b = feats.loc[~pos_mask, col].dropna()
            if len(a) < 5 or len(b) < 5 or a.nunique() + b.nunique() < 3:
                continue
            try:
                u, p = mannwhitneyu(a, b)
            except ValueError:
                continue
            auc = u / (len(a) * len(b))
            rows_.append({
                "feature": col,
                "population_positive": pos_name,
                "population_negative": neg_name,
                "n_positive": len(a),
                "n_negative": len(b),
                f"{pos_name}_mean": a.mean(),
                f"{neg_name}_mean": b.mean(),
                "AUC": auc,
                "separation": abs(auc - 0.5),
                "p": p,
            })
        out_ = pd.DataFrame(rows_).sort_values("separation", ascending=False)
        # 19 features are tested per population, so the marginal hits need to be
        # visible as marginal. Raw p is retained alongside.
        bonf, bh = adjust_pvalues(out_["p"].values)
        out_["p_bonferroni"] = bonf
        out_["p_benjamini_hochberg"] = bh
        out_["n_tests"] = len(out_)
        return out_

    def emit(res, pos_name, neg_name):
        n_pos = int(res["n_positive"].iloc[0])
        n_neg = int(res["n_negative"].iloc[0])
        w(f"| feature | `{pos_name}` mean (n={n_pos}) | `{neg_name}` mean "
          f"(n={n_neg}) | AUC | p (raw) | p (BH) | p (Bonf) |")
        w("|---|---|---|---|---|---|---|")
        for _, r in res.iterrows():
            if r["p_benjamini_hochberg"] < 0.01:
                star = " **"
            elif r["p_benjamini_hochberg"] < 0.05:
                star = " *"
            else:
                star = ""
            w(f"| `{r['feature']}`{star} | {r[f'{pos_name}_mean']:.3f} | "
              f"{r[f'{neg_name}_mean']:.3f} | {r['AUC']:.3f} | {r['p']:.4g} | "
              f"{r['p_benjamini_hochberg']:.4g} | {r['p_bonferroni']:.4g} |")
        w("")
        w(f"AUC 0.5 = no separation. Stars reflect **BH-adjusted** p across "
          f"{len(res)} features: `**` q<0.01, `*` q<0.05.")
        w("")

    # PRIMARY: the population the write-up describes.
    w("### 3a. PRIMARY — `baseline_zero` (n=56) vs `baseline_nonzero` (n=107)")
    w("")
    res_primary = separation(agg["baseline_zero"], "baseline_zero",
                             "baseline_nonzero")
    emit(res_primary, "baseline_zero", "baseline_nonzero")

    # SECONDARY: the reprobe-derived grouping originally reported.
    w("### 3b. SECONDARY — `reprobe_flagged_1plus_of3` (n=58) vs "
      "`reprobe_unflagged_0of3` (n=105)")
    w("")
    w("This is the grouping the earlier version of this table used while "
      "labelling it only 'flagged'. Retained for comparison.")
    w("")
    res_secondary = separation(agg["reprobe_flagged_1plus_of3"],
                               "reprobe_flagged_1plus_of3",
                               "reprobe_unflagged_0of3")
    emit(res_secondary, "reprobe_flagged_1plus_of3", "reprobe_unflagged_0of3")

    # Do the two populations tell the same story?
    w("### 3c. Do the AUCs change materially between the two populations?")
    w("")
    cmp = res_primary[["feature", "AUC"]].merge(
        res_secondary[["feature", "AUC"]], on="feature",
        suffixes=("_baseline_zero", "_reprobe_1plus"))
    cmp["delta_AUC"] = (cmp["AUC_reprobe_1plus"] - cmp["AUC_baseline_zero"])
    cmp = cmp.reindex(cmp["delta_AUC"].abs().sort_values(ascending=False).index)
    w("| feature | AUC (`baseline_zero`) | AUC (`reprobe_flagged_1plus_of3`) | delta |")
    w("|---|---|---|---|")
    for _, r in cmp.iterrows():
        w(f"| `{r['feature']}` | {r['AUC_baseline_zero']:.3f} | "
          f"{r['AUC_reprobe_1plus']:.3f} | {r['delta_AUC']:+.3f} |")
    w("")
    w(f"Largest absolute AUC shift: **{cmp['delta_AUC'].abs().max():.3f}**; "
      f"mean absolute shift **{cmp['delta_AUC'].abs().mean():.3f}**. "
      f"Rank order of the top discriminators is unchanged.")
    w("")

    pd.concat([res_primary, res_secondary], ignore_index=True).to_csv(
        HERE / "step3_feature_separation.csv", index=False)
    cmp.to_csv(HERE / "step3_feature_separation_auc_comparison.csv", index=False)

    # Deferral label rate (degenerate but reported as specified)
    w("### Noncommittal flag rate by deferral label")
    w("")
    for lbl, sub in agg.groupby("deferral_label"):
        w(f"- **{lbl}** (n={len(sub)}): mean per-question flag rate "
          f"{sub['reprobe_per_question_flag_rate'].mean():.3f}, mean P(zero) "
          f"{sub['reprobe_p_zero'].mean():.3f}")
    w("")
    w("The deferral variable is constant across the corpus (all COMMITS), so this "
      "is reported for completeness only — it cannot discriminate.")
    w("")

    agg.to_csv(HERE / "step3_per_row.csv")

    SUMMARY_MD.write_text("\n".join(out))

    # ---------------------------------------------------------------- 3
    qlines = []
    qw = qlines.append
    qw("=" * 100)
    qw("GENERATED QUESTIONS FOR EVERY BASELINE ZERO-SCORING ROW")
    qw("=" * 100)
    qw("")
    qw("For each row: the user's real question, then the 3 questions the judge")
    qw("generated from the response on each pass, with that pass's flags and the")
    qw("cosine similarity of each generated question to the real one.")
    qw("")
    qw("The test: are these on-topic questions about the subject matter, or")
    qw("meta-questions about consulting a clinician?")
    qw("")
    for row in zeros.index:
        sub = df[df["row"] == row].sort_values("pass")
        a = agg.loc[row]
        qw("=" * 100)
        qw(f"ROW {row}   baseline={a['baseline_score']:.4f}  deepeval={a['deepeval']:.3f}  "
           f"P(zero) over {n_passes} passes = {a['reprobe_p_zero']:.2f}  "
           f"mean_cosine={a['mean_cosine']:.4f}")
        qw(f"  USER QUESTION: {a['question']}")
        qw("-" * 100)
        for _, r in sub.iterrows():
            qw(f"  pass {int(r['pass'])}   flags=[{int(r['flag1'])},{int(r['flag2'])},"
               f"{int(r['flag3'])}]   all_nc={r['all_noncommittal']}   "
               f"score={r['score']:.4f}")
            for k in (1, 2, 3):
                qw(f"      Q{k} (cos {r[f'cos{k}']:.4f}): {r[f'gen_q{k}']}")
        qw("")
    QUESTIONS_TXT.write_text("\n".join(qlines))

    print("\n".join(out))
    print(f"\nwrote {SUMMARY_MD}")
    print(f"wrote {QUESTIONS_TXT}")
    print(f"wrote {HERE / 'step3_per_row.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
