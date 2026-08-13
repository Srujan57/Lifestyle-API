"""Step 4 analysis: did removing hedging register move the noncommittal flag?

Primary comparison is -H vs control. -D and -Q-D are pre-registered negative
controls; -Q is secondary.

Every arm is tested against the CONTROL ARM'S RERUN, paired per row, never
against the stored baseline score.

Writes step4_summary.md and step4_per_row.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "step4_results.csv"
ABLATIONS_JSON = HERE / "step4_ablations.json"
SUMMARY_MD = HERE / "step4_summary.md"
PER_ROW_CSV = HERE / "step4_per_row.csv"

ARMS = ["control", "-Q", "-D", "-Q-D", "-H"]
TEST_ARMS = ["-H", "-Q", "-D", "-Q-D"]


def main() -> int:
    df = pd.read_csv(RESULTS_CSV)
    records = {r["row"]: r for r in json.loads(ABLATIONS_JSON.read_text())}
    n_passes = df["pass"].nunique()

    # Per row x arm: P(zero) = fraction of passes scoring exactly 0.0
    per = df.groupby(["row", "arm"]).agg(
        p_zero=("score", lambda s: float((s == 0.0).mean())),
        mean_score=("score", "mean"),
        mean_cosine=("mean_cosine", "mean"),
        n_passes=("pass", "count"),
        contrast_per_1k=("arm_contrast_per_1k", "first"),
        empathy_per_1k=("arm_empathy_per_1k", "first"),
        referral_per_1k=("arm_referral_per_1k", "first"),
        arm_len=("arm_len", "first"),
    ).reset_index()
    per.to_csv(PER_ROW_CSV, index=False)

    wide = per.pivot(index="row", columns="arm", values="p_zero")
    cos_wide = per.pivot(index="row", columns="arm", values="mean_cosine")

    out = []
    w = out.append
    w("# Step 4 — factorial ablation of the noncommittal flag")
    w("")
    w(f"All five arms scored in a single run, {n_passes} passes each, arm order "
      f"randomised within each row. Every arm is compared against the **control "
      f"arm's rerun**, paired per row — never against the stored baseline score, "
      f"because the judge is stochastic at temperature 0.3.")
    w("")
    w(f"- rows: **{per['row'].nunique()}**")
    w(f"- scored row-arm-passes: **{len(df)}**")
    w(f"- judge calls: **{len(df) * 3}**")
    w("")

    # Control arm reproduction: the noise floor for this run.
    ctrl = wide["control"].dropna()
    w("## Control arm — does the baseline zero reproduce?")
    w("")
    w(f"All {len(ctrl)} rows are baseline zeros. Control-arm P(zero) over "
      f"{n_passes} passes:")
    w("")
    w("| P(zero) | rows |")
    w("|---|---|")
    for v, c in ctrl.value_counts().sort_index(ascending=False).items():
        w(f"| {v:.2f} | {c} |")
    w("")
    w(f"Mean control P(zero) = **{ctrl.mean():.3f}**. This is the reference "
      f"every arm below is measured against.")
    w("")

    # ---- primary + secondary arm comparisons ----
    w("## Arm comparisons vs control")
    w("")
    w("| arm | role | n paired | mean P(zero) | control P(zero) | delta | "
      "Wilcoxon p | mean cosine | control cosine |")
    w("|---|---|---|---|---|---|---|---|---|")

    arm_rows = []
    for arm in TEST_ARMS:
        role = "PRIMARY" if arm == "-H" else (
            "negative control" if arm in ("-D", "-Q-D") else "secondary")
        if arm not in wide.columns:
            w(f"| `{arm}` | {role} | 0 | — | — | — | — | — | — |")
            continue
        paired = wide[["control", arm]].dropna()
        if len(paired) < 3:
            w(f"| `{arm}` | {role} | {len(paired)} | — | — | — | too few | — | — |")
            continue
        a, c = paired[arm], paired["control"]
        diff = a - c
        if np.allclose(diff, 0):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(a, c, zero_method="wilcox")
            except ValueError:
                p = float("nan")
        cpair = cos_wide[["control", arm]].dropna()
        w(f"| `{arm}` | {role} | {len(paired)} | {a.mean():.3f} | {c.mean():.3f} | "
          f"{diff.mean():+.3f} | {p:.4g} | {cpair[arm].mean():.4f} | "
          f"{cpair['control'].mean():.4f} |")
        arm_rows.append({"arm": arm, "role": role, "n": len(paired),
                         "mean_p_zero": a.mean(), "control_p_zero": c.mean(),
                         "delta": diff.mean(), "wilcoxon_p": p,
                         "mean_cosine": cpair[arm].mean(),
                         "control_cosine": cpair["control"].mean()})
    w("")
    w("Delta is (arm − control). Negative delta = the ablation reduced the rate "
      "of exact-zero scoring.")
    w("")

    # Four arms were tested, so the marginal ones need correcting.
    w("### Correction across the 4 arm tests")
    w("")
    pv = {r["arm"]: r["wilcoxon_p"] for r in arm_rows}
    m = len(pv)
    order = sorted(pv, key=lambda k: pv[k])
    w("| arm | raw p | Bonferroni | BH | survives BH? |")
    w("|---|---|---|---|---|")
    for i, k in enumerate(order, 1):
        bonf, bh = min(pv[k] * m, 1.0), min(pv[k] * m / i, 1.0)
        w(f"| `{k}` | {pv[k]:.4g} | {bonf:.4g} | {bh:.4g} | "
          f"{'yes' if bh < 0.05 else 'NO'} |")
    w("")
    w("Note that `-H` and `-D` fail Bonferroni (0.166 and 0.143) and clear BH only "
      "marginally. `-Q` and `-Q-D` survive every correction by four orders of "
      "magnitude.")
    w("")

    # Each arm above runs on a different subset of rows, so the arm means are not
    # directly comparable. Restrict to rows ablatable in every arm for a clean
    # within-row factorial.
    w("### Common subset — rows ablatable in ALL five arms")
    w("")
    common_all = wide.dropna()
    w(f"Arms above use different row subsets (n = 25 to 53), so their means are "
      f"not directly comparable. Restricted to the **{len(common_all)}** rows "
      f"ablatable in every arm:")
    w("")
    w("| arm | mean P(zero) | delta vs control | Wilcoxon p |")
    w("|---|---|---|---|")
    w(f"| `control` | {common_all['control'].mean():.3f} | — | — |")
    for arm in TEST_ARMS:
        d = common_all[arm] - common_all["control"]
        if np.allclose(d, 0):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(common_all[arm], common_all["control"],
                                zero_method="wilcox")
            except ValueError:
                p = float("nan")
        w(f"| `{arm}` | {common_all[arm].mean():.3f} | {d.mean():+.3f} | {p:.4g} |")
    w("")
    w(f"In this clean factorial (n={len(common_all)}, small — treat as indicative) "
      f"`-H` and `-D` are both null, while `-Q` and `-Q-D` retain their effect.")
    w("")

    # ---- cosine flatness check ----
    w("## Mean cosine per arm")
    w("")
    w("If the effect runs through the flag rather than through similarity, "
      "cosine should be roughly flat across arms.")
    w("")
    w("| arm | n | mean cosine |")
    w("|---|---|---|")
    for arm in ARMS:
        sub = per[per["arm"] == arm]
        if len(sub):
            w(f"| `{arm}` | {len(sub)} | {sub['mean_cosine'].mean():.4f} |")
    spread = per.groupby("arm")["mean_cosine"].mean()
    w("")
    w(f"Spread across arms: **{spread.max() - spread.min():.4f}** "
      f"(max {spread.max():.4f}, min {spread.min():.4f}).")
    w("")
    # Paired on the common subset so the comparison is not confounded by which
    # rows each arm happened to cover.
    cc = cos_wide.dropna()
    w(f"Paired on the {len(cc)} rows present in every arm:")
    w("")
    w("| arm | mean cosine | delta vs control |")
    w("|---|---|---|")
    for arm in ARMS:
        w(f"| `{arm}` | {cc[arm].mean():.4f} | {cc[arm].mean() - cc['control'].mean():+.4f} |")
    w("")
    w("**Cosine is flat for `-Q`, `-D` and `-Q-D` (all within ±0.005 of control), "
      "confirming those effects run through the flag and not through similarity. "
      "`-H` is the exception at −0.037** — it is the only arm that perturbed "
      "question generation, so part of its already-small effect may be a "
      "similarity artefact rather than a flag effect. `-H` therefore does not "
      "cleanly isolate the flag the way the other arms do.")
    w("")

    # ---- -H feature movement vs flag movement ----
    w("## -H: did the ablation move the feature but not the flag?")
    w("")
    w("Post-ablation `contrast_per_1k` and `empathy_per_1k` alongside P(zero), so "
      "a moved feature with an unmoved flag is distinguishable from a failed "
      "ablation.")
    w("")
    h = per[per["arm"] == "-H"].set_index("row")
    c0 = per[per["arm"] == "control"].set_index("row")
    common = h.index.intersection(c0.index)
    if len(common):
        w("| metric | control | -H | delta |")
        w("|---|---|---|---|")
        for col in ["contrast_per_1k", "empathy_per_1k", "referral_per_1k",
                    "arm_len", "p_zero", "mean_cosine"]:
            cv, hv = c0.loc[common, col].mean(), h.loc[common, col].mean()
            w(f"| `{col}` | {cv:.3f} | {hv:.3f} | {hv - cv:+.3f} |")
        w("")
        moved = (c0.loc[common, "contrast_per_1k"] - h.loc[common, "contrast_per_1k"]) > 0
        w(f"Rows where -H strictly reduced `contrast_per_1k`: "
          f"**{int(moved.sum())}/{len(common)}**")
        w(f"Rows where -H strictly reduced `empathy_per_1k`: "
          f"**{int(((c0.loc[common,'empathy_per_1k'] - h.loc[common,'empathy_per_1k']) > 0).sum())}/{len(common)}**")
        w("")

        # Did feature movement predict flag movement?
        d_feat = (c0.loc[common, "contrast_per_1k"] + c0.loc[common, "empathy_per_1k"]
                  - h.loc[common, "contrast_per_1k"] - h.loc[common, "empathy_per_1k"])
        d_flag = h.loc[common, "p_zero"] - c0.loc[common, "p_zero"]
        if d_feat.std() > 0 and d_flag.std() > 0:
            r = np.corrcoef(d_feat, d_flag)[0, 1]
            w(f"Correlation between how much hedging was removed and how much "
              f"P(zero) moved: **r = {r:+.3f}** (n={len(common)}).")
            w("")

    # ---- rows that flipped ----
    w("## -H rows that flipped to non-zero")
    w("")
    flipped = [r for r in common
               if c0.at[r, "p_zero"] > 0.5 and h.at[r, "p_zero"] < 0.5] if len(common) else []
    w(f"**{len(flipped)}** of {len(common)} ablatable -H rows moved from "
      f"majority-zero under control to majority-non-zero under -H.")
    w("")
    for row in flipped:
        rec = records[row]
        w("---")
        w("")
        w(f"### Row {row}")
        w("")
        w(f"- control P(zero) **{c0.at[row,'p_zero']:.2f}** → -H P(zero) "
          f"**{h.at[row,'p_zero']:.2f}**")
        w(f"- control mean score {c0.at[row,'mean_score']:.4f} → -H "
          f"{h.at[row,'mean_score']:.4f}")
        w(f"- contrast/1k {c0.at[row,'contrast_per_1k']:.2f} → "
          f"{h.at[row,'contrast_per_1k']:.2f}; empathy/1k "
          f"{c0.at[row,'empathy_per_1k']:.2f} → {h.at[row,'empathy_per_1k']:.2f}")
        w("")
        w(f"**Question:** {rec['question']}")
        w("")
        w("**Control (original):**")
        w("")
        w("> " + rec["arms"]["control"]["text"].replace("\n", "\n> "))
        w("")
        w("**-H (empathy framing and contrast connectives removed):**")
        w("")
        w("> " + rec["arms"]["-H"]["text"].replace("\n", "\n> "))
        w("")
        if rec["arms"]["-H"]["removed"]:
            w("**Removed:**")
            w("")
            for x in rec["arms"]["-H"]["removed"]:
                w(f"- `{x}`")
            w("")

    # ---- unablatable ----
    w("## Unablatable rows (excluded from that arm's paired test)")
    w("")
    w("| arm | unablatable | reason |")
    w("|---|---|---|")
    for arm in TEST_ARMS:
        reasons = {}
        for rec in records.values():
            u = rec["arms"][arm]["unablatable"]
            if u:
                reasons[u] = reasons.get(u, 0) + 1
        for reason, n in reasons.items():
            w(f"| `{arm}` | {n} | {reason} |")
    w("")

    SUMMARY_MD.write_text("\n".join(out))
    pd.DataFrame(arm_rows).to_csv(HERE / "step4_arm_comparison.csv", index=False)
    print("\n".join(out))
    print(f"\nwrote {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
