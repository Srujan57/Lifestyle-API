"""Assemble the self-contained results/ folder for external review.

Produces the two clearly separated arm tables (within-arm primary, common-subset
secondary), the supporting per-row CSVs, and copies the Step 3 outputs in.

Cross-arm rule enforced here: the four arms are NEVER emitted in a single ranked
table without their populations. table5 names each arm's population explicitly and
records the row ids; table6 is the only table where arms may be compared to each
other, and says so in its own columns.

All CSVs carry full precision. Rounding happens only in FINDINGS.md.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ARMS = ["control", "-Q", "-D", "-Q-D", "-H"]
TEST_ARMS = ["-Q", "-D", "-Q-D", "-H"]

POPULATION = {
    "-Q": "rows where a trailing question exists (= Step 2 Group A)",
    "-D": "rows containing at least one clinician-referral sentence",
    "-Q-D": "rows where a trailing question and/or a referral sentence exists",
    "-H": "rows with empathy framing or a removable contrast marker",
}
PAIRED_NOTE = ("paired against the SAME rows in the control arm; "
               "control is rescored, never the stored baseline score")


def adjust(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    bonf = np.minimum(p * m, 1.0)
    order = np.argsort(p)
    bh_s = np.minimum(p[order] * m / np.arange(1, m + 1), 1.0)
    bh_s = np.minimum.accumulate(bh_s[::-1])[::-1]
    bh = np.empty_like(bh_s)
    bh[order] = bh_s
    return bonf, bh


def paired_stats(wide_p, wide_c, arm, rows):
    a = wide_p.loc[rows, arm]
    c = wide_p.loc[rows, "control"]
    d = a - c
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            _, p = wilcoxon(a, c, zero_method="wilcox")
        except ValueError:
            p = float("nan")
    return {
        "n": len(rows),
        "mean_p_zero_control": c.mean(),
        "mean_p_zero_ablated": a.mean(),
        "delta_p_zero": d.mean(),
        "wilcoxon_raw_p": p,
        "mean_cosine_control": wide_c.loc[rows, "control"].mean(),
        "mean_cosine_ablated": wide_c.loc[rows, arm].mean(),
        "mean_cosine_delta": (wide_c.loc[rows, arm] - wide_c.loc[rows, "control"]).mean(),
        "rows_included": " ".join(str(int(r)) for r in rows),
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    records = {r["row"]: r for r in json.loads((HERE / "step4_ablations.json").read_text())}

    # --- Step 4 per row x arm, 3 passes (all five arms) ---
    raw = pd.read_csv(HERE / "step4_results.csv")
    ext_path = HERE / "step4_extension_results.csv"
    n_ext_passes = 0
    if ext_path.exists():
        ext = pd.read_csv(ext_path)
        n_ext_passes = ext["pass"].nunique()
        raw_all = pd.concat([raw, ext], ignore_index=True)
    else:
        raw_all = raw

    def per_row(df):
        return df.groupby(["row", "arm"]).agg(
            p_zero=("score", lambda s: float((s == 0.0).mean())),
            mean_score=("score", "mean"),
            mean_cosine=("mean_cosine", "mean"),
            n_passes=("pass", "count"),
            contrast_per_1k=("arm_contrast_per_1k", "first"),
            empathy_per_1k=("arm_empathy_per_1k", "first"),
            referral_per_1k=("arm_referral_per_1k", "first"),
            arm_len=("arm_len", "first"),
        ).reset_index()

    per3 = per_row(raw)                     # 3 passes, all arms
    per3.to_csv(RESULTS / "step4_per_row.csv", index=False)
    w3 = per3.pivot(index="row", columns="arm", values="p_zero")
    c3 = per3.pivot(index="row", columns="arm", values="mean_cosine")

    # -D and control on the -D subset get the full 9 passes.
    d_rows = sorted(r for r in records if not records[r]["arms"]["-D"]["unablatable"])
    per9 = per_row(raw_all[raw_all["row"].isin(d_rows)
                           & raw_all["arm"].isin(["control", "-D"])])
    w9 = per9.pivot(index="row", columns="arm", values="p_zero")
    c9 = per9.pivot(index="row", columns="arm", values="mean_cosine")
    per9.to_csv(RESULTS / "step4_D_extended_per_row.csv", index=False)

    # ---------------- table 5: within-arm (PRIMARY) ----------------
    rows5 = []
    for arm in TEST_ARMS:
        if arm == "-D" and n_ext_passes:
            st = paired_stats(w9, c9, arm, d_rows)
            npass = 3 + n_ext_passes
        else:
            valid = [r for r in w3.index if not pd.isna(w3.at[r, arm])]
            st = paired_stats(w3, c3, arm, valid)
            npass = 3
        st.update({
            "arm": arm,
            "role": {"-H": "PRIMARY (pre-registered)",
                     "-Q": "secondary",
                     "-D": "pre-registered negative control",
                     "-Q-D": "pre-registered negative control"}[arm],
            "population": f"{POPULATION[arm]} (n={st['n']})",
            "paired_control": PAIRED_NOTE,
            "n_passes_per_row": npass,
        })
        rows5.append(st)
    t5 = pd.DataFrame(rows5)
    bonf, bh = adjust(t5["wilcoxon_raw_p"].values)
    t5["p_bonferroni"] = bonf
    t5["p_benjamini_hochberg"] = bh
    t5["n_arm_tests_corrected_over"] = len(t5)
    t5["cross_arm_comparison_valid"] = (
        "NO - each row uses a different population; compare each arm ONLY to its "
        "own paired control column, never to another arm. See table6.")
    t5 = t5[["arm", "role", "population", "n", "n_passes_per_row", "paired_control",
             "mean_p_zero_control", "mean_p_zero_ablated", "delta_p_zero",
             "wilcoxon_raw_p", "p_bonferroni", "p_benjamini_hochberg",
             "n_arm_tests_corrected_over", "mean_cosine_control",
             "mean_cosine_ablated", "mean_cosine_delta",
             "cross_arm_comparison_valid", "rows_included"]]
    t5.to_csv(RESULTS / "table5_within_arm.csv", index=False)

    # ---------------- table 6: common subset (SECONDARY) ----------------
    common = sorted(w3.dropna().index)
    rows6 = []
    for arm in TEST_ARMS:
        st = paired_stats(w3, c3, arm, common)
        st.update({
            "arm": arm,
            "population": f"rows ablatable in ALL five arms (n={len(common)})",
            "paired_control": PAIRED_NOTE,
            "n_passes_per_row": 3,
        })
        rows6.append(st)
    t6 = pd.DataFrame(rows6)
    bonf, bh = adjust(t6["wilcoxon_raw_p"].values)
    t6["p_bonferroni"] = bonf
    t6["p_benjamini_hochberg"] = bh
    t6["cross_arm_comparison_valid"] = (
        f"YES - this is the ONLY valid cross-arm comparison, because all arms "
        f"share the same {len(common)} rows. UNDERPOWERED: n={len(common)}; "
        f"treat as indicative, not confirmatory.")
    t6 = t6[["arm", "population", "n", "n_passes_per_row", "paired_control",
             "mean_p_zero_control", "mean_p_zero_ablated", "delta_p_zero",
             "wilcoxon_raw_p", "p_bonferroni", "p_benjamini_hochberg",
             "mean_cosine_control", "mean_cosine_ablated", "mean_cosine_delta",
             "cross_arm_comparison_valid", "rows_included"]]
    t6.to_csv(RESULTS / "table6_common_subset.csv", index=False)

    # ---------------- -Q-D vs -Q (does D add anything on top of Q?) ----------
    qd_rows = sorted(set(w3.dropna(subset=["-Q", "-Q-D"]).index))
    a, b = w3.loc[qd_rows, "-Q-D"], w3.loc[qd_rows, "-Q"]
    d = a - b
    p = 1.0 if np.allclose(d, 0) else wilcoxon(a, b, zero_method="wilcox")[1]
    pd.DataFrame([{
        "comparison": "-Q-D vs -Q (paired, rows where both arms exist)",
        "n": len(qd_rows), "n_passes_per_row": 3,
        "mean_p_zero_minus_Q": b.mean(),
        "mean_p_zero_minus_QD": a.mean(),
        "delta_QD_minus_Q": d.mean(),
        "wilcoxon_raw_p": p,
        "interpretation": ("delta < 0 means removing deferral ADDS effect on top of "
                           "removing the trailing question; delta ~ 0 means D "
                           "contributes nothing"),
        "rows_included": " ".join(str(int(r)) for r in qd_rows),
    }]).to_csv(RESULTS / "table7_QD_vs_Q.csv", index=False)

    # ---------------- hedging dose-response ----------------
    h = per3[per3.arm == "-H"].set_index("row")
    c0 = per3[per3.arm == "control"].set_index("row")
    idx = h.index.intersection(c0.index)
    dose = pd.DataFrame({
        "row": idx,
        "contrast_per_1k_control": c0.loc[idx, "contrast_per_1k"].values,
        "contrast_per_1k_ablated": h.loc[idx, "contrast_per_1k"].values,
        "empathy_per_1k_control": c0.loc[idx, "empathy_per_1k"].values,
        "empathy_per_1k_ablated": h.loc[idx, "empathy_per_1k"].values,
        "p_zero_control": c0.loc[idx, "p_zero"].values,
        "p_zero_ablated": h.loc[idx, "p_zero"].values,
    })
    dose["hedging_removed"] = ((dose.contrast_per_1k_control - dose.contrast_per_1k_ablated)
                               + (dose.empathy_per_1k_control - dose.empathy_per_1k_ablated))
    dose["delta_p_zero"] = dose.p_zero_ablated - dose.p_zero_control
    r = np.corrcoef(dose.hedging_removed, dose.delta_p_zero)[0, 1]
    dose["pearson_r_hedging_removed_vs_delta_p_zero"] = r
    dose.to_csv(RESULTS / "step4_hedging_dose_response.csv", index=False)

    # ---------------- cosine by arm ----------------
    cos_rows = []
    for arm in ARMS:
        sub = per3[per3.arm == arm]
        pc = c3[["control", arm]].dropna() if arm != "control" else None
        cos_rows.append({
            "arm": arm,
            "n_within_arm": len(sub),
            "mean_cosine_within_arm": sub["mean_cosine"].mean(),
            "n_common_subset": len(common),
            "mean_cosine_common_subset": c3.loc[common, arm].mean(),
            "delta_vs_control_common_subset": (c3.loc[common, arm]
                                               - c3.loc[common, "control"]).mean(),
            "n_paired_own_population": 0 if pc is None else len(pc),
            "mean_cosine_paired_own_population": np.nan if pc is None else pc[arm].mean(),
        })
    pd.DataFrame(cos_rows).to_csv(RESULTS / "step4_cosine_by_arm.csv", index=False)

    # ---------------- Step 3 files ----------------
    shutil.copy(HERE / "probe_results.csv", RESULTS / "probe_raw.csv")
    shutil.copy(HERE / "step3_per_row.csv", RESULTS / "probe_per_row.csv")
    shutil.copy(HERE / "step3_feature_separation.csv",
                RESULTS / "step3_feature_separation.csv")

    print(f"results/ rebuilt: {len(list(RESULTS.glob('*.csv')))} CSVs")
    for f in sorted(RESULTS.glob("*.csv")):
        print(f"   {f.name}")
    print()
    print("table5 (within-arm, primary):")
    print(t5[["arm", "n", "n_passes_per_row", "mean_p_zero_control",
              "mean_p_zero_ablated", "delta_p_zero", "wilcoxon_raw_p",
              "p_bonferroni", "p_benjamini_hochberg"]].to_string(index=False))
    print()
    print("table6 (common subset, secondary):")
    print(t6[["arm", "n", "mean_p_zero_control", "mean_p_zero_ablated",
              "delta_p_zero", "wilcoxon_raw_p"]].to_string(index=False))
    print()
    print("table7 (-Q-D vs -Q):")
    print(f"   n={len(qd_rows)}  -Q={b.mean():.4f}  -Q-D={a.mean():.4f}  "
          f"delta={d.mean():+.4f}  p={p:.4g}")
    print(f"\nhedging dose-response r = {r:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
