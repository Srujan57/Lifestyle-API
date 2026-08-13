# Why 56 baseline rows scored exactly 0.0 on `ragas_answer_relevancy`

Investigation of `back-end/evals/RAGAS/RAGAS_Baseline.csv` (163 rows).
Nothing under `evals/RAGAS/` or `evals/TruLens/` was edited or re-run. All outputs
are additive and live in this folder; full-precision data is in `results/`.

Numbers below are rounded. **Every figure names the CSV column it comes from** —
`file.csv :: column`. The CSVs are authoritative.

---

## Configuration

| item | value |
|---|---|
| ragas | **0.4.3** (unchanged throughout; never upgraded) |
| metric | `ragas.metrics._answer_relevance.ResponseRelevancy` |
| judge LLM | `ChatOpenAI(model="gpt-4o-mini")` wrapped `LangchainLLMWrapper(..., bypass_n=True)` |
| embeddings | `LangchainEmbeddingsWrapper(OpenAIEmbeddings())` → `text-embedding-ada-002` |
| `strictness` | **3** (3 questions generated per response) |
| temperature | **0.3**, not set directly — `temperature=None` is passed, so `LangchainLLMWrapper.agenerate_text` calls `get_temperature(n=3)` which returns 0.3 (`ragas/llms/base.py:71-73`) |
| deferral classifier | `gpt-4o`, temperature 0 — deliberately a *different* model from the judge, so a shared blind spot cannot manufacture the correlation |
| passes: Step 3 probe | 3 per row × 163 rows = 1,467 judge calls |
| passes: Step 4 arms | 3 per row-arm (657 row-arm-passes, 1,971 judge calls) |
| passes: Step 4 `-D` + its control | **9** (3 original + 6 extension; 300 extra row-arm-passes, 900 calls) |
| run dates (UTC) | deferral labels 2026-08-13 16:43 · probe 2026-08-13 17:14 · ablation 2026-08-13 17:29 · `-D` extension 2026-08-13 17:40 |
| judge config source | `evals/RAGAS/run_ragas_eval.py:430-431`, reproduced in `probe_noncommittal.py::build_judge` |

The judge is stochastic, so every measurement is repeated and every comparison is
against a **rescored control**, never the stored baseline value.

---

## The mechanism

```python
# ragas/metrics/_answer_relevance.py :: ResponseRelevancy._calculate_score
score = cosine_sim.mean() * int(not all_noncommittal)
```

If all 3 generated questions are flagged noncommittal the multiplier is 0 and the
embedding similarity is discarded. A row can score exactly 0.0 while being highly
relevant by every other measure.

---

## Result 1 — the zeros are reproducible, not noise

`probe_per_row.csv :: n_passes_all_noncommittal`, `baseline_zero`

| passes with `all_noncommittal` | `baseline_zero` (n=56) | `baseline_nonzero` (n=107) |
|---|---|---|
| 0/3 | 2 (3.6%) | 103 (96.3%) |
| 1/3 | 2 | 3 |
| 2/3 | 1 | 1 |
| **3/3** | **51 (91.1%)** | **0 (0.0%)** |

Baseline-vs-reprobe agreement **96.9%**. Noise floor ~3–4%; the zeros sit far
outside it.

## Result 2 — question generation is intact; the flag alone causes the zeros

`probe_per_row.csv :: mean_cosine`

| group | n | mean cosine |
|---|---|---|
| `baseline_zero` | 56 | **0.9170** |
| `baseline_nonzero` | 107 | **0.9379** |

Gap **0.021**. The zeros would have scored ~0.92 on similarity alone.
**0 of 504** questions generated from zero rows are meta-questions about consulting
a clinician (`probe_raw.csv :: gen_q1..gen_q3`). Row 0 produced *"Is it safe to
exercise with bone metastases?"* nine times at cosine 0.9297 and was flagged every
time.

---

## Result 3 — ablation

### Table 5 (primary) — within-arm, `results/table5_within_arm.csv`

**Each arm has its own population. These rows are NOT comparable to each other.**
Every arm is paired against *the same rows* in the control arm, rescored.

| arm | role | population (`:: population`) | n | passes | control P(0) | ablated P(0) | delta | raw p | Bonf | BH |
|---|---|---|---|---|---|---|---|---|---|---|
| `-Q` | secondary | rows where a trailing question exists (= Step 2 Group A) | 47 | 3 | 0.915 | 0.496 | **−0.418** | 1.2e-05 | 4.9e-05 | 2.4e-05 |
| `-Q-D` | neg. control | trailing question and/or referral sentence exists | 53 | 3 | 0.925 | 0.503 | **−0.421** | 2.9e-06 | 1.2e-05 | 1.2e-05 |
| `-D` | neg. control | rows containing ≥1 clinician-referral sentence | 25 | **9** | 0.969 | 0.818 | −0.151 | 0.065 | 0.260 | 0.065 |
| `-H` | **PRIMARY** | rows with empathy framing or removable contrast marker | 38 | 3 | 0.956 | 0.851 | −0.105 | 0.042 | 0.166 | 0.055 |

`-Q`'s **n=47 equals Step 2 Group A** — the arm can only be built where the feature
exists, so its population *is* the group of zero-scoring rows ending in `?`.

Only `-Q` and `-Q-D` survive correction. **`-H`, the pre-registered primary arm,
and `-D` are both null.**

### Table 6 (secondary) — common subset, `results/table6_common_subset.csv`

The **only valid cross-arm comparison**: all arms on the same 14 rows.
**Underpowered — indicative, not confirmatory.**

| arm | control P(0) | ablated P(0) | delta | raw p |
|---|---|---|---|---|
| `-Q-D` | 0.952 | 0.452 | −0.500 | 0.0066 |
| `-Q` | 0.952 | 0.667 | −0.286 | 0.041 |
| `-D` | 0.952 | 0.810 | −0.143 | 0.221 |
| `-H` | 0.952 | 0.881 | −0.071 | 0.593 |

---

## Result 4 — `-D` resolved: deferral contributes nothing

Two independent tests, the second stronger.

**(a) Arm-level test at 9 passes** (`results/table5_within_arm.csv :: -D`, and
`results/step4_D_extended_per_row.csv`). Taking `-D` and its paired control rows
from 3 to 9 passes **changes the conclusion**:

| passes | n | control P(0) | `-D` P(0) | delta | raw p | verdict |
|---|---|---|---|---|---|---|
| 3 | 25 | 0.973 | 0.800 | −0.173 | 0.036 | marginally significant, failed Bonferroni |
| **9** | 25 | 0.969 | 0.818 | **−0.151** | **0.065** | **not significant at any threshold** |

The 3-pass result was a low-power artefact. With triple the passes the effect
shrinks and the p-value crosses out of significance.

**(b) `-Q-D` does not exceed `-Q`** (`results/table7_QD_vs_Q.csv`) — the primary
argument on D:

> paired on the 47 rows where both arms exist, `-Q` P(0) = **0.4965**,
> `-Q-D` P(0) = **0.4397**, delta **−0.057**, Wilcoxon **p = 0.111**.

Removing deferral *on top of* removing the trailing question adds nothing
detectable. Since `-Q-D`'s large effect is fully accounted for by its `-Q`
component, **D contributes nothing**. This is stronger than the arm-level test
because it is a within-row comparison of two arms that differ by exactly the D
operation, rather than an arm-vs-control test on a smaller subset.

---

## Rejected hypotheses (negative results, retained)

### H1 — clinical deferral. REJECTED on five lines.

1. `gpt-4o` classifier: **163 COMMITS, 0 DEFERS** (`deferral_labels.json`). The 2×2
   has an empty row; Fisher undefined. The classifier is sound — on synthetic inputs
   it emits DEFERS for pure referrals, bare empathy, and refusals. The corpus simply
   never substitutes referral for an answer.
2. **0 of 504** generated questions from zero rows are meta-questions (0.0%).
3. **31 of 56** zero rows contain no referral sentence at all — nothing to defer.
4. `referral_per_1k` fails multiple-testing correction: raw p 0.033, **BH q 0.063**
   (`step3_feature_separation.csv :: p, p_benjamini_hochberg`).
5. Causal test: `-D` null at 9 passes (p 0.065) and `-Q-D` ≯ `-Q` (p 0.111).

### H2 — hedging register. REJECTED by intervention despite strong correlation.

Correlationally the best signal in the study
(`step3_feature_separation.csv`, `population_positive == baseline_zero`):

| feature | `baseline_zero` | `baseline_nonzero` | AUC | raw p | BH q |
|---|---|---|---|---|---|
| `contrast_per_1k` | 1.059 | 0.327 | **0.764** | 1.1e-09 | 2.1e-08 |
| `empathy_per_1k` | 0.525 | 0.092 | 0.671 | 1.6e-07 | 1.5e-06 |
| `opens_with_empathy` | 0.393 | 0.075 | 0.659 | 7.1e-07 | 4.5e-06 |

All three survive **Bonferroni**. But the intervention refutes it:

- `-H` delta only **−0.105**, BH q 0.055, null on the common subset (p 0.593).
- **No dose-response**: `r = −0.014` between hedging removed and P(0) movement
  (`results/step4_hedging_dose_response.csv :: hedging_removed`, `delta_p_zero`,
  `pearson_r_hedging_removed_vs_delta_p_zero`). The ablation moved its targets hard
  (contrast/1k 1.189→0.265, empathy/1k 0.881→0.000, reduced in 27/38 and 25/38 rows)
  and the flag did not follow. **Feature moved, flag did not** — not a failed ablation.
- **Cosine confound**: on the common subset `-Q`, `-D`, `-Q-D` are all within ±0.005
  of control, but `-H` is at **−0.037** (`results/step4_cosine_by_arm.csv ::
  delta_vs_control_common_subset`). `-H` is the only arm that perturbed question
  generation, so it does not cleanly isolate the flag and part of its small effect
  may be a similarity artefact.

The strongest correlational signal in the study has the weakest causal support.

---

## The identified cause is PARTIAL

**After removing the trailing question, ~50% of rows are still flagged.**

`results/table5_within_arm.csv :: mean_p_zero_ablated` for `-Q` = **0.4965** —
that is, on the 47 rows where the ablation is possible, roughly half still score
exactly 0.0 in the majority of passes. The trailing conversational question is the
strongest identified contributor, **not a complete explanation**. Whatever drives
the remaining ~50% was not identified by this study, and the two candidates tested
(deferral, hedging register) were both refuted.

---

## Claims supported / not supported

### Supported

| claim | evidence |
|---|---|
| The zeros are a reproducible property of the text, not judge noise | 51/56 reproduce 3/3; 0/107 non-zeros do; agreement 96.9% (`probe_per_row.csv`) |
| The flag, not similarity, causes the zeros | zeros' mean cosine 0.9170 pre-multiplier (`probe_per_row.csv :: mean_cosine`) |
| Question generation is not corrupted | 0/504 meta-questions (`probe_raw.csv :: gen_q1..3`) |
| Removing the trailing question substantially reduces flagging | `-Q` delta −0.418, p 1.2e-05, survives Bonferroni (`table5_within_arm.csv`) |
| Deferral contributes nothing | `-D` p 0.065 at 9 passes; `-Q-D` ≯ `-Q` p 0.111 (`table5`, `table7`) |
| Hedging register does not cause the flag | `-H` BH q 0.055, common-subset p 0.593, dose-response r −0.014 |
| The zeros are not a quality signal | DeepEval scores the same 56 rows at mean 0.967 (`probe_per_row.csv :: deepeval`) |

### NOT supported

| claim | why not |
|---|---|
| "The trailing question explains the zeros" | ~50% remain flagged after `-Q` (`table5 :: mean_p_zero_ablated` = 0.4965) |
| "Ending in a question causes a zero" | not sufficient — 68 rows end in `?` and scored fine (Step 2, OR 3.00) |
| "Hedging register causes the flag" | strong correlation (AUC 0.764), refuted by intervention |
| "Clinical deferral causes the flag" | refuted on five independent lines |
| Any cross-arm ranking from Table 5 | different populations per arm; only Table 6 permits it, and it is underpowered (n=14) |
| A complete causal account of the 56 zeros | not established; ~50% unexplained |

---

## Audit trail

Corrections made during the investigation, recorded rather than silently fixed.

### 1. Population mislabelling — n=58 vs n=56

**Claimed:** a feature table reporting a "flagged" group of **n=58** against a
baseline with **56** zeros, with prose stating the split was made at the median of
`p_zero`.

**What changed:** both were wrong. The code used `p_zero > 0` — *reprobe flagged in
≥1 of 3 passes* — a different population from the baseline zeros. No median split
was ever performed. Reconciliation **56 − 2 + 4 = 58**:

| direction | row | baseline score | passes `all_noncommittal` |
|---|---|---|---|
| `baseline_zero` only | 17 | 0.0000 | 0/3 |
| `baseline_zero` only | 42 | 0.0000 | 0/3 |
| `reprobe_flagged_1plus_of3` only | 73 | 0.9076 | 1/3 |
| `reprobe_flagged_1plus_of3` only | 78 | 0.9231 | 1/3 |
| `reprobe_flagged_1plus_of3` only | 79 | 0.9043 | 2/3 |
| `reprobe_flagged_1plus_of3` only | 152 | 0.9374 | 1/3 |

**Why it matters:** the reported n did not match the population being described.
Fixed by renaming every column to an explicit population name (`baseline_zero`,
`baseline_nonzero`, `reprobe_flagged_1plus_of3`, `reprobe_flagged_3of3`,
`n_passes_all_noncommittal`, `reprobe_p_zero`) and reporting the feature table for
**both** populations. They agree: max AUC shift 0.037, mean 0.021, rank order
unchanged.

### 2. `referral_per_1k` reversal

**Claimed:** "referral density is not significant, p = 0.14, AUC 0.561" — used to
argue deferral was dead.

**What changed:** that was measured on `reprobe_flagged_1plus_of3` (n=58). On the
primary population `baseline_zero` (n=56) it is raw p **0.033**, AUC 0.588 —
significant at the uncorrected level. The original statement was true of the
population tested but **overstated for the primary population**.

**Where it lands:** BH-adjusted across 19 features, **q = 0.063** — fails
correction. The conclusion survives but now rests on the correction, not the raw
p-value. Any write-up must cite the `baseline_zero` figures with BH.

### 3. The Step 2 trailing-question rejection was premature

**Claimed:** after Step 2, "the trailing-question hypothesis is dead," on the
grounds that 68 rows end in `?` and scored fine (Fisher OR 3.00, p 0.0068). That
conclusion was carried forward and the arm was demoted to "secondary" while `-H`
was promoted to primary.

**What changed:** the ablation shows `-Q` is the **strongest causal lever measured**
(delta −0.418, p 1.2e-05, survives Bonferroni), while the hypotheses that replaced
it were both refuted.

**Why:** Step 2 was correlational and answered "is ending in a question *sufficient*
to cause a zero?" — correctly, no. The ablation answers "among rows that *are*
zeros, does removing the question help?" — yes, for ~42% of them. A cause can be
insufficient on its own and still be the dominant contributor within the affected
subgroup. Rejecting it on correlational grounds conflated *not sufficient* with
*not causal*; only the intervention distinguishes them.

---

## Files

`results/` is self-contained for external review. Full precision throughout.

| file | contents |
|---|---|
| `probe_raw.csv` | every row × pass: 3 generated questions, 3 flags, 3 cosines, score |
| `probe_per_row.csv` | per-row aggregate, explicit population columns |
| `step3_feature_separation.csv` | 19 features × 2 populations, raw + Bonferroni + BH |
| `table5_within_arm.csv` | **primary** arm results, population named per row |
| `table6_common_subset.csv` | **secondary**, the only valid cross-arm comparison |
| `table7_QD_vs_Q.csv` | does D add anything on top of Q |
| `step4_per_row.csv` | per row × arm: P(zero), cosine, post-ablation features |
| `step4_D_extended_per_row.csv` | `-D` and its control at 9 passes |
| `step4_hedging_dose_response.csv` | per-row hedging removed vs P(zero) movement |
| `step4_cosine_by_arm.csv` | cosine per arm, within-arm and common-subset |

Scripts: `classify_deferral.py`, `analyze_step2b.py`, `probe_noncommittal.py`,
`analyze_step3.py`, `build_ablations.py`, `score_ablations.py`, `extend_d_passes.py`,
`analyze_step4.py`, `make_results.py`.
Review artefacts: `step2b_review.txt`, `step3_generated_questions.txt`,
`step4_ablation_review.txt`, `step4_summary.md`.

---

## Appendix — TruLens groundedness (separate metric, separate pipeline)

Diagnostic summarisation over `evals/TruLens/trulens_scored_results.csv`, joined
to `flagged_records.json` for `retrieved_context`. Read-only; TruLens was not
re-run; zero API calls. **This is a different metric on a different pipeline from
everything above — do not read across.**

### Selection caveat

**n = 59.** Mean groundedness **0.284**, median 0.286, min 0.000, max **0.500**.
Distribution: 15 rows ≤0.2, 33 in (0.2, 0.4], 11 in (0.4, 0.6], **none above 0.6**.

These rows come from `flagged_records.json` — they were **selected on outcome**
(already flagged as problematic on ragas faithfulness / context recall). This is a
**diagnostic sample, not a prevalence estimate**. The ceiling of 0.500 exists
because high scorers were never eligible for inclusion. Do not quote "groundedness
is 0.28" as a property of the pipeline.

### What the metric is actually scoring

The reasoning column is one `Criteria:` / `Supporting Evidence:` block per
(claim × context chunk). Deduped: **490 distinct claims across 59 rows**, mean 8.3
per row, of which **226 (46.1%) returned `NOTHING FOUND`**.

### Failure types (derived from the text, then encoded as rules)

`results/table10_groundedness_failure_types.csv`

| failure type | claims | % | rows |
|---|---|---|---|
| `other_unclassified` | 39 | 17.3% | 21 |
| `exercise_modality_description` | 38 | 16.8% | 19 |
| `nutrition_enumeration` | 35 | 15.5% | 10 |
| `generic_wellness_advice` | 28 | 12.4% | 13 |
| `cancer_specific_claim` | 25 | 11.1% | 18 |
| `off_domain_topic` | 13 | 5.8% | 3 |
| `numeric_prescription` | 11 | 4.9% | 7 |
| `physiological_mechanism_claim` | 10 | 4.4% | 8 |
| `progression_pacing_guidance` | 8 | 3.5% | 7 |
| `list_scaffold_header` | 7 | 3.1% | 7 |
| `interrogative_intake_question` | 6 | 2.7% | 5 |
| `clinician_referral` | 3 | 1.3% | 2 |
| `named_authority_attribution` | 3 | 1.3% | 3 |

The lowest-scoring row (**0.000**) is scored entirely on **MI intake questions**:
*"To start, can you share what your current physical activity looks like?"* —
interrogatives that assert nothing and therefore cannot be grounded in anything.

`off_domain_topic` (13 claims, 3 rows) is content wholly outside the cancer-exercise
corpus — breastfeeding, pediatric nutrition, DNA-based nutrition. No retrieval
strategy over this corpus could ground those, because the corpus has no such
material.

### Mechanism

Rule: content-word overlap between claim and that row's `retrieved_context`.

| mechanism | claims | % |
|---|---|---|
| general knowledge added unprompted (**grounding problem**) | 144 | 63.7% |
| **could not classify** (overlap 0.30–0.60) | 47 | 20.8% |
| genuinely absent from context (**retrieval problem**) | 16 | 7.1% |
| no factual content, still counted as a claim | 13 | 5.8% |
| present but paraphrased (**judge strictness**) | 6 | 2.7% |

**Judge strictness is not the story — it is the smallest category at 2.7%.** The
dominant mode is the model volunteering generic health knowledge the corpus never
contained (hydration, sleep hygiene, food lists, exercise modality descriptions).
Retrieval failure proper is 7.1%.

**47 claims (20.8%) could not be classified** — intermediate overlap, where the rule
cannot distinguish loose paraphrase from coincidental vocabulary. The on-topic /
off-topic split is keyed on cancer vocabulary and is a heuristic, not an
adjudication.

### Source document and question type

**No source document dominates.** Per-`source_pdf` means range 0.105–0.500, but
26 of 30 non-null rows sit in single-digit-n buckets — `exercise_guidelines_for_cancer_survivors__23.pdf`
(n=8, mean 0.329) and `weight_training_and_risk_of_10_common_types_of_7.pdf` (n=5,
mean 0.337) are the only ones with meaningful n, and both sit near the overall mean.
`source_pdf` is null for 29/59 rows (the deepeval `rag_grounded` set).

**No question type dominates either**: Kruskal-Wallis across categories
**H = 2.990, p = 0.393**. Physical Activity (n=21) mean 0.308, `rag_grounded`
(n=29) mean 0.282.

Unsupported-claim *rate* varies more than the score does — Cancer & Survivorship
Education 0.553 and Diet 0.533 versus SMART Goal 0.000 and Blood Pressure 0.000 —
but those low-rate categories have 4–9 claims total, far too few to support a claim
of difference.

### Reading

`step6_groundedness_report.txt` — full verbatim reasoning for the 10
lowest-scoring rows, with bot answer, question, category and source_pdf.
Per-claim data: `results/step6_groundedness_claims.csv`.

---

## Recommendation

Treat `ragas_answer_relevancy == 0.0` as **non-informative for this corpus**. The
flag fires reproducibly on substantively correct answers whose questions generate at
cosine ~0.92, and DeepEval scores the same rows at mean 0.967. Report the DeepEval
column or the pre-multiplier cosine alongside it. The strongest identified
contributor is the trailing conversational question, and it accounts for only about
half the flagged rows.
