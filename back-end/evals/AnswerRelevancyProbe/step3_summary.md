# Step 3 — instrumenting the ragas `answer_relevancy` noncommittal flag

Judge reconstructed exactly as the baseline built it: `gpt-4o-mini`, `bypass_n=True`, `OpenAIEmbeddings()` (ada-002), `strictness=3`, temperature left as `None` so the wrapper applies `get_temperature(3)=0.3`.

- rows probed: **163**
- passes per row: **3**
- judge calls: **1467**

## 1. Flag stability — the noise floor

Passes (of 3) in which all 3 generated questions were flagged noncommittal, i.e. the row scored exactly 0.0:

| passes with all_noncommittal | `baseline_zero` (n=56) | `baseline_nonzero` (n=107) |
|---|---|---|
| 0/3 | 2 (3.6%) | 103 (96.3%) |
| 1/3 | 2 (3.6%) | 3 (2.8%) |
| 2/3 | 1 (1.8%) | 1 (0.9%) |
| 3/3 | 51 (91.1%) | 0 (0.0%) |

(The non-zero column is the complete set of 107, not a sample.)

- baseline zeros reproducing in **all 3** passes: **51/56** (91.1%)
- baseline zeros that **never** reproduced: **2/56** (3.6%)
- baseline non-zeros that flipped to 0.0 at least once: **4/107** (3.7%)

### Baseline-vs-reprobe agreement on landing at 0.0

Majority of passes scoring 0.0 vs the stored baseline:

| | reprobe majority 0.0 | reprobe majority > 0 |
|---|---|---|
| baseline 0.0 | 52 | 4 |
| baseline > 0 | 1 | 106 |

Agreement: **96.9%**

## 2. Cosine similarity with the flag multiplier removed

`mean_cosine` is the similarity between the user's real question and the 3 questions the judge generated from the response — the score ragas would have produced *before* multiplying by `int(not all_noncommittal)`.

| group | n | mean cosine | median | min | max |
|---|---|---|---|---|---|
| baseline zeros | 56 | **0.9170** | 0.9236 | 0.7876 | 1.0000 |
| baseline non-zeros | 107 | **0.9379** | 0.9414 | 0.7965 | 0.9942 |

Mann-Whitney U: p = 0.000517, AUC = 0.334

Difference in mean cosine: **+0.0210** (non-zeros minus zeros).

### Are the generated questions meta-questions about consulting a clinician?

| group | generated questions | meta-questions | rate |
|---|---|---|---|
| from baseline zeros | 504 | 0 | 0.0% |
| from baseline non-zeros | 963 | 9 | 0.9% |

Cosine of each generated question, split by that question's own flag:

| flag | n | mean cosine |
|---|---|---|
| 0 | 931 | 0.9396 |
| 1 | 536 | 0.9153 |

## 3. What separates the flagged rows from the rest

### Population definitions

Three groupings are in play and must not be conflated:

| name | definition | n |
|---|---|---|
| `baseline_zero` | stored `ragas_answer_relevancy` == 0.0 | 56 |
| `baseline_nonzero` | stored score > 0.0 | 107 |
| `reprobe_flagged_1plus_of3` | `all_noncommittal` in >= 1 of 3 passes | 58 |
| `reprobe_flagged_3of3` | `all_noncommittal` in all 3 passes | 51 |

`baseline_zero` and `reprobe_flagged_1plus_of3` differ by 2 + 4 rows:

| direction | row | baseline score | passes all_noncommittal |
|---|---|---|---|
| in `baseline_zero` only | 17 | 0.0000 | 0/3 |
| in `baseline_zero` only | 42 | 0.0000 | 0/3 |
| in `reprobe_flagged_1plus_of3` only | 73 | 0.9076 | 1/3 |
| in `reprobe_flagged_1plus_of3` only | 78 | 0.9231 | 1/3 |
| in `reprobe_flagged_1plus_of3` only | 79 | 0.9043 | 2/3 |
| in `reprobe_flagged_1plus_of3` only | 152 | 0.9374 | 1/3 |

Arithmetic: 56 - 2 + 4 = 58.

Features are ranked by separation strength, not by a preset hypothesis.

### 3a. PRIMARY — `baseline_zero` (n=56) vs `baseline_nonzero` (n=107)

| feature | `baseline_zero` mean (n=56) | `baseline_nonzero` mean (n=107) | AUC | p (raw) | p (BH) | p (Bonf) |
|---|---|---|---|---|---|---|
| `contrast_per_1k` ** | 1.059 | 0.327 | 0.764 | 1.111e-09 | 2.111e-08 | 2.111e-08 |
| `empathy_per_1k` ** | 0.525 | 0.092 | 0.671 | 1.599e-07 | 1.519e-06 | 3.037e-06 |
| `opens_with_empathy` ** | 0.393 | 0.075 | 0.659 | 7.092e-07 | 4.491e-06 | 1.347e-05 |
| `conditional_per_1k` * | 2.014 | 1.350 | 0.646 | 0.002205 | 0.01047 | 0.0419 |
| `n_digits_per_1k` * | 2.291 | 3.477 | 0.386 | 0.0129 | 0.03064 | 0.2452 |
| `has_bold_headers` * | 1.375 | 2.505 | 0.387 | 0.009091 | 0.0277 | 0.1727 |
| `n_questions_in_body` * | 0.946 | 0.738 | 0.607 | 0.00476 | 0.01809 | 0.09045 |
| `has_numbered_list` * | 0.268 | 0.477 | 0.396 | 0.01021 | 0.0277 | 0.1939 |
| `mean_sentence_len` | 106.497 | 98.152 | 0.597 | 0.04233 | 0.07311 | 0.8042 |
| `referral_per_1k` | 1.128 | 0.627 | 0.588 | 0.03302 | 0.06275 | 0.6275 |
| `second_person_per_1k` | 8.904 | 7.952 | 0.586 | 0.07331 | 0.1161 | 1 |
| `importance_frame_per_1k` | 0.514 | 0.305 | 0.584 | 0.02452 | 0.05176 | 0.4659 |
| `hedge_per_1k` | 6.608 | 5.887 | 0.583 | 0.08305 | 0.1214 | 1 |
| `modal_ratio` | 0.042 | 0.038 | 0.570 | 0.1441 | 0.1912 | 1 |
| `len_chars` | 926.375 | 1029.439 | 0.431 | 0.1509 | 0.1912 | 1 |
| `n_sentences` | 9.429 | 11.449 | 0.433 | 0.1617 | 0.192 | 1 |
| `n_exclamations` | 0.571 | 0.701 | 0.454 | 0.2904 | 0.3245 | 1 |
| `first_sent_len` | 102.071 | 93.206 | 0.535 | 0.4587 | 0.4842 | 1 |
| `personalization_per_1k` | 0.618 | 0.655 | 0.493 | 0.8687 | 0.8687 | 1 |

AUC 0.5 = no separation. Stars reflect **BH-adjusted** p across 19 features: `**` q<0.01, `*` q<0.05.

### 3b. SECONDARY — `reprobe_flagged_1plus_of3` (n=58) vs `reprobe_unflagged_0of3` (n=105)

This is the grouping the earlier version of this table used while labelling it only 'flagged'. Retained for comparison.

| feature | `reprobe_flagged_1plus_of3` mean (n=58) | `reprobe_unflagged_0of3` mean (n=105) | AUC | p (raw) | p (BH) | p (Bonf) |
|---|---|---|---|---|---|---|
| `contrast_per_1k` ** | 0.994 | 0.349 | 0.727 | 1.177e-07 | 1.118e-06 | 2.236e-06 |
| `empathy_per_1k` ** | 0.539 | 0.076 | 0.680 | 3.022e-08 | 5.742e-07 | 5.742e-07 |
| `opens_with_empathy` ** | 0.397 | 0.067 | 0.665 | 2.163e-07 | 1.37e-06 | 4.11e-06 |
| `has_bold_headers` ** | 1.155 | 2.648 | 0.350 | 0.0004643 | 0.001894 | 0.008822 |
| `n_digits_per_1k` ** | 1.953 | 3.686 | 0.353 | 0.001223 | 0.003575 | 0.02323 |
| `has_numbered_list` ** | 0.224 | 0.505 | 0.360 | 0.0004984 | 0.001894 | 0.00947 |
| `conditional_per_1k` ** | 1.979 | 1.358 | 0.637 | 0.003704 | 0.008797 | 0.07038 |
| `mean_sentence_len` * | 107.965 | 97.183 | 0.630 | 0.006142 | 0.01167 | 0.1167 |
| `n_questions_in_body` ** | 0.966 | 0.724 | 0.620 | 0.001317 | 0.003575 | 0.02502 |
| `second_person_per_1k` * | 9.061 | 7.848 | 0.609 | 0.02097 | 0.03622 | 0.3984 |
| `len_chars` * | 903.621 | 1043.971 | 0.395 | 0.02641 | 0.04068 | 0.5018 |
| `n_sentences` * | 8.983 | 11.733 | 0.396 | 0.02783 | 0.04068 | 0.5288 |
| `importance_frame_per_1k` * | 0.547 | 0.283 | 0.604 | 0.005207 | 0.01099 | 0.09893 |
| `hedge_per_1k` * | 6.693 | 5.826 | 0.602 | 0.03081 | 0.04182 | 0.5854 |
| `modal_ratio` | 0.042 | 0.037 | 0.587 | 0.06774 | 0.0858 | 1 |
| `referral_per_1k` | 1.039 | 0.667 | 0.561 | 0.1375 | 0.1633 | 1 |
| `n_exclamations` | 0.569 | 0.705 | 0.452 | 0.2688 | 0.3004 | 1 |
| `first_sent_len` | 102.103 | 93.019 | 0.532 | 0.4957 | 0.5232 | 1 |
| `personalization_per_1k` | 0.596 | 0.667 | 0.483 | 0.6796 | 0.6796 | 1 |

AUC 0.5 = no separation. Stars reflect **BH-adjusted** p across 19 features: `**` q<0.01, `*` q<0.05.

### 3c. Do the AUCs change materially between the two populations?

| feature | AUC (`baseline_zero`) | AUC (`reprobe_flagged_1plus_of3`) | delta |
|---|---|---|---|
| `has_bold_headers` | 0.387 | 0.350 | -0.037 |
| `n_sentences` | 0.433 | 0.396 | -0.037 |
| `len_chars` | 0.431 | 0.395 | -0.037 |
| `contrast_per_1k` | 0.764 | 0.727 | -0.036 |
| `has_numbered_list` | 0.396 | 0.360 | -0.036 |
| `n_digits_per_1k` | 0.386 | 0.353 | -0.033 |
| `mean_sentence_len` | 0.597 | 0.630 | +0.033 |
| `referral_per_1k` | 0.588 | 0.561 | -0.027 |
| `second_person_per_1k` | 0.586 | 0.609 | +0.024 |
| `hedge_per_1k` | 0.583 | 0.602 | +0.020 |
| `importance_frame_per_1k` | 0.584 | 0.604 | +0.020 |
| `modal_ratio` | 0.570 | 0.587 | +0.017 |
| `n_questions_in_body` | 0.607 | 0.620 | +0.014 |
| `personalization_per_1k` | 0.493 | 0.483 | -0.010 |
| `conditional_per_1k` | 0.646 | 0.637 | -0.009 |
| `empathy_per_1k` | 0.671 | 0.680 | +0.008 |
| `opens_with_empathy` | 0.659 | 0.665 | +0.006 |
| `first_sent_len` | 0.535 | 0.532 | -0.003 |
| `n_exclamations` | 0.454 | 0.452 | -0.002 |

Largest absolute AUC shift: **0.037**; mean absolute shift **0.021**. Rank order of the top discriminators is unchanged.

### Noncommittal flag rate by deferral label

- **COMMITS** (n=163): mean per-question flag rate 0.365, mean P(zero) 0.331

The deferral variable is constant across the corpus (all COMMITS), so this is reported for completeness only — it cannot discriminate.
