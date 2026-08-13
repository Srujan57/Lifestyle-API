"""
Deterministic unit tests for self-reported diabetes detection.

DELIBERATELY SEPARATE FROM test_le8_scoring.py. That file tests the scoring
maths (given has_diabetes, what score?). This file tests the classifier that
decides has_diabetes in the first place — a text-parsing problem with an
entirely different failure mode. The bug that prompted these tests was not a
wrong threshold; it was "Does that mean I have diabetes?" being read as a
diagnosis because the positive pattern is a bare substring match.

Why that matters beyond a mislabel: the flag selects which HbA1c scale
_score_hba1c uses, and the two disagree at every value. A user asking whether
6.0% means diabetes was scored 40 (diabetic scale) instead of 60 — and
_build_computed_value_note injects that number into the prompt as authoritative
pre-computed truth, specifically so the model won't recalculate it. The misread
also persists, because _detect_diabetes_status scans the whole history.

Functions/constants under test:
    _detect_diabetes_status      app.py  (message list -> True/False/None)
    _DIABETES_INTERROGATIVE_RE   app.py  (veto: asking is not reporting)
    _score_hba1c                 app.py  (downstream consumer of the flag)

Plain pytest — no LLM judge, no API calls. See README.md.
"""

import pytest

from le8_loader import load_functions

_fns = load_functions(
    "_detect_diabetes_status",
    "_score_hba1c",
    "_DIABETES_POSITIVE_RE",
    "_DIABETES_NEGATIVE_RE",
    # _DIABETES_INTERROGATIVE_RE is built from _DIABETES_CLAIM, so the shared
    # alternation has to be loaded too — the loader emits in source order, so
    # it lands before the regex that interpolates it.
    "_DIABETES_CLAIM",
    "_DIABETES_INTERROGATIVE_RE",
)
_detect_diabetes_status = _fns["_detect_diabetes_status"]
_score_hba1c = _fns["_score_hba1c"]


# ===========================================================================
# Questions and conditionals are NOT self-reports.
#
# Every case here returned True before the _DIABETES_INTERROGATIVE_RE veto.
# Rows 5 and 37 of the Manual Questions bank are marked; they are real
# question-bank entries, not invented probes.
# ===========================================================================
@pytest.mark.parametrize("text", [
    "Does that mean I have diabetes?",
    "My HbA1c came back at 6.0%. Does that mean I have diabetes?",   # bank row 37
    "Can you tell me if my HbA1c means I have diabetes or not?",      # bank row 5
    "Do I have diabetes?",
    "Did I have diabetes when I was diagnosed?",
    "Would it change things if I have diabetes?",
    "How would I know if I have diabetes?",
    "If I have diabetes, does that change my score?",
    "Whether I have diabetes or not, should I still exercise?",
    "If I'm diabetic, do I need a different plan?",
])
def test_interrogative_is_not_a_diagnosis(text):
    assert _detect_diabetes_status([text]) is None


# ===========================================================================
# Genuine self-reports must still register. The veto is allowed to cost
# nothing here — a fix that silences real diagnoses is worse than the bug.
# ===========================================================================
@pytest.mark.parametrize("text", [
    "I have diabetes.",
    "I'm diabetic.",
    "I am a diabetic.",
    "I was diagnosed with diabetes last year.",
    "My diabetes is well controlled.",
])
def test_self_report_still_detected(text):
    assert _detect_diabetes_status([text]) is True


@pytest.mark.parametrize("text", [
    "I don't have diabetes.",
    "I do not have diabetes.",
    "I'm not diabetic.",
    "No diabetes here.",
])
def test_explicit_negation_still_detected(text):
    assert _detect_diabetes_status([text]) is False


# ===========================================================================
# Clause scoping. The veto keys on the claim's OWN clause, so a sentence or
# clause boundary between the claim and the question mark must not veto it.
# ===========================================================================
def test_statement_then_separate_question_is_still_a_diagnosis():
    # Period between claim and "?" — bank rows 38/39 are shaped like this.
    assert _detect_diabetes_status(
        ["I have diabetes. How does that change my blood sugar score?"]) is True


def test_statement_then_comma_question_is_still_a_diagnosis():
    # Comma is treated as a clause boundary too, so this real self-report
    # survives even though a "?" follows it with no period in between.
    assert _detect_diabetes_status(
        ["I have diabetes, does that change my score?"]) is True


def test_semicolon_boundary_is_still_a_diagnosis():
    assert _detect_diabetes_status(
        ["I have diabetes; what should I aim for?"]) is True


# ===========================================================================
# History scanning. The veto SKIPS a message rather than breaking the scan,
# so a question must not erase a diagnosis stated earlier — and must not
# resurrect one that was later retracted.
# ===========================================================================
def test_question_does_not_erase_earlier_diagnosis():
    assert _detect_diabetes_status([
        "I have diabetes.",
        "Does that mean I have diabetes for life?",
    ]) is True


def test_question_does_not_create_a_diagnosis_after_negation():
    assert _detect_diabetes_status([
        "I don't have diabetes.",
        "Does that mean I have diabetes?",
    ]) is False


def test_most_recent_explicit_statement_still_wins():
    assert _detect_diabetes_status([
        "I have diabetes.",
        "Sorry, I misspoke — I don't have diabetes.",
    ]) is False


def test_never_mentioned_is_none():
    assert _detect_diabetes_status(["What should my step goal be?"]) is None
    assert _detect_diabetes_status([]) is None


# ===========================================================================
# KNOWN LIMITATION — documented, deliberately NOT asserted as passing.
#
# Epistemic hedges still register as positive: "I'm worried I have diabetes",
# "I think I have diabetes", "I might have diabetes". There is no
# interrogative head word and no question mark to key on, so surface syntax
# cannot separate worry from diagnosis. Erring toward "treat it as reported"
# matches the file's existing bias for this population.
#
# This test pins the CURRENT behaviour so a future change is a deliberate
# decision rather than an accident. If someone adds hedge handling, this test
# fails loudly and should be rewritten to the new expectation — same approach
# the difficulty suite takes with its acknowledged veto gap.
# ===========================================================================
@pytest.mark.parametrize("text", [
    "I'm worried I have diabetes.",
    "I think I have diabetes.",
])
def test_known_limitation_hedges_read_as_positive(text):
    assert _detect_diabetes_status([text]) is True, (
        "Hedge handling may have been added — if intentional, update this "
        "documented-limitation test to the new expectation."
    )


# ===========================================================================
# Downstream: the flag changes the reported score, which is why the misread
# mattered. Row 37's value is called out explicitly.
# ===========================================================================
@pytest.mark.parametrize("hba1c", [5.4, 5.7, 6.0, 6.5, 7.0])
def test_diabetic_flag_changes_hba1c_score(hba1c):
    assert _score_hba1c(hba1c, False) != _score_hba1c(hba1c, True)


def test_bank_row37_scores_on_the_non_diabetic_scale():
    """HbA1c 6.0% asked as a question scores 60, not the diabetic-scale 40."""
    status = _detect_diabetes_status(
        ["My HbA1c came back at 6.0%. Does that mean I have diabetes?"])
    assert status is None
    assert _score_hba1c(6.0, bool(status)) == 60
    assert _score_hba1c(6.0, True) == 40    # what the bug produced
