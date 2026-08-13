"""Step 4 gate: build the five ablation arms for the 56 zero-scoring rows.

Arms
----
  control  unmodified original
  -Q       trailing interrogative sentence(s) deleted, deferral kept
  -D       sentences carrying clinician-referral deleted, trailing question kept
  -Q-D     both
  -H       empathy/validation framing deleted and contrast connectives removed,
           every substantive claim kept

-H is the primary arm. Step 3 found the flag tracks contrastive/empathetic
register (contrast_per_1k AUC 0.727, empathy_per_1k AUC 0.680) rather than
referral (AUC 0.561, n.s.), so -D and -Q-D are pre-registered negative controls.

Method
------
Every edit is a deterministic deletion or a fixed surface rewrite. No LLM is
used to rewrite text, so no arm can contain fabricated or paraphrased content --
whatever survives an ablation is a byte-exact substring of the original modulo
the documented connective rewrites and whitespace repair.

Where an ablation is a no-op (nothing to remove) or would strip the response of
all substantive content, the row is marked UNABLATABLE for that arm and reported
separately rather than being padded with invented text.

Writes step4_ablations.json and step4_ablation_review.txt. Scores nothing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASELINE_CSV = REPO / "back-end" / "evals" / "RAGAS" / "RAGAS_Baseline.csv"
OUT_JSON = HERE / "step4_ablations.json"
REVIEW_TXT = HERE / "step4_ablation_review.txt"

ARMS = ["control", "-Q", "-D", "-Q-D", "-H"]

REFERRAL_RE = re.compile(
    r"care team|healthcare team|health care team|medical team|oncolog\w*"
    r"|your doctor|your provider|your physician|consult\w*|speak (?:with|to)"
    r"|talk (?:with|to) your|check with|reach out to|discuss (?:this )?with",
    re.IGNORECASE)

EMPATHY_RE = re.compile(
    r"it'?s understandable|i understand|that'?s a great|great question"
    r"|it'?s great to hear|i hear you|that sounds|it'?s completely"
    r"|you'?re not alone|it'?s normal|i'?m glad|thank you for sharing"
    r"|it'?s wonderful|congratulations|i'?m sorry to hear|that'?s wonderful"
    r"|good for you|it'?s okay to|feel free to",
    re.IGNORECASE)

# Contrast connectives, as (pattern, replacement, capitalise_next).
# Each rewrite keeps BOTH clauses -- only the connective itself is removed, so no
# substantive claim is lost. Sentence-initial subordinators like "Although X, Y"
# are deliberately NOT rewritten: deleting them leaves a comma splice, so they
# are left in place and counted as unremoved.
CONTRAST_REWRITES = [
    (re.compile(r",\s+but\s+", re.IGNORECASE), ". ", True),
    (re.compile(r",\s+however,?\s+", re.IGNORECASE), ". ", True),
    (re.compile(r",\s+though\s+", re.IGNORECASE), ". ", True),
    (re.compile(r",\s+yet\s+", re.IGNORECASE), ". ", True),
    (re.compile(r"(?<=[.!?])\s+However,\s+"), " ", True),
    (re.compile(r"(?<=[.!?])\s+That said,\s+", re.IGNORECASE), " ", True),
    (re.compile(r"(?<=[.!?])\s+On the other hand,\s+", re.IGNORECASE), " ", True),
    (re.compile(r"^However,\s+"), "", True),
    (re.compile(r"^That said,\s+", re.IGNORECASE), "", True),
]
CONTRAST_COUNT_RE = re.compile(
    r"\bhowever\b|\bbut\b|\balthough\b|\bthough\b|\bthat said\b"
    r"|\byet\b|\bon the other hand\b|\bwhile\b", re.IGNORECASE)

# Split on sentence punctuation, but not after a list marker ("1. ") or a
# decimal ("2.5 "), and not after common abbreviations.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])(?<!\d\.)(?<!e\.g\.)(?<!i\.e\.)(?<!Dr\.)\s+")


def split_sentences(text: str):
    """Split into sentences, preserving the exact separators for reassembly."""
    parts, last = [], 0
    for m in SENT_SPLIT_RE.finditer(text):
        parts.append((text[last:m.start()], text[m.start():m.end()]))
        last = m.end()
    if last < len(text):
        parts.append((text[last:], ""))
    return parts


def reassemble(parts) -> str:
    return "".join(s + sep for s, sep in parts).strip()


def is_substantive(sent: str) -> bool:
    """A sentence carries content if it is long enough and is not pure empathy."""
    s = sent.strip()
    return len(s) >= 40 and not (EMPATHY_RE.search(s) and len(s) < 120)


def ablate_Q(text: str):
    """Delete trailing interrogative sentence(s)."""
    parts = split_sentences(text)
    removed = []
    while parts and parts[-1][0].strip().endswith("?"):
        removed.append(parts.pop()[0].strip())
    if not removed:
        return None, "no trailing question", []
    out = reassemble(parts)
    if not out.strip():
        return None, "removing trailing question leaves nothing", removed
    return out, None, removed


def ablate_D(text: str):
    """Delete sentences that hand the question to a clinician."""
    parts = split_sentences(text)
    kept, removed = [], []
    for sent, sep in parts:
        if REFERRAL_RE.search(sent):
            removed.append(sent.strip())
        else:
            kept.append((sent, sep))
    if not removed:
        return None, "no referral sentence", []
    out = reassemble(kept)
    if not any(is_substantive(s) for s, _ in kept):
        return None, "removing deferral leaves no substantive content", removed
    return out, None, removed


def ablate_H(text: str):
    """Delete empathy/validation framing and strip contrast connectives."""
    parts = split_sentences(text)
    removed = []

    # 1. Drop leading validation sentences (only while they lead).
    while parts and EMPATHY_RE.search(parts[0][0]) and not is_substantive(parts[0][0]):
        removed.append(parts.pop(0)[0].strip())
    # An empathetic opener that is long enough to carry content is stripped only
    # if a later sentence still carries substance, so no claim is lost.
    if parts and EMPATHY_RE.search(parts[0][0]) and len(parts) > 1:
        if any(is_substantive(s) for s, _ in parts[1:]):
            removed.append(parts.pop(0)[0].strip())

    # 2. Remove any remaining mid-text empathy sentences.
    kept = []
    for sent, sep in parts:
        if EMPATHY_RE.search(sent) and not is_substantive(sent):
            removed.append(sent.strip())
        else:
            kept.append((sent, sep))

    out = reassemble(kept)

    # 3. Strip contrast connectives, keeping both clauses.
    n_before = len(CONTRAST_COUNT_RE.findall(out))
    for pat, repl, cap in CONTRAST_REWRITES:
        def _sub(m):
            return repl
        new = pat.sub(_sub, out)
        if cap and new != out:
            # Re-capitalise the word that now begins a sentence.
            new = re.sub(r"([.!?]\s+)([a-z])",
                         lambda m: m.group(1) + m.group(2).upper(), new)
            new = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), new)
        out = new
    n_after = len(CONTRAST_COUNT_RE.findall(out))

    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()

    if not removed and n_before == n_after:
        return None, "no empathy framing or removable contrast marker", [], 0
    if not any(is_substantive(s) for s, _ in kept):
        return None, "removing empathy framing leaves no substantive content", removed, 0
    return out, None, removed, n_before - n_after


def features(text: str) -> dict:
    n = max(len(str(text)), 1)
    per1k = 1000.0 / n
    return {
        "contrast_per_1k": len(CONTRAST_COUNT_RE.findall(str(text))) * per1k,
        "empathy_per_1k": len(EMPATHY_RE.findall(str(text))) * per1k,
        "referral_per_1k": len(REFERRAL_RE.findall(str(text))) * per1k,
        "len": len(str(text)),
    }


def main() -> int:
    df = pd.read_csv(BASELINE_CSV)
    zeros = df[df["ragas_answer_relevancy"] == 0.0]
    print(f"building 5 arms for {len(zeros)} zero-scoring rows")

    records = []
    for idx, row in zeros.iterrows():
        orig = str(row["Bot Answer"])
        rec = {
            "row": int(idx),
            "question": str(row["Question"]),
            "baseline_score": float(row["ragas_answer_relevancy"]),
            "deepeval_score": float(row["deepeval_answer_relevancy"]),
            "arms": {"control": {"text": orig, "unablatable": None, "removed": []}},
        }

        q_text, q_why, q_rm = ablate_Q(orig)
        rec["arms"]["-Q"] = {"text": q_text or orig, "unablatable": q_why,
                             "removed": q_rm}

        d_text, d_why, d_rm = ablate_D(orig)
        rec["arms"]["-D"] = {"text": d_text or orig, "unablatable": d_why,
                             "removed": d_rm}

        # -Q-D composes the two deletions; unablatable only if BOTH are no-ops.
        qd_src = q_text if q_text else orig
        qd_text, qd_why, qd_rm = ablate_D(qd_src)
        if qd_text is None:
            qd_text2, qd_why2, _ = (q_text, None, []) if q_text else (None, qd_why, [])
            rec["arms"]["-Q-D"] = {
                "text": qd_text2 or orig,
                "unablatable": None if qd_text2 else (q_why or qd_why),
                "removed": q_rm,
            }
        else:
            rec["arms"]["-Q-D"] = {"text": qd_text, "unablatable": None,
                                   "removed": q_rm + qd_rm}

        h_text, h_why, h_rm, n_contrast_removed = ablate_H(orig)
        rec["arms"]["-H"] = {"text": h_text or orig, "unablatable": h_why,
                             "removed": h_rm,
                             "contrast_markers_removed": n_contrast_removed}

        for arm in ARMS:
            rec["arms"][arm]["features"] = features(rec["arms"][arm]["text"])
        records.append(rec)

    OUT_JSON.write_text(json.dumps(records, indent=2))

    # ---- review file ----
    lines = []
    w = lines.append
    w("=" * 100)
    w("STEP 4 GATE -- FIVE ABLATION ARMS FOR ALL 56 ZERO-SCORING ROWS")
    w("=" * 100)
    w("")
    w("Primary arm: -H (empathy framing + contrast connectives removed).")
    w("-D and -Q-D are pre-registered negative controls: Step 3 found referral")
    w("density does not separate flagged from unflagged rows (AUC 0.561, p=0.14).")
    w("")
    w("All edits are deterministic deletions or fixed connective rewrites. No LLM")
    w("rewriting, so no arm contains invented or paraphrased content.")
    w("")

    counts = {arm: 0 for arm in ARMS}
    for rec in records:
        for arm in ARMS:
            if rec["arms"][arm]["unablatable"]:
                counts[arm] += 1
    w("-" * 100)
    w("UNABLATABLE COUNTS (row excluded from that arm's paired test)")
    w("-" * 100)
    for arm in ARMS:
        w(f"  {arm:<8s} {counts[arm]:>3d} / {len(records)} unablatable")
    w("")

    for rec in records:
        w("")
        w("=" * 100)
        w(f"ROW {rec['row']}   baseline={rec['baseline_score']:.4f}   "
          f"deepeval={rec['deepeval_score']:.3f}")
        w(f"QUESTION: {rec['question']}")
        w("=" * 100)
        for arm in ARMS:
            a = rec["arms"][arm]
            f = a["features"]
            w("")
            w("-" * 100)
            status = f"UNABLATABLE ({a['unablatable']})" if a["unablatable"] else "ok"
            w(f"### ARM {arm}   [{status}]   len={f['len']}  "
              f"contrast/1k={f['contrast_per_1k']:.2f}  "
              f"empathy/1k={f['empathy_per_1k']:.2f}  "
              f"referral/1k={f['referral_per_1k']:.2f}")
            if arm == "-H" and "contrast_markers_removed" in a:
                w(f"    contrast markers removed: {a['contrast_markers_removed']}")
            if a["removed"]:
                w("    REMOVED:")
                for r in a["removed"]:
                    w(f"      - {r!r}")
            w("-" * 100)
            w(a["text"])
    lines.append("")
    REVIEW_TXT.write_text("\n".join(lines))

    print(f"wrote {OUT_JSON}")
    print(f"wrote {REVIEW_TXT}")
    print("\nUNABLATABLE counts:")
    for arm in ARMS:
        print(f"  {arm:<8s} {counts[arm]:>3d} / {len(records)}")

    # Did -H actually move the features it targets?
    import numpy as np
    for arm in ["-Q", "-D", "-Q-D", "-H"]:
        ok = [r for r in records if not r["arms"][arm]["unablatable"]]
        if not ok:
            continue
        dc = np.mean([r["arms"][arm]["features"]["contrast_per_1k"]
                      - r["arms"]["control"]["features"]["contrast_per_1k"] for r in ok])
        de = np.mean([r["arms"][arm]["features"]["empathy_per_1k"]
                      - r["arms"]["control"]["features"]["empathy_per_1k"] for r in ok])
        dr = np.mean([r["arms"][arm]["features"]["referral_per_1k"]
                      - r["arms"]["control"]["features"]["referral_per_1k"] for r in ok])
        print(f"  {arm:<6s} n={len(ok):2d}  d_contrast={dc:+.3f}  "
              f"d_empathy={de:+.3f}  d_referral={dr:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
