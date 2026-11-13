"""Local scorer for Part 1, following the Kaggle evaluation description.

Per question (max 1.0):
  0.40 answer   share of required facts found in the answer (numbers within 1%; text facts: case-insensitive match)
  0.20 status   exact match
  0.20 figures  F1 between predicted and expected figure/table labels (NONE vs NONE = 1)
  0.20 cites    half: F1 of cited documents (DOC1/DOC2/DOC3); half: F1 of exact citation strings

The Kaggle description does not spell out every detail; choices made here: a number matches if
|predicted - expected| <= 1% of |expected| (sign ignored, so "a deficit of 1.7%" matches -1.7), and F1 on empty
vs non-empty is 0.

Usage:
  python eval/score_p1.py --questions eval/golden_set.csv eval/golden_questions.csv   # write the questions file
  python eval/score_p1.py submission.csv eval/golden_set.csv                           # score a run
"""
import re
import sys

import pandas as pd

NUM = re.compile(r"[-−]?\d+(?:\.\d+)?")


def split(cell):
    cell = str(cell).strip()
    return set() if cell in ("", "NONE") else {x.strip() for x in cell.split(",") if x.strip()}


def f1(pred, gold):
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def fact_found(fact, answer):
    fact = fact.strip()
    try:
        target = abs(float(fact.replace("−", "-")))
    except ValueError:
        return fact.lower() in answer.lower()
    nums = [abs(float(n.replace("−", "-"))) for n in NUM.findall(answer)]
    return any(abs(n - target) <= (0.01 * target if target else 1e-9) for n in nums)


def score_row(pred, gold):
    facts = [f for f in str(gold["required_facts"]).split(";") if f.strip()]
    answer = sum(fact_found(f, str(pred["answer"])) for f in facts) / len(facts) if facts else 1.0
    status = float(pred["status"].strip() == gold["expected_status"].strip())
    figs = f1(split(pred["figure_references"]), split(gold["figure_references"]))
    pc, gc = split(pred["citations"]), split(gold["citations"])
    docs = f1({c.split(":")[0] for c in pc}, {c.split(":")[0] for c in gc})
    cites = 0.5 * docs + 0.5 * f1(pc, gc)
    total = 0.4 * answer + 0.2 * status + 0.2 * figs + 0.2 * cites
    return dict(answer=answer, status=status, figures=figs, citations=cites, total=total)


def main(argv):
    if argv[0] == "--questions":
        g = pd.read_csv(argv[1], dtype=str, keep_default_na=False)
        g[["id", "conversation_id", "turn", "question_id", "question"]].to_csv(argv[2], index=False)
        print("wrote", len(g), "questions to", argv[2])
        return
    sub = pd.read_csv(argv[0], dtype=str, keep_default_na=False).set_index("question_id")
    gold = pd.read_csv(argv[1], dtype=str, keep_default_na=False)
    rows = []
    for _, gq in gold.iterrows():
        pred = sub.loc[gq["question_id"]] if gq["question_id"] in sub.index else None
        s = score_row(pred, gq) if pred is not None else dict(answer=0, status=0, figures=0, citations=0, total=0)
        rows.append(dict(question_id=gq["question_id"], category=gq["category"], **s))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print(res.round(2).to_string(index=False))
    print("\nby category:\n", res.groupby("category")[["answer", "status", "figures", "citations", "total"]].mean().round(3))
    print("\nmean:", res[["answer", "status", "figures", "citations", "total"]].mean().round(3).to_dict())
    print(f"correct (all facts + right status): {int(((res.answer == 1) & (res.status == 1)).sum())}/{len(res)}")


if __name__ == "__main__":
    main(sys.argv[1:])
