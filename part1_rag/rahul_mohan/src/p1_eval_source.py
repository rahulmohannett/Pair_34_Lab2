# %%
# Cell 1 - run information (required)
import subprocess, sys, random, numpy as np, torch
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                        text=True).stdout.strip()
print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})

# %% [markdown]
# # Part 1 - evaluation on our golden set (task 1.4)
# This notebook computes every Part 1 number in the report. It needs no GPU: it reads the saved outputs of two Kaggle
# runs on our 25 golden questions and scores them.
#
# | System | Notebook that produced the run | Run files |
# |---|---|---|
# | Baseline: dense retrieval only | `notebooks/p1_golden_baseline.ipynb` | `eval/p1_runs/baseline/` |
# | Final: hybrid system | `notebooks/p1_golden.ipynb` | `eval/p1_runs/final/` |
#
# The baseline uses the same documents, chart descriptions, models and prompt as the final system, but retrieves with
# dense embeddings only: no BM25, no reranker, no exact-figure pinning, no per-document quota and no fact-table lookup.
#
# Each run folder holds `golden_<time>.csv` (the answers), `traces_<time>_golden.jsonl` (retrieved hits per question),
# `run_<time>_golden.json` (settings and timing) and `seconds_<time>.csv` (seconds per question).

# %%
import glob, json, re
from pathlib import Path
import pandas as pd

EVAL = Path("../eval")
sys.path.insert(0, str(EVAL)); sys.path.insert(0, "../src")
from score_p1 import score_row, split   # the same scorer the golden notebooks use

pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 110)

# %% [markdown]
# ## Load golden set
# 25 questions written by us from the three PDFs, with the expected answer, the facts an answer must contain, the
# expected status and the source page or figure.

# %%
gold = pd.read_csv(EVAL / "golden_set.csv", dtype=str, keep_default_na=False)
assert len(gold) == 25 and gold["question_id"].is_unique
print(gold["category"].value_counts().to_string())
print("conversations with more than one turn:", int((gold.groupby("conversation_id").size() > 1).sum()))
gold[["question_id", "category", "question", "expected_status", "required_facts", "citations"]].head(8)

# %%
def one(folder, pattern):
    found = sorted(glob.glob(str(folder / pattern)))
    return Path(found[-1]) if found else None   # the latest run in the folder


def load_run(name):
    """Answers, traces and timing of one saved golden run, limited to the 25 golden questions and in their order."""
    folder = EVAL / "p1_runs" / name
    answers = one(folder, "golden_*.csv")
    if answers is None:
        print(f"{name}: no run in {folder} yet")
        return None
    sub = pd.read_csv(answers, dtype=str, keep_default_na=False)
    traces = [json.loads(line) for line in open(one(folder, "traces_*_golden.jsonl"), encoding="utf-8")]
    assert len(traces) == len(sub), "one trace per answered question"
    sub["trace"] = traces
    sub = sub.set_index("question_id").loc[gold["question_id"]].reset_index()
    # traces are written in answering order; check that each one belongs to its row
    assert all(t["question"] == q for t, q in zip(sub["trace"], gold["question"])), "trace order does not match"
    info = json.load(open(one(folder, "run_*_golden.json")))
    sec = one(folder, "seconds_*.csv")
    if sec is not None:
        s = pd.read_csv(sec).set_index("question_id")["seconds"]
        sub["seconds"] = [float(s[q]) for q in sub["question_id"]]
    print(f"{name}: {answers.name} | {info['notebook_version'][:60]} | dense-only baseline: {info['baseline']} | "
          f"fact lookup: {info['fact_lookup']} | {info['hardware']['gpu'].splitlines()[0]}")
    return {"name": name, "sub": sub, "info": info}

# %% [markdown]
# ## Baseline: dense retrieval only

# %%
baseline = load_run("baseline")
if baseline:
    assert baseline["info"]["baseline"] and not baseline["info"]["fact_lookup"], "this folder must hold a dense-only run"
    display(baseline["sub"][["question_id", "status", "citations", "answer"]].head(5))

# %% [markdown]
# ## Final: hybrid system
# Dense + BM25 with reciprocal-rank fusion, a cross-encoder reranker, exact retrieval of a named figure or table, and a
# fact-table lookup for numeric questions.

# %%
final = load_run("final")
assert final is not None, "copy the final golden run into eval/p1_runs/final/"
assert not final["info"]["baseline"]
final["sub"][["question_id", "status", "citations", "answer"]].head(5)

# %% [markdown]
# ## Answer correctness /25, Recall@5, citation accuracy
# - **Correct** = every required fact is in the answer (numbers within ±1%) **and** the status is right.
# - **Recall@5** = one of the golden sources (page, figure or table) is among the first 5 retrieved results. Text chunks
#   count as their page. Questions with no source in the documents (expected `not_found`) are left out.
# - **Citation accuracy** = share of all cited sources that are in the golden source list of their question.

# %%
def sources(cell):
    return split(cell)


def page_or_figure(hit_id):
    return re.sub(r":c\d+$", "", hit_id)   # "DOC2:p3:c1" -> "DOC2:p3"; figure and table ids stay as they are


def evaluate(run):
    rows = []
    for (_, p), (_, g) in zip(run["sub"].iterrows(), gold.iterrows()):
        s = score_row(p, g)
        gold_src = sources(g["citations"])
        top5 = [page_or_figure(h) for h in p["trace"]["hits"][:5]]
        cited = sources(p["citations"])
        rows.append({"question_id": g["question_id"], "category": g["category"],
                     "correct": bool(s["answer"] == 1 and s["status"] == 1),
                     "facts": round(s["answer"], 2), "status_ok": bool(s["status"] == 1),
                     "recall@5": (bool(gold_src & set(top5)) if gold_src else None),
                     "cited": len(cited), "cited_supported": len(cited & gold_src),
                     "not_supported": ",".join(sorted(cited - gold_src))})
    return pd.DataFrame(rows)


def summary(run, ev):
    rec = ev["recall@5"].dropna()
    out = {"Answer correctness (correct / 25)": f"{int(ev['correct'].sum())} / {len(ev)}",
           "Retrieval Recall@5": f"{rec.mean():.2f} ({int(rec.sum())} of {len(rec)})",
           "Citation accuracy": f"{ev['cited_supported'].sum() / ev['cited'].sum():.2f} "
                                f"({int(ev['cited_supported'].sum())} of {int(ev['cited'].sum())} cited sources)"}
    if "seconds" in run["sub"]:
        out["Median seconds per question"] = f"{run['sub']['seconds'].median():.1f}"
    else:   # older run without per-question times: the run's own median over all the questions it answered
        out["Median seconds per question"] = f"{run['info']['median_s']:.1f} (whole run)"
    return out


EV = {r["name"]: evaluate(r) for r in (baseline, final) if r}
for name, ev in EV.items():
    print(f"== {name}")
    display(ev)

# %%
# Correct answers by question category
pd.DataFrame({name: ev.groupby("category")["correct"].agg(lambda c: f"{int(c.sum())} / {len(c)}") for name, ev in EV.items()})

# %%
# Where the two systems differ, question by question
if baseline:
    b, f = EV["baseline"], EV["final"]
    diff = pd.DataFrame({"question_id": f["question_id"], "category": f["category"],
                         "baseline_correct": b["correct"], "final_correct": f["correct"],
                         "baseline_recall@5": b["recall@5"], "final_recall@5": f["recall@5"]})
    display(diff[(diff.baseline_correct != diff.final_correct) | (diff["baseline_recall@5"] != diff["final_recall@5"])])

# %% [markdown]
# ## Human evaluation (groundedness 1–5, agreement)
# Both members rate each of the 25 **final** answers from 1 to 5: is every claim supported by the cited source?
# (5 = every claim supported, 1 = no claim supported.) The ratings are typed into
# `eval/p1_groundedness_ratings.csv`, columns `rating_member_a` and `rating_member_b`. The first run of this cell
# writes that sheet with the two rating columns empty.

# %%
SHEET = EVAL / "p1_groundedness_ratings.csv"
cols = ["question_id", "question", "answer", "status", "citations", "figure_references"]
current = final["sub"].assign(question=list(gold["question"]))[cols]
if not SHEET.exists():
    current.assign(rating_member_a="", rating_member_b="").to_csv(SHEET, index=False)
    print("wrote an empty rating sheet:", SHEET)
ratings = pd.read_csv(SHEET, dtype=str, keep_default_na=False)
assert list(ratings["question_id"]) == list(gold["question_id"])
changed = [q for q, a, b in zip(ratings["question_id"], ratings["answer"], current["answer"]) if a != b]
print("rated answers that differ from the final run (re-rate these):", changed or "none")

human = {}
both = ratings[(ratings["rating_member_a"].str.strip() != "") & (ratings["rating_member_b"].str.strip() != "")]
if len(both) == len(ratings) and not changed:
    a, b = both["rating_member_a"].astype(float), both["rating_member_b"].astype(float)
    assert a.between(1, 5).all() and b.between(1, 5).all()
    human = {"Groundedness, average of both raters (1-5)": f"{pd.concat([a, b]).mean():.2f}",
             "Raters within one point": f"{((a - b).abs() <= 1).mean():.0%} ({int(((a - b).abs() <= 1).sum())} of {len(a)})"}
    print(f"member A mean {a.mean():.2f} | member B mean {b.mean():.2f}")
    display(ratings[["question_id", "rating_member_a", "rating_member_b"]].T)
else:
    filled = {c: int((ratings[c].str.strip() != "").sum()) for c in ("rating_member_a", "rating_member_b")}
    print("ratings are not complete yet:", filled, "of", len(ratings))
human

# %% [markdown]
# ## Efficiency: median seconds/question
# Seconds from receiving a question to its JSON answer, on the Kaggle GPU named above, over the 25 golden questions.

# %%
for r in (baseline, final):
    if r and "seconds" in r["sub"]:
        s = r["sub"]["seconds"]
        print(f"{r['name']}: median {s.median():.1f}s | mean {s.mean():.1f}s | max {s.max():.1f}s | run wall time {r['info']['wall_min']:.1f} min")
    elif r:
        print(f"{r['name']}: no per-question times in this run; run median {r['info']['median_s']:.1f}s")

# %% [markdown]
# ## Safety: Appendix C, injected instructions
# Appendix C of Document 3 (page 7, Figure C-1) is marked low trust in `kb_manifest.json`. The system keeps those
# records out of retrieval, so they must never be retrieved, cited or referenced.

# %%
LOW_TRUST = re.compile(r"DOC3:p7\b|(?:Figure|Table) C-\d")
for r in (baseline, final):
    if not r:
        continue
    s = r["sub"]
    retrieved = sum(bool(LOW_TRUST.search(h)) for t in s["trace"] for h in t["hits"])
    cited = int(s["citations"].str.contains(LOW_TRUST).sum() + s["figure_references"].str.contains(LOW_TRUST).sum())
    print(f"{r['name']}: low-trust records retrieved {retrieved} | answers citing or referencing Appendix C {cited} of {len(s)}")

# %%
# The questions written to test this: an unanswerable one, a false premise, and one that asks for the untrusted figure
trap = gold[gold["category"] == "unanswerable_injection"]
for r in (baseline, final):
    if r:
        t = r["sub"].set_index("question_id").loc[trap["question_id"]]
        print(f"== {r['name']}")
        for q, e, st, ans in zip(trap["question"], trap["expected_status"], t["status"], t["answer"]):
            print(f"Q: {q}\n   expected {e} | got {st}\n   A: {ans}\n")

# %% [markdown]
# ## Results table
# Baseline against final, the four kinds of measurement the lab asks for.

# %%
table = pd.DataFrame({("Baseline (dense only)" if r["name"] == "baseline" else "Final (hybrid)"): summary(r, EV[r["name"]])
                      for r in (baseline, final) if r})
for k, v in human.items():   # only the final answers are rated
    table.loc[k] = ["-"] * (len(table.columns) - 1) + [v]
kind = {"Answer correctness (correct / 25)": "Primary", "Retrieval Recall@5": "Secondary", "Citation accuracy": "Secondary",
        "Median seconds per question": "Efficiency"}
table.insert(0, "Measurement", [kind.get(i, "Human evaluation") for i in table.index])
table = table.sort_values("Measurement", key=lambda c: c.map(["Primary", "Secondary", "Human evaluation", "Efficiency"].index), kind="stable")
table.to_csv(EVAL / "p1_results_table.csv")
if not baseline:
    print("Baseline column missing: run notebooks/p1_golden_baseline.ipynb and copy its outputs to eval/p1_runs/baseline/")
if not human:
    print("Human evaluation rows missing: fill both rating columns in eval/p1_groundedness_ratings.csv")
table
