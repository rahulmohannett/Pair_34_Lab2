"""Build notebooks/p1_golden.ipynb: the p1_rag pipeline with RUN_GOLDEN on, plus a first cell that writes our own
golden-set files (eval/golden_questions.csv, eval/golden_set.csv, eval/score_p1.py) to the working folder, so the
run needs no extra Kaggle dataset. The golden questions are ours, written from the documents; they are not the
competition's questions.

With --baseline it builds notebooks/p1_golden_baseline.ipynb instead: the same pipeline on the 25 golden questions
with dense retrieval only (no BM25, no reranker, no exact-figure pinning, no per-document quota, no fact-table lookup).
That run is the baseline row of the Part 1 results table (notebooks/p1_eval.ipynb).

With --dev it builds notebooks/p1_golden_dev.ipynb: the final pipeline on the 25 golden questions plus our 22 dev
questions (eval/dev_set.csv, tuning data). That notebook is a local tuning tool and stays out of the team repository;
notebooks/p1_golden.ipynb and notebooks/p1_golden_baseline.ipynb hold only the 25 golden questions.

Usage: python3 src/build_p1_golden.py [--baseline | --dev] [src/some_candidate_source.py]
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LAB2 = os.path.dirname(HERE)
# Optional argument: another source file (a candidate version under test); default is the final source.
ARGS = [a for a in sys.argv[1:] if a not in ("--baseline", "--dev")]
BASELINE = "--baseline" in sys.argv[1:]
WITH_DEV = "--dev" in sys.argv[1:]
assert not (BASELINE and WITH_DEV), "the dev questions are scored with the final system only"
SRC = ARGS[0] if ARGS else os.path.join(HERE, "p1_rag_source.py")
src = open(SRC, encoding="utf-8").read()
old = 'RUN_GOLDEN = os.environ.get("RUN_GOLDEN", "0") == "1"'
assert src.count(old) == 1
src = src.replace(old, 'RUN_GOLDEN = os.environ.get("RUN_GOLDEN", "1") == "1"   # golden-set notebook: on by default')

# Per-question seconds, so the evaluation notebook can take the median over the 25 golden questions only.
c = "sub.to_csv(WORK / f\"{OUT}_{stamp.replace('_golden', '')}.csv\", index=False)   # keep every run\n"
assert src.count(c) == 1
src = src.replace(c, c + (
    'pd.DataFrame({"question_id": [r["question_id"] for r in rows], "seconds": timings}).to_csv(\n'
    '    WORK / f"seconds_{stamp.replace(\'_golden\', \'\')}.csv", index=False)\n'))
if BASELINE:   # dense retrieval only; the fact-table lookup is part of the final system, so it is off as well
    for old, new in (('RUN_BASELINE = os.environ.get("RUN_BASELINE", "0") == "1"', 'RUN_BASELINE = os.environ.get("RUN_BASELINE", "1") == "1"'),
                     ('USE_FACTS = os.environ.get("USE_FACTS", "1") == "1"', 'USE_FACTS = os.environ.get("USE_FACTS", "0") == "1"')):
        assert src.count(old) == 1
        src = src.replace(old, new)

# Dev questions (eval/dev_set.csv, tuning data) run in the same pass and are scored separately. These two patches
# apply to the golden notebook only; p1_rag_source.py and notebooks/p1_rag.ipynb are not changed by them.
if WITH_DEV and 'dev_questions.csv' not in src:
    a = '    QUESTIONS_PATH = str(EVAL_DIR / "golden_questions.csv")\n'
    assert src.count(a) == 1
    src = src.replace(a, a + (
        '    if (EVAL_DIR / "dev_questions.csv").exists():\n'
        '        _qs = pd.concat([pd.read_csv(EVAL_DIR / n, dtype=str, keep_default_na=False)\n'
        '                         for n in ("golden_questions.csv", "dev_questions.csv")], ignore_index=True)\n'
        '        QUESTIONS_PATH = str(WORK / "golden_dev_questions.csv"); _qs.to_csv(QUESTIONS_PATH, index=False)\n'))
    b = ('    r = subprocess.run([sys.executable, str(EVAL_DIR / "score_p1.py"), str(out_csv), str(EVAL_DIR / "golden_set.csv")],\n'
         '                       capture_output=True, text=True)\n'
         '    print(r.stdout, r.stderr)\n'
         '    (WORK / f"golden_score_{stamp.replace(\'_golden\', \'\')}.txt").write_text(r.stdout + r.stderr)\n')
    assert src.count(b) == 1
    src = src.replace(b, (
        '    for _set, _name in (("golden_set.csv", "golden_score"), ("dev_set.csv", "dev_score")):\n'
        '        if not (EVAL_DIR / _set).exists():\n'
        '            continue\n'
        '        r = subprocess.run([sys.executable, str(EVAL_DIR / "score_p1.py"), str(out_csv), str(EVAL_DIR / _set)],\n'
        '                           capture_output=True, text=True)\n'
        '        print(f"== {_set}\\n", r.stdout, r.stderr)\n'
        '        (WORK / f"{_name}_{stamp.replace(\'_golden\', \'\')}.txt").write_text(r.stdout + r.stderr)\n'))

files = {name: open(os.path.join(LAB2, "eval", name), encoding="utf-8").read()
         for name in (("golden_questions.csv", "golden_set.csv", "dev_questions.csv", "dev_set.csv", "score_p1.py") if WITH_DEV else
                      ("golden_questions.csv", "golden_set.csv", "score_p1.py"))}
writer = ["# %% [markdown]",
          "# # Part 1 - golden-set evaluation" + (" (baseline: dense retrieval only)" if BASELINE else ""),
          ("# The p1_rag pipeline run on our own 25 golden questions with dense retrieval only: no BM25, no reranker, no exact-figure"
           if BASELINE else
           "# The p1_rag pipeline run on our own 25 golden questions plus 22 dev questions (all written from the documents, not"
           if WITH_DEV else
           "# The final p1_rag pipeline run on our own 25 golden questions (written from the documents, not the competition's"),
          ("# pinning, no per-document quota and no fact-table lookup. Documents, chart descriptions, models and prompt are the same as in the final system. Attach the competition data; no other dataset is needed."
           if BASELINE else
           "# the competition's questions), each set scored separately with eval/score_p1.py. Attach the competition data; no other dataset is needed."
           if WITH_DEV else
           "# questions), scored with eval/score_p1.py. Attach the competition data; no other dataset is needed."),
          "# Writes golden_<time>.csv, seconds_<time>.csv, traces, run_<time>_golden.json and golden_score_<time>.txt; never writes submission.csv.",
          "",
          "# %%",
          "# Golden-set files, written to the working folder (unless an attached dataset already has them)",
          "import os as _os",
          "from pathlib import Path as _P",
          "_dst = _P('/kaggle/working/eval' if _os.path.isdir('/kaggle/working') else '../eval')",
          "_dst.mkdir(parents=True, exist_ok=True)",
          f"_FILES = {files!r}",
          "for _n, _t in _FILES.items():",
          "    if not (_dst / _n).exists():",
          "        (_dst / _n).write_text(_t, encoding='utf-8')",
          "print('golden-set files in', _dst, sorted(p.name for p in _dst.iterdir()))",
          ""]
# The p1 source opens with a markdown title cell; the writer cell goes first.
src = "\n".join(writer) + "\n" + src
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
    fh.write(src)
    tmp = fh.name
try:
    subprocess.run([sys.executable, os.path.join(HERE, "build_notebook.py"), tmp,
                    os.path.join(LAB2, "notebooks", "p1_golden_baseline.ipynb" if BASELINE else "p1_golden_dev.ipynb" if WITH_DEV else "p1_golden.ipynb")], check=True)
finally:
    os.remove(tmp)
