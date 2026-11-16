# Part 1: Multimodal RAG over a read-only knowledge base

Answers questions about three economics PDFs (text, figures and tables) as JSON with status, citations and figure
references, following the lab's answering rules. Open models only; no hosted API.

## Notebooks

| Notebook | What it does | Where it runs | Time |
|---|---|---|---|
| `notebooks/p1_rag.ipynb` | Final system. Reads `QUESTIONS_PATH` (default: the competition's `Questions.csv`) and writes `submission.csv`. Saved with the outputs of the Kaggle run that scored 0.922 on the public leaderboard. | Kaggle, GPU T4 x2, competition data attached | about 43 min for 40 questions |
| `notebooks/p1_golden.ipynb` | Final system on our 25 golden questions; writes `golden_<time>.csv`, traces, seconds per question and the score. | Kaggle, same settings | about 22 min |
| `notebooks/p1_golden_baseline.ipynb` | Baseline on the same 25 questions: dense retrieval only (no BM25, reranker, exact-figure retrieval, per-document quota or fact-table lookup). | Kaggle, same settings | about 17 min |
| `notebooks/p1_eval.ipynb` | Results table (task 1.4) from the two saved golden runs in `eval/p1_runs/`: correctness /25, Recall@5, citation accuracy, groundedness ratings, median seconds per question, Appendix C checks. | Any CPU | seconds |

## Run it

1. On Kaggle, open `notebooks/p1_rag.ipynb`, attach the competition data, choose GPU T4 x2 and run all. To answer
   another question file, set the environment variable `QUESTIONS_PATH` to it. Output: `submission.csv`.
2. For the evaluation, run `notebooks/p1_golden.ipynb` and `notebooks/p1_golden_baseline.ipynb` the same way. Both
   write the golden-set files they need themselves, so no extra dataset is attached. Copy each run's
   `golden_*.csv`, `traces_*_golden.jsonl`, `run_*_golden.json` and `seconds_*.csv` into `eval/p1_runs/final/`
   and `eval/p1_runs/baseline/`.
3. Run `notebooks/p1_eval.ipynb` locally (`jupyter nbconvert --to notebook --execute notebooks/p1_eval.ipynb --inplace`).
   It writes `eval/p1_results_table.csv`.

## Source files

Each notebook is built from a plain source file with cells separated by `# %%`:

```bash
python src/build_notebook.py src/p1_rag_source.py notebooks/p1_rag.ipynb
python src/build_p1_golden.py               # notebooks/p1_golden.ipynb
python src/build_p1_golden.py --baseline    # notebooks/p1_golden_baseline.ipynb
python src/build_notebook.py src/p1_eval_source.py notebooks/p1_eval.ipynb
```

## Evaluation files

- `eval/golden_set.csv`: our 25 questions written from the three PDFs, with expected answer, required facts, status
  and source page or figure. `eval/golden_questions.csv` holds the questions only. They were written and committed
  before tuning, in the course repository (`data266-9067`, commit `86e7812`, 2026-11-13); the first tuning run on
  them was on 2026-11-14.
- `eval/score_p1.py`: scorer (0.4 facts, 0.2 status, 0.2 figure references, 0.2 citations).
- `eval/p1_runs/`: the saved baseline and final golden runs used by `p1_eval.ipynb`.
