# Part 1: Multimodal RAG over a read-only knowledge base

Answers questions about three economics PDFs (text, figures and tables) as JSON with status, citations and figure
references, following the lab's answering rules. Open models only; no hosted API.

## Notebooks

| Notebook | What it does | Where it runs | Time |
|---|---|---|---|
| `notebooks/p1_rag.ipynb` | Final system. Reads `QUESTIONS_PATH` (default: the competition's `Questions.csv`) and writes `submission.csv`. Saved with the outputs of the Kaggle run that scored 0.937 on the public leaderboard (version p1-v3.16). | Kaggle, GPU T4 x2, competition data attached | about 42 min for 40 questions |
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
   It writes `eval/p1_results_table.csv`. The first run also writes an empty `eval/p1_groundedness_ratings.csv`.
4. Groundedness ratings: `python src/p1_rating_page.py <downloaded p1_golden output folder>` writes
   `outputs/p1/rating.html` (needs the three PDFs in `data/part1/`). Each member rates the 25 answers in their own
   tab (1-5, judged only against the cited pages and figure crops shown beside each answer) and downloads
   `p1_ratings_A.csv` / `p1_ratings_B.csv`; the Combine tab writes `p1_groundedness_ratings.csv` for `eval/`.
   Rerun `notebooks/p1_eval.ipynb`.

Kaggle runs a notebook outside a Git checkout, so Cell 1 of the three Kaggle notebooks prints an empty `commit`;
it prints the code version instead (`p1-v3.16`). Their code is identical to the notebooks at tag `p1-v1.0`.
The dev questions we used while tuning are not in this repository; the golden set was never used for tuning
decisions before it was committed.

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
  and source page or figure. `eval/golden_questions.csv` holds the questions only. They were written on 2026-11-10
  and have not changed since; they were first committed in our course repository (`data266-9067`, commit `86e7812`,
  2026-11-13; private, read access on request) and the first run of the golden notebook was on 2026-11-14.
- `eval/score_p1.py`: scorer (0.4 facts, 0.2 status, 0.2 figure references, 0.2 citations).
- `eval/p1_runs/`: the saved baseline and final golden runs used by `p1_eval.ipynb`.
- `eval/p1_groundedness_ratings.csv`: both members' groundedness ratings (1-5) of the final answers;
  `eval/p1_ratings/` holds each rater's own file.
- `eval/p1_results_table.csv`: the task 1.4 table.

## Results (golden set, 25 questions)

| Measurement | Baseline (dense only) | Final (hybrid) |
|---|---|---|
| Answer correctness | 15 / 25 | 22 / 25 |
| Retrieval Recall@5 | 1.00 (23 of 23) | 0.96 (22 of 23) |
| Citation accuracy | 0.85 (46 of 54) | 0.91 (60 of 66) |
| Groundedness, average of both raters (1-5) | - | 4.66 |
| Raters within one point | - | 88% (22 of 25) |
| Median seconds per question | 40.1 | 48.2 |

Kaggle public leaderboard: 0.937.
