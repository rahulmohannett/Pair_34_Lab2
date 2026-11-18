# Part 1 - Centhurvelan Ramalingam Sakthivel (baseline runs)

Baseline runs of the team's Part 1 RAG notebook (version p1-v3.3) on a single Colab T4 GPU, with Qwen2.5-VL-7B-Instruct
reading the figures and Qwen2.5-7B-Instruct writing the answers. The team's final Part 1 system and its evaluation are
in `../rahul_mohan/`.

## Runs

| Run | GPU | Questions | Total time | Files |
|---|---|---|---|---|
| `20261115_0927` | Tesla T4 | 40 practice questions | 14.0 min | `outputs/run_20261115_0927.json`, `outputs/submission_20261115_0927.csv`, `outputs/traces_20261115_0927.jsonl` |
| `20261116_0055` | Tesla T4 | 40 practice questions | 14.8 min (median 20.5 s per question) | `eval/p1_runs/`, `outputs/`; summary in `eval/p1_results_table.csv` |

`outputs/facts_table.csv` and `outputs/figure_descriptions_*.json` are the fact table and the cached figure
transcriptions written by these runs.

## Files

- `notebooks/p1_rag.ipynb`, `src/`: the notebook and its source; the run outputs are the files above.
- `data/part1/`: the three PDFs, `kb_manifest.json` and `validate_submission.py`. `Questions.csv` comes from the
  Kaggle competition data.
- `eval/score_p1.py`: the scorer.

## Human evaluation

Rater B of the groundedness ratings of the final system: `../rahul_mohan/eval/p1_ratings/p1_ratings_B.csv`.

## AI use

See `AI_USE.md`.
