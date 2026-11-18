# DATA266 Lab 2 — Team 34

Shared team repository for the three Lab 2 parts. Each team member's work is kept in their own named folder under the
matching part directory. Graders use the tags `p1-v1.0`, `p2-v1.0` and `p3-v1.0`.

## Repository layout

```text
README.md
requirements.txt    # points to each part's pinned requirements (the parts run on different runtimes)
part1_rag/          # multimodal RAG over the competition documents (Kaggle)
  rahul_mohan/
    notebooks/p1_rag.ipynb              # final system, executed Kaggle run (2x T4), writes submission.csv
    notebooks/p1_golden.ipynb           # final system on the 25 golden questions (executed)
    notebooks/p1_golden_baseline.ipynb  # dense-only baseline on the same questions (executed)
    notebooks/p1_eval.ipynb             # task 1.4 results table from the two golden runs (executed)
    src/                                # notebook sources, notebook builders, scorer, rating page
    eval/                               # golden set, saved golden runs, groundedness ratings, results table
    requirements.txt
part2_lora/         # Stable Diffusion 1.5 LoRA fine-tuning
  rahul_mohan/
    notebooks/p2_train.ipynb          # LoRA r4 and r32 training (executed)
    notebooks/p2_generate_eval.ipynb  # generation and evaluation (executed)
    notebooks/p2_push_hf.ipynb        # Hugging Face upload (executed): adapters, model cards, public access check
    src/                              # notebook sources + rater app
    eval/                             # 20 prompts, 90/10 split, lineage hashes (fixed before training)
    outputs/p2/                       # results and training tables, figures, human-eval files, HF links
    requirements.txt
part3_agents/       # five-agent image understanding + generation system (LangGraph, Gradio)
  rahul_mohan/
    notebooks/p3_agents.ipynb   # executed Colab T4 run: 15 scenarios, 10 red-team cases, results table, app
    src/                        # notebook source, review-page builder
    eval/                       # 15 scenarios and 10 red-team cases with pass/fail rubrics
    outputs/p3/                 # JSON traces, human ratings, results table
    requirements.txt
```

## How to run each part

Each part's README has the full steps; in short:

| Part | Runtime | Steps | Details |
|---|---|---|---|
| 1 | Kaggle, GPU T4 x2, competition data attached | Run all `p1_rag.ipynb` (about 42 min for 40 questions); set `QUESTIONS_PATH` for another question file. For the evaluation, run `p1_golden.ipynb` and `p1_golden_baseline.ipynb`, copy their outputs into `eval/p1_runs/`, then run `p1_eval.ipynb` on any CPU | `part1_rag/rahul_mohan/README.md` |
| 2 | Colab, GPU (L4 used; peak 3.6 GiB, so a T4 fits), secret `HF_TOKEN` | Run all `p2_train.ipynb`, then `p2_generate_eval.ipynb`, then `p2_push_hf.ipynb` (CPU is enough) | `part2_lora/rahul_mohan/README.md` |
| 3 | Colab, GPU T4, secret `HF_TOKEN` | Run all `p3_agents.ipynb` (about 10 min with saved traces, 25 min from scratch); set `LAUNCH_APP = True` for the Gradio demo | `part3_agents/rahul_mohan/README.md` |

Rebuild a notebook from its source (cells separated by `# %%`):

```bash
python part1_rag/rahul_mohan/src/build_notebook.py part1_rag/rahul_mohan/src/p1_rag_source.py p1_rag.ipynb
```

All notebooks use seed 266. Model weights and generated images are on Hugging Face, not in this repository. No hosted
API is used: every model is open and runs on the notebook's GPU.

## Run information printed in Cell 1

Cell 1 of every notebook prints the Git commit, library versions, GPU and seed.

- **Part 1:** Kaggle runs a notebook outside a Git checkout, so the `commit` field is empty. Cell 1 prints the code
  version instead (`p1-v3.16`), and the notebooks here are identical to the ones run.
- **Parts 2 and 3:** the notebooks were developed and run from our course repository (`data266-9067`, private,
  read access on request). The commit they print (`49bacd3`, `86e7812`) is a commit of that repository. The
  executed notebooks were copied here unchanged.

## When the evaluation sets were fixed

| Part | Evaluation set | Fixed | Evidence |
|---|---|---|---|
| 1 | `eval/golden_set.csv` (25 questions) | Written 2026-11-10, unchanged since | Course repository commit `86e7812` (2026-11-13); first golden-notebook run 2026-11-14 |
| 2 | `eval/p2_prompts.csv` (20 prompts) and the 90/10 split | Before any training run | Course repository commits `f2b8162` and `49bacd3` (2026-11-11); SHA-256 of both in `eval/p2_lineage.json`, which the notebooks check |
| 3 | `eval/p3_scenarios.csv`, `eval/p3_redteam.csv` | Before the first run | The first run (2026-11-14, `outputs/p3/v1_run/run_info.json`) recorded their SHA-256 prefixes `c0735f470b98` and `d59d584a5ca4`; the committed files (`51e1ee8`) have the same hashes |
