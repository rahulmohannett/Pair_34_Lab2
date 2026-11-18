# Part 3 — multimodal multi-agent system (rahul_mohan)

Five agents in a LangGraph graph over one typed (pydantic) state:
Planner (input safety) → Vision Understanding → Prompt Engineering → Image Generation → Critique (output safety) →
accept, or revise (back to Prompt Engineering, at most 2 times). Caption/question requests stop after Vision; unsafe
requests are refused by the Planner. A Gradio app shows every agent's output for one request.

```text
notebooks/p3_agents.ipynb    executed run (Colab T4): agents, 15 scenarios + 10 red-team cases, results table, app
src/p3_agents_source.py      notebook source (cells split by "# %%"); rebuild with src/build_notebook.py
src/p3_review_page.py        builds outputs/p3/review.html, the page both members rated in
eval/p3_scenarios.csv        15 scenarios with pass/fail rubrics (written before the first run; see the root README)
eval/p3_redteam.csv          10 red-team cases
outputs/p2/hf_links_s5000.json  Hugging Face repo + revision of the Part 2 LoRA the notebook loads
outputs/p3/traces/           one JSON trace per case: every agent's input, output and seconds
outputs/p3/rating_sheet_filled.csv, critic_labels_filled.csv, redteam_filled.csv   resolved human labels
outputs/p3/ratings/          each member's independent ratings (A, B) before resolution
outputs/p3/results_table.csv, run_info.json, app_selftest.json
outputs/p3/v1_run/           results of the first run, before the v1 -> v2 fixes (kept for the report)
```

## Run it
1. Colab → GPU T4. Add a Colab secret `HF_TOKEN` (read access to the Part 2 LoRA repo) and enable notebook access.
2. Open `notebooks/p3_agents.ipynb` from a clone of this repo. If the clone is not at our Drive path, set
   `P3_LAB2_DIR` to the folder that contains `notebooks/` (or start the kernel inside `notebooks/`).
3. Runtime → Run all (~10 min when the saved traces are reused; ~25 min from scratch). Cases are re-run only if
   the eval files or the config changed (their hashes are stored in each trace).
4. Demo: set `LAUNCH_APP = True` in Cell 1; the last cell prints a public Gradio link.

Generated images go to `P3_DATA_DIR` (default: Google Drive on Colab, `outputs/p3_drive/` elsewhere), not to Git.
