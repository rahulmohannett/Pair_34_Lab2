# Part 3 - Centhurvelan Ramalingam Sakthivel (baseline run of the complete pipeline)

Baseline run of the team's five-agent system (notebook version p3-v2) on one NVIDIA GeForce RTX 5090, using the
rank-32 LoRA from my own Part 2 run (`outputs/p2/hf_links_s5000.json`). The team's final run and its evaluation are in
`../rahul_mohan/`.

| Measurement | Value | Where |
|---|---|---|
| Cases run | 15 scenarios and 10 red-team cases, each with a JSON trace of every agent's input, output and time | `outputs/p3/traces/` |
| Red-team attack success | 0/10 (8 refused by the Planner, 2 broken files returned a clear error) | `outputs/p3/redteam_results.csv` |
| Median seconds per request | 14.0 without the critique loop, 15.0 with it | `outputs/p3/run_info.json` |
| Peak GPU memory | 9.31 GiB | `outputs/p3/run_info.json` |
| Gradio app self-test | Pass: domain-style request, request that must be refused, broken file | `outputs/p3/app_selftest.json` |

- `notebooks/p3_agents.ipynb`: the notebook (setup, models, agents and a demo request); the run's evaluation files
  are in `outputs/p3/`.
- `outputs/p3/review.html`, `rating_sheet.csv`, `critic_labels.csv`: the rating files for this run.
- `eval/`: the team's 15 scenarios and 10 red-team cases.
- `WINDOWS_VS_CODE.md`, `requirements.txt`: how this run was set up. Generated images are kept outside GitHub.

Human evaluation: rater B in the team's pass/fail labels (`../rahul_mohan/outputs/p3/ratings/p3_ratings_B.csv`).
