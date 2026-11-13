# DATA266 Lab 2 — Team 34

Shared team repository for the three Lab 2 parts. Each team member's work is kept in their own named folder under the matching part directory.

## Repository layout

```text
README.md
requirements.txt
part1_rag/          # multimodal RAG over the competition documents (Kaggle)
  rahul_mohan/
    notebooks/p1_rag.ipynb      # executed Kaggle run (2x T4)
    src/p1_rag_source.py        # notebook source, cells split by "# %%"
    src/build_notebook.py       # rebuilds a notebook from its source file
    src/score_p1.py             # golden-set scorer
    src/prompts/
part2_lora/         # Stable Diffusion 1.5 LoRA fine-tuning
  rahul_mohan/
    notebooks/p2_train.ipynb          # LoRA r4 and r32 training (executed)
    notebooks/p2_generate_eval.ipynb  # generation and evaluation (executed)
    notebooks/p2_push_hf.ipynb        # Hugging Face upload
    src/                              # notebook sources + rater app
part3_agents/       # multi-agent system (to be added)
```

Rebuild a notebook from its source:

```bash
python part1_rag/rahul_mohan/src/build_notebook.py part1_rag/rahul_mohan/src/p1_rag_source.py p1_rag.ipynb
```

All notebooks use seed 266 and print the Git commit, library versions and GPU in their first cell. Model weights and generated images are kept on Hugging Face, not in this repository.
