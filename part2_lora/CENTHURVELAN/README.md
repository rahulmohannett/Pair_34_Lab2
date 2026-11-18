# Part 2 - Centhurvelan Ramalingam Sakthivel (baseline run of the complete pipeline)

Baseline run of the team's complete Part 2 pipeline (notebook versions p2-train-v4 and p2-gen-eval-v4) on one
NVIDIA GeForce RTX 5090, with the same data, split, prompts and settings as the team's final run in
`../rahul_mohan/`.

| Step | Result | Where |
|---|---|---|
| Data preparation | 1,221 images, 0 removed; 1,099 train / 122 test, identical to the hashed split | `notebooks/p2_train.ipynb`, `outputs/p2/data_prep.json` |
| Training: LoRA rank 4 and rank 32, 5,000 steps each | Both runs finished all 5,000 steps | `outputs/p2/run1_lora_r4_s5000_meta.json`, `outputs/p2/run2_lora_r32_s5000_meta.json` |
| Generation and evaluation: 20 prompts x 2 seeds; KID, CLIP score, LPIPS, NSFW flags, memorization, grids | Final run chosen by the same rule as the team: rank 32 | `notebooks/p2_generate_eval.ipynb`, `outputs/p2/results_table.csv`, figures in `outputs/p2/` |
| Hugging Face | Both adapters uploaded | `outputs/p2/hf_links_s5000.json` |

Human evaluation: rater `centhur` in the team's blinded comparison (`../rahul_mohan/outputs/p2/human_eval/`).
The dataset copy, generated images and adapter weights are kept outside GitHub.
