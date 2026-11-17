# Part 2 — LoRA fine-tuning of Stable Diffusion 1.5 (rahul_mohan)

This part fine-tunes SD 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`) on `lambdalabs/naruto-blip-captions` with two
LoRA runs:

| Run | LoRA rank | Steps | LR | Batch |
|---|---|---|---|---|
| 1 | 4 | 5000 | 1e-4 | 1 × 4 gradient accumulation |
| 2 | 32 | 5000 | 1e-4 | 1 × 4 gradient accumulation |

Both runs use the same data, seed 266 and the 90/10 split. Each run is compared with the base model on 20 fixed prompts
× seeds {0, 1}. Run 2 is the final run, chosen by a rule written into the notebook before any result existed: the lower
KID on the test captions.

```text
notebooks/p2_train.ipynb          executed run (Colab L4): data preparation (2.1), both LoRA runs + training table (2.2)
notebooks/p2_generate_eval.ipynb  executed run: 120 + 366 images, KID / CLIP / LPIPS, grids, memorization, failure
                                  modes, NSFW audit, human-evaluation pack and win rates, HF dataset upload (2.3, 2.4)
notebooks/p2_push_hf.ipynb        executed run (CPU): adapters + model cards to Hugging Face, download-and-hash check
src/*_source.py                   notebook sources (cells split by "# %%"); rebuild with src/build_notebook.py
src/p2_build_rater_app.py         builds the blinded rater page from the human-evaluation pack
eval/p2_prompts.csv               the 20 prompts: 10 in-domain (test captions), 5 out-of-domain, 5 compositional
eval/p2_split.json                the 90/10 split (seed 266): 1099 train / 122 test file indices
eval/p2_lineage.json              SHA-256 of the prompts and of the split, hashed before any training run
outputs/p2/data_prep.json         removed images (0 corrupted, 0 duplicates), caption statistics
outputs/p2/training_table.csv     trainable parameters, LR, steps, batch, peak GPU memory, time, GPU-hours per run
outputs/p2/run*_s5000_meta.json   per-run settings, versions, prompt and split hashes; *_loss.csv = loss curves
outputs/p2/results_table.csv      base vs run 1 vs run 2: KID, CLIP score, LPIPS, NSFW flags, time, GPU-hours
outputs/p2/generations.csv        one row per generated image (model, prompt, seed, NSFW flag, image source, CLIP score)
outputs/p2/grid_1.png, grid_2.png side-by-side grids for 8 prompts
outputs/p2/memorization.csv, memorization_top3.png   nearest training image (CLIP) for 10 in-domain images
outputs/p2/failure_*.png, closeup_rates.json, edge_bar_rates.json   failure-mode evidence
outputs/p2/nsfw_audit.json, nsfw_flags.csv, same_sample_check.csv   safety-checker flags and their audit
outputs/p2/human_eval/            key.csv (answer key) and the three raters' files rater_<name>.csv
outputs/p2/human_eval_*.csv       win rates per rater, by prompt type, rater agreement, and every pick
outputs/p2/hf_links_s5000.json, hf_dataset_link.json   Hugging Face repos and the commit of each upload
outputs/p2/stale_v2_run/, stale_v3_run/   tables from the earlier evaluation runs, kept for the report (see below)
```

The 1500-step runs that came before the final 5000-step runs are recorded in `run1_lora_r4_meta.json`,
`run2_lora_r32_meta.json`, the matching `*_loss.csv` files and `hf_links_s1500.json`.

## Evaluation fixes (v2 → v3 → v4)
The Stable Diffusion safety checker replaces an image it flags with a black one. It flagged 36–39 % of the LoRA
images and none of the base images.
- **v2** scored the black images as if they were real outputs.
- **v3** left the flagged images out. As a result, each model was scored on a different, non-random set of images
  (12 vs 20 in-domain images for run 2).
- **v4** (the final run) scores every model on all 40 / 122 images:
  - The team inspected the 14 flagged images of the 120-image set and found only false positives.
  - Each flagged image is generated again with the same seed and the checker off. `same_sample_check.csv`
    regenerates unflagged images the same way and shows they match the stored samples up to GPU rounding.
  - The v3 numbers stay in `results_table.csv` as the `f_*` columns.

## Run it
1. **Runtime:** Colab → GPU (L4 used; a T4 also works for evaluation).
2. **Secret:** add a Colab secret `HF_TOKEN` with write access and turn on its notebook access.
3. **Paths:** the notebooks run from `notebooks/` and read `../eval` and `../outputs/p2`.
4. **`p2_train.ipynb` → Run all.** Exports the 512 × 512 images to Drive, then trains both runs (~70 min each on an
   L4). It resumes from the last checkpoint after a disconnect.
5. **`p2_generate_eval.ipynb` → Run all.** About 10 minutes when the images already exist on Drive. Images that
   exist are not generated again.
6. **`p2_push_hf.ipynb` → Run all.** A CPU runtime is enough.

Images, the 512 × 512 dataset copy, checkpoints and adapters go to Google Drive (`MyDrive/DATA266_Lab2/p2/`), not to
Git. The adapters and the 120 generated images are on Hugging Face (links in `outputs/p2/hf_links_s5000.json` and
`hf_dataset_link.json`).
