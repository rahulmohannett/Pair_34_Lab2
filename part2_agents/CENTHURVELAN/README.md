# DATA 266 Lab 2 — Part 2 VS Code GPU Package

This package is configured for a Linux GPU system running notebooks in VS Code. It uses the Naruto BLIP Captions dataset and Stable Diffusion 1.5, then trains two LoRA adapters with ranks 4 and 32.

## Folder structure

```text
part2_vscode/
├── notebooks/
│   ├── p2_train.ipynb
│   ├── p2_generate_eval.ipynb
│   └── p2_push_hf.ipynb
├── src/
├── eval/
└── outputs/p2/
```

Large files are intentionally created under `outputs/p2_drive/` and are not included in this ZIP. The dataset and pretrained models download from Hugging Face during the first run.

## Setup in VS Code

From a terminal opened in this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-vscode.txt
export P2_ROOT="$PWD"
export P2_DATA_ROOT="$PWD/outputs/p2_drive"
code .
```

Use the `.venv` Python kernel in VS Code. Confirm that `torch.cuda.is_available()` is `True` before training. The CUDA-enabled PyTorch build is normally provided by the lab system; do not replace it blindly with a CPU-only wheel. On Windows, keep `dataloader_num_workers` at 0 because the training script uses a local preprocessing function and `spawn` cannot pickle it.

## Run order

1. Open and run `notebooks/p2_train.ipynb` from top to bottom. This downloads and prepares the dataset, creates the fixed 20 prompts, and trains both LoRA runs.
2. Open and run `notebooks/p2_generate_eval.ipynb` from top to bottom. This generates the base and LoRA images, computes KID/CLIP/LPIPS, performs the NSFW and memorization checks, creates grids, and creates the human-evaluation pack.
3. Have both team members and one outside rater complete the generated blinded pairs. Save their files as `outputs/p2/human_eval/rater_<name>.csv`, then rerun the human-evaluation result cell.
4. After training is complete, set `PRIVATE = True` or `False` as appropriate in `p2_push_hf.ipynb`, set `HF_TOKEN` in the terminal environment, and run it to upload both adapters and their model cards.

```bash
export HF_TOKEN="<your-write-token>"
```

Do not put the token in the notebook or commit it to GitHub.

## Expected resource use

The reference L4 run took about 68 minutes per 5,000-step LoRA run. A slower GPU may take longer. Checkpoints are saved so an interrupted training run can be resumed.

## Reproducibility

The notebooks use seed 266, a pinned dataset revision after the first preparation run, a 90/10 split, 512×512 images, the same training budget for both runs, and the same generation settings for all three models.
