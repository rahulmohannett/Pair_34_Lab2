# Source of truth for notebooks/p2_push_hf.ipynb. Rebuild with:
#   python3 src/build_notebook.py src/p2_push_hf_source.py notebooks/p2_push_hf.ipynb

# %% [markdown]
# # Part 2 — Push the trained LoRA adapters to Hugging Face
# Uploads the two adapters that `p2_train.ipynb` saved on Drive (`MyDrive/DATA266_Lab2/p2/runs/`), so nothing has
# to be retrained. Each repo gets the adapter, its `run_meta.json` (settings, prompt and split hashes) and a short
# model card. **A CPU runtime is enough.**
#
# Before running: add a Colab secret named `HF_TOKEN` (key icon in the left sidebar) with **write** access, and
# turn on "Notebook access" for it. Repos are created **private**; set `PRIVATE = False` (or add the graders)
# before submission.

# %%
# Cell 1 - run information (required)
NOTEBOOK_VERSION = "p2-push-v2 2026-11-11: model cards + hash check for the adapters on Drive"
import os, subprocess, sys, random, numpy as np, torch
IN_COLAB = os.path.exists("/var/colab/hostname")
LAB2 = "/content/drive/Othercomputers/My Mac/data266-9067/Lab2"   # Drive-synced copy of this repo's Lab2/
AUTO_DISCONNECT = True   # release the runtime when the run ends or any cell errors (Colab only)
if IN_COLAB:
    from google.colab import drive, runtime
    drive.mount("/content/drive")
    os.chdir(f"{LAB2}/notebooks")
    def _disconnect_on_error(shell, etype, evalue, tb, tb_offset=None):
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        if AUTO_DISCONNECT:
            print("Stopped - disconnecting the runtime."); runtime.unassign()
    get_ipython().set_custom_exc((Exception,), _disconnect_on_error)
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
text=True).stdout.strip()
print({"version": NOTEBOOK_VERSION, "commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})

# %%
# Settings and the adapters to upload
import json, hashlib
from pathlib import Path
from huggingface_hub import HfApi
OUT_DIR = Path("../outputs/p2")
RUNS_ROOT = Path("/content/drive/MyDrive/DATA266_Lab2/p2/runs") if IN_COLAB else Path("../outputs/p2_drive/runs")
STEPS = 5000                     # which training budget to upload (1500 = the v3 runs, kept on Drive)
RUN_NAMES = [f"run1_lora_r4_s{STEPS}", f"run2_lora_r32_s{STEPS}"] if STEPS != 1500 else ["run1_lora_r4", "run2_lora_r32"]
PRIVATE = True
REPO_PREFIX = "data266-lab2-naruto-sd15"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

runs = {}
for name in RUN_NAMES:
    d = RUNS_ROOT / name
    weights, meta_file = d / "pytorch_lora_weights.safetensors", d / "run_meta.json"
    assert weights.exists() and meta_file.exists(), f"{name}: adapter or run_meta.json missing in {d}"
    meta = json.loads(meta_file.read_text())
    assert meta.get("finished"), f"{name}: run_meta.json says the run did not finish"
    runs[name] = {"dir": d, "meta": meta, "weights_sha256": sha256(weights),
                  "weights_mb": round(weights.stat().st_size / 2**20, 2)}
    print(name, {"rank": meta["rank"], "steps": meta["steps"], "trainable_params": meta["trainable_params"],
                 "weights_mb": runs[name]["weights_mb"], "weights_sha256": runs[name]["weights_sha256"][:16]})

# %%
# Hugging Face login from the Colab secret
token = os.environ.get("HF_TOKEN")
if IN_COLAB and not token:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")      # raises if the secret is missing or notebook access is off
api = HfApi(token=token)
user = api.whoami()["name"]
print("Logged in to Hugging Face as", user)

# %%
# Model card for each repo (training facts only; results are added to the final run's card after evaluation)
CARD = """---
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
library_name: diffusers
license: other
license_name: research-only-non-commercial
datasets:
- lambdalabs/naruto-blip-captions
tags:
- stable-diffusion
- lora
- text-to-image
---
# {repo} — LoRA rank {rank} on Stable Diffusion 1.5 (Naruto BLIP captions)

DATA 266 Lab 2, Part 2. LoRA adapter for the SD 1.5 UNet attention layers, trained with the diffusers example
script `train_text_to_image_lora.py` (diffusers {diffusers}, peft {peft}).

| Setting | Value |
|---|---|
| Base model | stable-diffusion-v1-5/stable-diffusion-v1-5 |
| Data | lambdalabs/naruto-blip-captions (revision `{rev}`), {n_train} training images after a 90/10 split with seed 266 |
| Resolution | 512 × 512, centre crop, random flip |
| LoRA rank | {rank} ({params:,} trainable parameters) |
| Steps / batch | {steps} optimizer steps, batch {batch} × {accum} gradient accumulation |
| Learning rate | {lr}, constant |
| Precision / seed | fp16 mixed precision, seed 266 |
| Hardware / time | {gpu}, {minutes} min |

## Use
```python
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("{repo}")
image = pipe("a man with black hair and a hoodie", num_inference_steps=30, guidance_scale=7.5).images[0]
```

## License and limits
The training images show copyrighted Naruto characters, so this adapter is for **non-commercial research and
teaching only**. The captions are short BLIP captions that never name the characters or the style, and most
describe a man; generations inherit these biases.

Lineage: prompts SHA-256 `{prompts_sha}`, split SHA-256 `{split_sha}`, adapter SHA-256 `{w_sha}`.
"""

# %%
# Upload: adapter + run_meta.json + model card, one repo per run
links = {}
for name, r in runs.items():
    m = r["meta"]; repo = f"{user}/{REPO_PREFIX}-{name.replace('_', '-')}"   # same repo names p2_train.ipynb pushes to
    api.create_repo(repo, private=PRIVATE, exist_ok=True)
    card = CARD.format(repo=repo, rank=m["rank"], diffusers=m["diffusers"], peft=m["peft"], rev=m["dataset_revision"][:12],
                       n_train=m["train_images"], params=m["trainable_params"], steps=m["steps"], batch=m["batch_size"],
                       accum=m["grad_accum"], lr=m["learning_rate"], gpu=m["gpu"], minutes=round(m["train_seconds"] / 60, 1),
                       prompts_sha=m["prompts_sha256"][:16], split_sha=m["split_sha256"][:16], w_sha=r["weights_sha256"][:16])
    (r["dir"] / "README.md").write_text(card)
    info = api.upload_folder(repo_id=repo, folder_path=str(r["dir"]),
                             allow_patterns=["pytorch_lora_weights.safetensors", "run_meta.json", "README.md"],
                             commit_message=f"{name}: LoRA rank {m['rank']}, {m['steps']} steps")
    links[name] = {"url": f"https://huggingface.co/{repo}", "commit": info.oid, "private": PRIVATE,
                   "weights_sha256": r["weights_sha256"]}
    print(name, "->", links[name]["url"], "commit", info.oid[:10])

# %%
# Check: download each adapter back and compare its hash with the file on Drive
from huggingface_hub import hf_hub_download
for name, l in links.items():
    repo = l["url"].removeprefix("https://huggingface.co/")
    got = hf_hub_download(repo, "pytorch_lora_weights.safetensors", revision=l["commit"], token=token)
    assert sha256(got) == l["weights_sha256"], f"{name}: downloaded adapter differs from the Drive copy"
    print(name, "download matches the Drive copy")
(OUT_DIR / f"hf_links_s{STEPS}.json").write_text(json.dumps(links, indent=1))
print("Wrote", OUT_DIR / f"hf_links_s{STEPS}.json")

# %%
# Run finished - release the runtime (Colab)
if IN_COLAB and AUTO_DISCONNECT:
    print("Run complete - disconnecting.")
    runtime.unassign()
