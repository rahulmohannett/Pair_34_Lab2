---
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
# Centhur/data266-lab2-naruto-sd15-run1-lora-r4-s5000 — LoRA rank 4 on Stable Diffusion 1.5 (Naruto BLIP captions)

DATA 266 Lab 2, Part 2. LoRA adapter for the SD 1.5 UNet attention layers, trained with the diffusers example
script `train_text_to_image_lora.py` (diffusers 0.42.1, peft 0.21.2).

| Setting | Value |
|---|---|
| Base model | stable-diffusion-v1-5/stable-diffusion-v1-5 |
| Data | lambdalabs/naruto-blip-captions (revision `1ed13e8ef280`), 1099 training images after a 90/10 split with seed 266 |
| Resolution | 512 × 512, centre crop, random flip |
| LoRA rank | 4 (797,184 trainable parameters) |
| Steps / batch | 5000 optimizer steps, batch 1 × 4 gradient accumulation |
| Learning rate | 0.0001, constant |
| Precision / seed | fp16 mixed precision, seed 266 |
| Hardware / time | NVIDIA GeForce RTX 5090, 44.1 min |

## Use
```python
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("Centhur/data266-lab2-naruto-sd15-run1-lora-r4-s5000")
image = pipe("a man with black hair and a hoodie", num_inference_steps=30, guidance_scale=7.5).images[0]
```

## License and limits
The training images show copyrighted Naruto characters, so this adapter is for **non-commercial research and
teaching only**. The captions are short BLIP captions that never name the characters or the style, and most
describe a man; generations inherit these biases.

Lineage: prompts SHA-256 `5b587498de357aea`, split SHA-256 `62516f7de0fc9b6c`, adapter SHA-256 `f29f16f0e3b3245f`.
