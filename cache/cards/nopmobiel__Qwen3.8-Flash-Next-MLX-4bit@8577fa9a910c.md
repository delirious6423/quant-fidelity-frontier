---
library_name: mlx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
tags:
  - mlx
  - vision-language
  - qwen4-exp
---

# Qwen3.8-Flash-Next MLX 4-bit

Local MLX conversion of `Qwen/Qwen3.8-Flash-Next`.

- Language weights: affine 4-bit, group size 64.
- PLE n-gram embedding: affine 4-bit, group size 32.
- MoE routing gates: affine 8-bit, group size 64.
- Vision tower: BF16.
- Native MTP tensors are not included in this base checkpoint.

This model requires the Qwen4-Exp support in the accompanying local
`mlx-vlm` checkout.

## Runtime setup

Qwen4-Exp is newer than the current stock `mlx-vlm` model registry. Until the
implementation is upstreamed, use the runtime module bundled in this repo:

```bash
git clone https://github.com/Blaizzy/mlx-vlm.git
cp -R mlx_vlm_qwen4_exp/qwen4_exp mlx-vlm/mlx_vlm/models/qwen4_exp

python3 -m venv .venv
.venv/bin/pip install -e ./mlx-vlm
.venv/bin/pip install --upgrade git+https://github.com/huggingface/transformers.git

.venv/bin/mlx_vlm.generate \
  --model . \
  --prompt "Hello!" \
  --max-tokens 256
```

The bundled module registers the Qwen3-VL-compatible processor and multimodal
chat format automatically. Native MTP speculative decoding is not included.

For an interactive terminal chat after setup:

```bash
.venv/bin/mlx_vlm.chat \
  --model . \
  --max-tokens 512 \
  --temperature 0.0
```

Commands inside the chat include `/image <path>`, `/clear`, and `/exit`.

## LM Studio (experimental)

LM Studio's current MLX runtime does not yet bundle `qwen4_exp`. From this
model directory, install the included compatibility module:

```bash
chmod +x install_lmstudio_patch.sh
./install_lmstudio_patch.sh
```

Then unload and reload the model in LM Studio. On a 128 GB Mac, LM Studio may
still reject the checkpoint based on its conservative memory estimate. Enable
Developer Mode and select **Settings → Hardware → Model loading guardrails →
OFF** only after closing other memory-heavy applications. The measured peak is
about 104.1 GB, so smaller-memory Macs are not recommended.

Verified with LM Studio 0.4.21 (Build 2), MLX runtime 1.11.0, including
multi-turn text chat and image understanding. The compatibility module also
supplies the sparse-attention path needed by larger image prompts. Updating LM
Studio's MLX runtime may replace the installed module; rerun the script if
`qwen4_exp` becomes unrecognized again or vision generation fails after an
update.

## Verified locally

Tested on a 128 GB Apple M5 Max:

- Text: 9.13 prompt tokens/s, 60.95 generation tokens/s.
- Image + text: 214.84 prompt tokens/s, 48.92 generation tokens/s.
- Peak unified memory: 104.07 GB.
- LM Studio multi-turn text chat: 36.63 generation tokens/s.
- LM Studio image understanding: 31.01 generation tokens/s.
