---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
language:
  - en
  - zh
tags:
  - glm
  - glm-5
  - glm5_next
  - mlx
  - apple-silicon
  - quantized
  - 2-8bit
  - moe
  - vision-language
  - image-text-to-text
  - multimodal
  - orcasaq
  - dynamic-quant
  - reasoning
---

<div align="center">

<a href="https://www.orcarouter.ai" target="_blank">
  <img src="https://www.orcarouter.ai/orca-logo-classic.png" alt="OrcaRouter" width="110">
</a>

<h1>GLM-5.3-Flash-MLX</h1>

<p><em>An MLX build of the official GLM-5.3-Flash — 2bit-lite / 2 / 3 / 4 / 6-bit OrcaSAQ quant for Apple Silicon &amp; MLX</em></p>


<!-- orcacyber-promo:start -->
<table>
<tr>
<td width="150" style="vertical-align: middle;">
<div align="center"><a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.5"><img src="https://www.orcarouter.ai/orca-logo-classic.png" width="90" alt="OrcaCyber Zero 1.5"></a></div>
</td>
<td>
<p>🛡️ <strong>New from Orca &mdash; <a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.5">OrcaCyber Zero 1.5</a></strong></p>
<p>Post-trained for cybersecurity: vulnerability analysis, vulnerability reproduction, red teaming &amp; offensive-security research. <strong>98.01% pass@1 on CyberGym Level&nbsp;1</strong> &middot; 1M context &middot; function calling &middot; OpenAI-compatible API.</p>
<p><a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.5"><img src="https://img.shields.io/badge/Try%20it%20on%20OrcaRouter%20%E2%86%92-FF6A00?style=for-the-badge&amp;labelColor=0B1F3A" alt="Try it on OrcaRouter"></a></p>
</td>
</tr>
<tr>
<td colspan="2">
<p>🔌 <strong>New from Orca &mdash; <a href="https://www.orcarouter.ai/solutions/orca-connect">Orca Connect</a></strong></p>
<p>Have a rig running Ollama, LM Studio, vLLM, llama.cpp or SGLang? <strong>Orca Connect</strong> puts it behind your usual OrcaRouter API key, next to 200+ hosted models, reachable from a laptop, a phone or a cloud agent. Outbound only: no port forwarding, no tunnel, no firewall rule. <strong>Requests your own hardware serves cost $0</strong>, and <code>orca-connect claude</code> points Claude Code at your rig for one session.</p>
<video src="https://huggingface.co/spaces/orcarouter/README/resolve/main/assets/orca-connect.mp4" controls autoplay muted loop playsinline width="100%"></video>
<p><a href="https://www.orcarouter.ai/solutions/orca-connect"><img src="https://img.shields.io/badge/Connect%20your%20rig%20%E2%86%92-FF6A00?style=for-the-badge&amp;labelColor=0B1F3A" alt="Connect your rig"></a></p>
</td>
</tr>
</table>
<!-- orcacyber-promo:end -->

<p><strong>▶ <a href="https://www.orcarouter.ai/models/z-ai/glm-5.3-flash">Run GLM-5.3-Flash instantly on the OrcaRouter API</a></strong> — no download, full precision.</p>

<p>
<a href="https://www.orcarouter.ai"><img src="https://img.shields.io/badge/Website-orcarouter.ai-1E6FEB" alt="Website"></a>
<a href="https://www.orcarouter.ai/models"><img src="https://img.shields.io/badge/OrcaRouter-Model%20Catalog-2EA043" alt="Model Catalog"></a>
<a href="https://www.orcarouter.ai/models/z-ai/glm-5.3-flash"><img src="https://img.shields.io/badge/OrcaRouter-Hosted%20API-6F42C1" alt="Hosted API"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-4C8BF5" alt="License"></a>
<img src="https://img.shields.io/badge/Format-MLX-00A67E" alt="MLX">
<img src="https://img.shields.io/badge/Quants-lite%20|%202%20|%203%20|%204%20|%206--bit-FF8800" alt="Quants">
<img src="https://img.shields.io/badge/Method-OrcaSAQ-E67E22" alt="OrcaSAQ">
<img src="https://img.shields.io/badge/Vision%20%2B%20indexer-BF16-9B59B6" alt="BF16">
<a href="https://github.com/Continuum-AI-Corp/Orca-Code-Review"><img src="https://img.shields.io/badge/OrcaCode-Review-181717?logo=github" alt="OrcaCode Review"></a>
</p>

<p><strong>One Gateway. Every Model.</strong> — Route Smarter · Ship Safer · Spend Less.<br>
<a href="https://www.orcarouter.ai/models/z-ai/glm-5.3-flash">GLM-5.3-Flash on OrcaRouter →</a></p>

<p>
<a href="https://www.orcarouter.ai">Website</a> ·
<a href="https://www.orcarouter.ai/models">Model Catalog</a> ·
<a href="https://www.orcarouter.ai/models/z-ai/glm-5.3-flash">Model API</a> ·
<a href="https://github.com/Continuum-AI-Corp">GitHub</a> ·
<a href="https://github.com/Continuum-AI-Corp/Orca-Code-Review">OrcaCode Review</a> ·
<a href="https://discord.gg/yAh6Tex6kx">Discord</a> ·
<a href="https://x.com/OrcaRouter">X</a>
</p>

</div>

---

> An **MLX** build of the official [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
> — a **320B-total / 18B-active** MoE, the first natively multimodal model in the GLM-5 series, built on
> the new **`glm5_next`** hybrid architecture (sparse + linear attention with Manifold-Constrained
> Hyper-Connections) — quantized to **MLX** for **Apple Silicon** and the MLX CUDA backend. Five
> builds (**2bit-lite / 2 / 3 / 4 / 6-bit**) ship as subfolders, with the **4-bit build mirrored at the repo root**,
> produced with **OrcaSAQ (Sensitivity-Aware Quantization)** — calibration-free and
> architecture-aware: sensitive tensors get more bits
> (shared experts `+2`, `down_proj` `+1`), and every tensor that was *not* FP8 in the base release —
> **the 34 linear-attention layers, the sparse indexer, hyper-connections, norms, `embed_tokens`,
> `lm_head`, and the entire vision tower — is carried through in BF16**. The four projections of the
> sparse-attention blocks *were* FP8 upstream and are quantized at the base width. Browse all models in the
> [OrcaRouter Model Catalog](https://www.orcarouter.ai/models); deployed as API
> [here](https://www.orcarouter.ai/models/z-ai/glm-5.3-flash). Put this model to work reviewing your
> pull requests with **[OrcaCode Review](https://github.com/Continuum-AI-Corp/Orca-Code-Review)**.

---

## Available quantizations

| Folder | Base bits | Group size | Size | Min RAM | Quality vs FP8 |
|---|---|---|---:|---|---|
| `6-bit/` | 6 | 64 | ~296 GB | 320 GB | **Near-lossless** — best quality |
| `4-bit/` | 4 | 64 | ~204 GB | 224 GB | Very good — **recommended default** |
| `3-bit/` | 3 | 32 | ~184 GB | 200 GB | Good — aggressive but usable |
| `2-bit/` | 2 | 32 | ~145 GB | 160 GB | Aggressive — best-effort |
| `2bit-lite/` | 2 | 128 | ~102 GB | 112 GB | Smallest — fits 128 GB Macs & a single H200 |

The **repository root is a mirror of `4-bit/`**, so `orcarouter/GLM-5.3-Flash-MLX` resolves to the
recommended default with no subfolder needed.

> Quantized from the official **FP8** release. Bit-widths above are the **base** precision; the OrcaSAQ
> policy below raises sensitive tensors above the base. Group size is 64 for 4/6-bit, 32 for 2/3-bit
> (shared experts always use 64), and 128 throughout `2bit-lite`.

No 8-bit build was produced. `2bit-lite` follows a separate, memory-first recipe — see below.

---

## OrcaSAQ — Sensitivity-Aware Quantization

OrcaSAQ is a **calibration-free, architecture-aware mixed-precision quantization method** designed to
preserve model quality while aggressively reducing memory footprint.

Instead of running expensive calibration datasets or per-layer sensitivity searches, OrcaSAQ uses
**architectural and tensor-role priors** to determine which weights are most sensitive to quantization.
Critical components receive higher precision, while more quantization-tolerant tensors remain at the
target base precision.

For MoE architectures, for example, OrcaSAQ can preserve shared experts at **base +2 bits**, `down_proj`
at **base +1 bit**, keep `gate_proj` and `up_proj` at the base precision, and retain sensitive
non-quantized components in BF16.  
The result is a **fast, deterministic, zero-calibration quantization pipeline** that can be applied to
new architectures without costly dataset-driven optimization.

**Architecture-aware. Calibration-free. Quality-preserving.**

### Bit allocation for GLM-5.3-Flash

| Component | Count | 2-bit | 3-bit | 4-bit | 6-bit | Policy |
|---|---:|---|---|---|---|---|
| Expert `gate_proj` / `up_proj` | 24 768 | 2 | 3 | 4 | 6 | base |
| Expert `down_proj` | 12 384 | 3 | 4 | 5 | 8 | base **+1** |
| Dense-MLP `gate_proj` / `up_proj` (layers 0–2) | 6 | 2 | 3 | 4 | 6 | base |
| Dense-MLP `down_proj` (layers 0–2) | 3 | 3 | 4 | 5 | 8 | base **+1** |
| Shared expert `gate_proj` / `up_proj` / `down_proj` | 129 | 4 | 5 | 6 | 8 | base **+2** |
| Sparse-attn `q_a_proj` / `q_b_proj` / `kv_a_proj_with_mqa` / `o_proj` | 48 | 2 | 3 | 4 | 6 | base |
| Linear-attn layers, sparse indexer, mHC, norms, `embed_tokens`, `lm_head`, vision tower | — | BF16 | BF16 | BF16 | BF16 | never quantized |

Bits are rounded up to the nearest MLX-supported width (`{2,3,4,5,6,8}`). The architectural prior that
selects the quantizable set is mechanical and needs no data: a tensor is re-quantized **iff the FP8
release shipped it with a `_scale_inv` companion**. For GLM-5.3-Flash that set is the MoE and dense-MLP
linears *plus* the four projections of every `deepseek_sparse_attention` block — the 11 sparse layers at
depth 3, 7, 11 … 43, and the MTP block, hence 12 × 4 = 48 tensors. The 34 `linear_attention` layers, the
sparse indexer, and the vision tower were never FP8 and stay BF16.

37 338 tensors are quantized in total. The assignment is recorded in `config.json` →
`quantization`, as **per-module `{group_size, bits}` overrides keyed by MLX module path** —
e.g. `model.layers.3.mlp.switch_mlp.down_proj`. MLX fuses a layer's routed experts into one
`switch_mlp`, so 173 entries cover all 37 338 tensors. This is what the loader reads at load time and
it is **required**: the top-level `bits`/`group_size` alone would mis-shape every raised-precision
tensor.

The **MTP (multi-token prediction) layer — layer 45 — is included inside the quantized weights** rather
than exported as a separate `mtp/` module.

---

## Quality vs FP8

All three tables compare each build against the full **FP8** reference (dequantized to BF16 and run
through the identical `glm5_next` forward, so the only variable is the quantization). Sizes are decimal
GB, matching the file sizes in this repo.

**Perplexity**

| Build | Size | PPL | ΔPPL |
|---|---:|---:|---:|
| **FP8 (ref)** | 328 GB | 2.7797 | — |
| **6-bit** | 296 GB | 2.7864 | +0.24 % |
| **4-bit** | 204 GB | 2.8620 | +2.96 % |
| **3-bit** | 184 GB | 3.0566 | +9.96 % |
| **2-bit** | 145 GB | 4.3622 | +56.9 % |
| **2bit-lite** | 102 GB | 6.7018 | +141 % |

**KL divergence &amp; Top-1 token agreement** — lower KLD and higher Top-1 = closer to the original.

| Build | Size | Mean KLD | KLD p95 | Top-1 agreement |
|---|---:|---:|---:|---:|
| **FP8 (ref)** | 328 GB | — | — | 100 % |
| **6-bit** | 296 GB | 0.0063 | 0.0142 | 97.76 % |
| **4-bit** | 204 GB | 0.0131 | 0.0477 | 96.13 % |
| **3-bit** | 184 GB | 0.0421 | 0.1332 | 92.06 % |
| **2-bit** | 145 GB | 0.1647 | 0.6528 | 86.56 % |
| **2bit-lite** | 102 GB | 0.3456 | 1.2617 | 77.19 % |

**Weight-space fidelity** (mean over sampled MoE/MLP tensors, quantized vs FP8):

| Build | Size | Cosine sim | SNR (dB) | Rel. error |
|---|---:|---:|---:|---:|
| **6-bit** | 296 GB | 0.9998 | 37.3 | 1.6 % |
| **4-bit** | 204 GB | 0.9969 | 22.9 | 7.6 % |
| **3-bit** | 184 GB | 0.9892 | 17.6 | 14.0 % |
| **2-bit** | 145 GB | 0.9518 | 11.1 | 29.7 % |
| **2bit-lite** | 102 GB | 0.9135 | 7.5 | 42.4 % |

The three measurements agree on the ordering, and they agree on where the cliff is: everything down to
`3-bit` degrades gently, `2-bit` costs a lot, and `2bit-lite` costs a lot more. Pick it for fit, not
for quality.

---

## `2bit-lite/` — smallest build, for 128 GB Macs and a single H200

A memory-first build that trades quality for the smallest footprint that still loads and runs on
`mlx-vlm`. It exists to fit machines where the regular 2-bit build (145 GB) is just too big:

- **128 GB MacBook Pro (M4 / M5 Max)** — regular 2-bit does not fit; this does, with a raised
  wired-memory limit.
- **A single H200 (141 GB)** — regular 2-bit overflows one card; `2bit-lite` fits with ~39 GB left for
  the KV cache.

**Recipe** (group_size 128 throughout):

| Component | Bits |
|---|---|
| Routed experts (`gate_proj` / `up_proj` / `down_proj`) | 2 |
| Attention MLA + dense MLP | 2 |
| Shared expert (fires on every token) | 3 |
| `embed_tokens` / `lm_head` / linear-attention path | BF16 |

**Size: ~102 GB** (≈2.25 bpw on the experts). That is the floor for this architecture: MLX's affine
quantizer cannot go below 2 bits, and the fused MoE experts — ~97 % of the weights — cannot use the
1-bit path.

### Field notes — what `2bit-lite` is actually good for

Verified on a single **H200**: it deploys cleanly, multi-turn conversation is stable at **~10 tok/s**,
and everyday Q&amp;A and short-form text come out fine.

**Long code generation is not reliable at this precision.** Three failure modes reproduce consistently:

- **Repetition loops** — the model gets stuck emitting the same block over and over.
- **Missing glue code** — the overall structure is right, but load-bearing lines (imports, wiring,
  error handling) are silently dropped.
- **Rewrite churn** — it keeps restarting the answer and never commits to a final version.

So: reach for `2bit-lite` when the hardware cannot hold anything larger, and keep it to chat,
Q&amp;A and short generations. For coding and long-horizon agentic work use [`2-bit`](./2-bit) or
higher — the quality tables above show why the gap shows up in practice, and the API option needs no
local memory at all.

---

## Usage

### Hosted API — no download

The full-precision model is live on OrcaRouter as **`z-ai/glm-5.3-flash`**, so you can call it
without pulling 100–300 GB of weights — endpoint, keys and pricing are on the model page:

**<https://www.orcarouter.ai/models/z-ai/glm-5.3-flash>**

### Run it locally (mlx-vlm)

GLM-5.3-Flash is a **vision-language** model, so it runs under **`mlx-vlm`** — not `mlx-lm`.

```bash
pip install -U "mlx-vlm>=0.6.17"
```

The repo root is the 4-bit build, so the recommended default needs no subfolder — the excludes are
what stop `hf download` from also pulling all five variant folders:

```bash
hf download orcarouter/GLM-5.3-Flash-MLX --local-dir ./GLM-5.3-Flash-MLX \
    --exclude "2bit-lite/*" "2-bit/*" "3-bit/*" "4-bit/*" "6-bit/*"

python -m mlx_vlm.generate --model ./GLM-5.3-Flash-MLX \
    --prompt "Explain quantum entanglement in one sentence." --max-tokens 256
```

To pick a different precision, download just that subfolder and load it from the local path:

```bash
hf download orcarouter/GLM-5.3-Flash-MLX --include "6-bit/*" --local-dir ./GLM-5.3-Flash-MLX

python -m mlx_vlm.generate --model ./GLM-5.3-Flash-MLX/6-bit \
    --prompt "Explain quantum entanglement in one sentence." --max-tokens 256
```

With an image or a video frame:

```bash
python -m mlx_vlm.generate --model ./GLM-5.3-Flash-MLX \
    --image photo.jpg --prompt "Describe this image." --max-tokens 256
```

Python API:

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("./GLM-5.3-Flash-MLX")          # repo root == 4-bit
prompt = apply_chat_template(processor, model.config, "Describe this image.", num_images=1)

print(generate(model, processor, prompt, ["photo.jpg"], max_tokens=256, verbose=True))
```

> **`glm5_next` runtime required.** This is a brand-new architecture (hybrid sparse + linear attention,
> mHC, MoE). Use an `mlx-vlm` build that has landed `glm5_next` support. On Apple Silicon the Metal
> backend is used automatically. On the Linux **CUDA** backend install `mlx[cuda]` and set
> `MLX_CUDA_USE_CUDNN_SDPA=0`.

---

## 🐳 Run it with OrcaCode Review

<a href="https://www.orcarouter.ai"><img src="https://img.shields.io/badge/Website-orcarouter.ai-1E6FEB" alt="OrcaRouter website"></a>
<a href="https://github.com/Continuum-AI-Corp/Orca-Code-Review"><img src="https://img.shields.io/badge/GitHub-Orca--Code--Review-181717?logo=github" alt="Orca Code Review on GitHub"></a>

**Models are only half the system.**

[**OrcaCode Review**](https://github.com/Continuum-AI-Corp/Orca-Code-Review) turns every model listed on
[**OrcaRouter**](https://www.orcarouter.ai) into a production code-review agent:

- reviews every PR
- finds security + correctness issues
- posts inline findings
- P0/P1 can block merges
- swap models anytime

**Open model. Open harness. Open bill.**

- Website — <https://www.orcarouter.ai>
- GitHub — <https://github.com/Continuum-AI-Corp/Orca-Code-Review>

---

# GLM-5.3-Flash

> Available full-precision on the OrcaRouter API as **`z-ai/glm-5.3-flash`** —
> **<https://www.orcarouter.ai/models/z-ai/glm-5.3-flash>**

GLM-5.3-Flash is the first natively multimodal model in the GLM-5 series. With **320B total parameters
and just 18B active parameters**, it outperforms GLM-5.2 across benchmarks and real-world workloads at
one-tenth the price, while approaching Claude Opus 4.8 on coding and agentic benchmarks.

For the first time in the GLM series, GLM-5.3-Flash introduces a **hybrid architecture combining sparse
and linear attention**, sharply reducing long-context serving costs while preserving precise long-context
capabilities. It also adopts **Manifold-Constrained Hyper-Connections (mHC)** to improve scaling
efficiency, and is trained on a 30T-token multimodal corpus.

## Model Overview

- **Type:** Mixture-of-Experts multimodal Causal LM (`glm5_next`) with vision encoder
- **Parameters:** 320B total · **18B active** per token
- **Experts:** 288 routed + 1 shared, top-8 routing
- **Layers:** 45 (3 dense + 42 MoE) · 11 `deepseek_sparse_attention` + 34 `linear_attention`, plus 1 MTP
  layer (also MoE, with its own sparse attention)
- **Attention:** hybrid **sparse + linear** attention with a learned sparse **indexer**, plus
  Manifold-Constrained **Hyper-Connections (mHC)**
- **Precision (base release):** FP8 (block-wise `e4m3`, 128×128), dynamic activation scheme
- **Modalities:** text, image, video

## Best Practices

- **Sampling:** follow the official GLM-5.3-Flash guidance; long-horizon agentic and coding tasks
  benefit from generous `max-tokens` headroom.
- **Long context:** the hybrid sparse/linear attention is designed for low-cost long-context serving;
  give the runtime enough KV budget for your target length.
- **Pick a precision:** `6-bit` for near-lossless, `4-bit` (repo root) as the everyday default, `3-bit`
  and `2-bit` when memory is the binding constraint, `2bit-lite` only when nothing else fits
  (see the quality tables above — the drop from `2-bit` to `2bit-lite` is steep).
- **Or skip the download:** the unquantized model is served at full precision on the
  [OrcaRouter API](https://www.orcarouter.ai/models/z-ai/glm-5.3-flash) as `z-ai/glm-5.3-flash`.

---

## Citation

```bibtex
@misc{glm5team2026glm5vibecodingagentic,
      title={GLM-5: from Vibe Coding to Agentic Engineering},
      author={GLM-5-Team and Aohan Zeng and Xin Lv and Zhenyu Hou and Zhengxiao Du and others},
      year={2026},
      eprint={2602.15763},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.15763},
}
```

## License

**MIT**, inherited from [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash).
Quantization does not change the underlying license obligations.
