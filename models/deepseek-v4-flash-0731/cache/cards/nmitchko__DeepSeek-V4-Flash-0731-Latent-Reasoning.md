---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
  - latent-reasoning
  - reasoning
  - deepseek-v4
  - nvfp4
  - fp4
  - speculative-decoding
revision: all-nvfp4-rider
---

# DeepSeek-V4-Flash-0731-Latent-Reasoning

[![Model](https://img.shields.io/badge/Model-NVFP4%20%2F%20Latent%20Reasoning-2b7fff)][model]
[![vllm fork](https://img.shields.io/badge/vllm%20fork-ds4%20branch-2b7fff)][fork]
[![Addon](https://img.shields.io/badge/Addon-reasoning%20head-2b7fff)][addon]
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)][license]

[model]: https://huggingface.co/nmitchko/DeepSeek-V4-Flash-0731-Latent-Reasoning
[fork]: https://github.com/nickmitchko/vllm-ds4-sm120
[addon]: https://github.com/nickmitchko/ds4-reasoning-addon
[license]: #license

**A complete, self-contained model. The DeepSeek-V4-Flash-0731 backbone is
quantized to all-NVFP4. A trained latent reasoning head is shipped together with
it.**

This is not an adapter. Everything that is needed to serve the model is in this
repository. That includes the full 43-layer backbone, the DSpark
speculative-decoding draft block, the tokenizer, and the latent reasoning head.
The weights are a quantization of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

The model reasons in a **compressed latent space**. It does not emit a long
token-by-token chain of thought. A small head reads the backbone's layer-35
hidden state. It projects that state into a 1024-d latent. It decodes the latent
back into the residual stream. One latent step stands in for several reasoning
tokens. A learned stop head decides when reasoning is complete. The latent phase
then self-terminates at a variable depth. The depth depends on the content. It
does not run a fixed number of steps.

---

## Benchmark

**BBH (BIG-Bench Hard), `cot_zeroshot`, 27 subtasks — aggregate 0.880 ± 0.008**

We measured this with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
0.4.12 against an OpenAI-compatible endpoint. Thinking was enabled. We used 50
items per subtask (1350 items total). The metric is `exact_match` /
`flexible-extract`.

| Subtask | Score | Subtask | Score |
|---|---|---|---|
| tracking_shuffled_objects_three_objects | 1.00 | date_understanding | 0.92 |
| tracking_shuffled_objects_five_objects | 1.00 | sports_understanding | 0.88 |
| tracking_shuffled_objects_seven_objects | 1.00 | logical_deduction_five_objects | 0.88 |
| penguins_in_a_table | 1.00 | web_of_lies | 0.86 |
| formal_fallacies | 1.00 | snarks | 0.84 |
| boolean_expressions | 1.00 | ruin_names | 0.84 |
| word_sorting | 0.98 | movie_recommendation | 0.84 |
| temporal_sequences | 0.98 | salient_translation_error_detection | 0.76 |
| object_counting | 0.98 | geometric_shapes | 0.74 |
| navigate | 0.98 | causal_judgement | 0.66 |
| logical_deduction_three_objects | 0.98 | disambiguation_qa | 0.58 |
| reasoning_about_colored_objects | 0.96 | dyck_languages | 0.26 |
| hyperbaton | 0.96 | | |
| multistep_arithmetic_two | 0.94 | | |
| logical_deduction_seven_objects | 0.94 | | |

The model is strongest on multi-step state tracking and logical deduction. It is
weakest on `dyck_languages` (bracket matching). That subtask is the clear
outlier.

> **Read the `flexible-extract` number, not `strict-match`.** The `strict-match`
> filter searches for the literal phrase `The answer is X`. This model does not
> emit that phrase. Its near-zero `strict-match` score is an answer-formatting
> artifact. It is not a measure of reasoning ability. The raw numbers are in
> [`bench/bbh_cot_zeroshot.json`](./bench/bbh_cot_zeroshot.json).

The scores use 50 items per subtask. Each per-subtask value carries about
±0.05–0.07. The aggregate is the reliable figure.

---

## Quantization

| | |
|---|---|
| source | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| scheme | all-NVFP4 (group size 16) |
| draft block | 3-layer DSpark, preserved from source |
| weights | 48 shards, `bfloat16` container |

The routed expert projections in all 43 layers are converted to NVFP4. The
DSpark draft block experts are also NVFP4. Attention projections, shared
experts, the LM head, and the draft block's three layers are kept at higher
precision. NVFP4 needs a Blackwell-class GPU (compute capability 12.0 / sm120)
for native kernel support.

---

## Latent reasoning

The latent reasoning loop is:

```
    layer 35 hidden (4096-d)
                 |
                 v  LayerNorm
    +--------- ReasoningCompressionHead ----------+
    |  Linear 4096 -> 2048 . SiLU                 |
    |  Linear 2048 -> 2048 . SiLU                 |
    |  Linear 2048 -> 2048  ->  [mu, log_sigma]   |
    |                                             |
    |  stop_head:                                 |
    |     Linear 4096 -> 1024 . SiLU              |
    |     Linear 1024 -> 1                        |  -> end of reasoning
    +---------------------------------------------+
                 | mu (1024-d latent)
                 v  LayerNorm
    +-------------- LatentDecoder ----------------+
    |  Linear 1024 -> 2048 . SiLU                 |
    |  Linear 2048 -> 2048 . SiLU                 |
    |  Linear 2048 -> 4096                        |
    +---------------------------------------------+
                 |
                 v  written back into the residual stream
      DeepSeek-V4-Flash-0731 backbone (frozen, NVFP4)
```

| Config | Value |
|---|---|
| hidden_size | 4096 |
| latent_dim | 1024 |
| mlp_dim | 2048 |
| source_layer / target_layer | 35 / 42 |
| activation | SiLU |
| learned stop head | yes |
| head + decoder params | 35.7M (`float32`) |
| backbone layers | 43 |

The head is `latent_reasoning_head.safetensors` (~152 MB). It is a single flat
tensor dict. Its submodules are distinguished by key prefix:

```
reasoning_head.net.0.weight        [2048, 4096]   reasoning_head.net.0.bias        [2048]
reasoning_head.net.2.weight        [2048, 2048]   reasoning_head.net.2.bias        [2048]
reasoning_head.net.4.weight        [2048, 2048]   reasoning_head.net.4.bias        [2048]
reasoning_head.stop_head.0.weight  [1024, 4096]   reasoning_head.stop_head.0.bias  [1024]
reasoning_head.stop_head.2.weight  [1, 1024]      reasoning_head.stop_head.2.bias  [1]
decoder.net.0.weight               [2048, 1024]   decoder.net.0.bias               [2048]
decoder.net.2.weight               [2048, 2048]   decoder.net.2.bias               [2048]
decoder.net.4.weight               [4096, 2048]   decoder.net.4.bias               [4096]
target_proj.weight                 [1024, 4096]
```

The geometry is mirrored in the file's safetensors metadata and in
[`latent_reasoning_config.json`](./latent_reasoning_config.json).

`target_proj` is a frozen `Linear(4096, 1024, bias=False)`. It defined the
regression target during training. It is included for completeness. It is **not
used at inference**.

---

## Sample code

### Load the head

[`examples/load_latent_head.py`](./examples/load_latent_head.py) rebuilds the
head from the checkpoint's own metadata. It runs one latent step. It needs no
first-party imports and no serving stack.

```python
import json
import torch
import torch.nn.functional as F
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file

CKPT = "latent_reasoning_head.safetensors"


class ReasoningCompressionHead(nn.Module):
    def __init__(self, hidden_size, latent_dim, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, 2 * latent_dim),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim // 2), nn.SiLU(),
            nn.Linear(mlp_dim // 2, 1),
        )

    def forward(self, h):
        mu, log_sigma = self.net(h).chunk(2, dim=-1)
        return mu, log_sigma.clamp(-10.0, 2.0)

    def stop_logit(self, h):
        return self.stop_head(h)


class LatentDecoder(nn.Module):
    def __init__(self, hidden_size, latent_dim, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, hidden_size),
        )

    def forward(self, z):
        return self.net(z)


with safe_open(CKPT, framework="pt") as f:
    cfg = json.loads(f.metadata()["config"])
hs, ld = cfg["hidden_size"], cfg["latent_dim"]

flat = load_file(CKPT)
mlp_dim = flat["reasoning_head.net.0.weight"].shape[0]
sub = lambda p: {k[len(p):]: v for k, v in flat.items() if k.startswith(p)}

head = ReasoningCompressionHead(hs, ld, mlp_dim)
head.load_state_dict(sub("reasoning_head.")); head.eval()
decoder = LatentDecoder(hs, ld, mlp_dim)
decoder.load_state_dict(sub("decoder.")); decoder.eval()

# One latent step.
# h_src is the layer-35 hidden state at the current position, shape (B, 4096).
h_src = torch.randn(2, hs)
h_n = F.layer_norm(h_src, (hs,))
mu, _ = head(h_n)
inject = decoder(F.layer_norm(mu, (ld,)))       # (B, 4096) back into the stream
p_stop = head.stop_logit(h_n).sigmoid()         # end reasoning above threshold
```

### Query a served endpoint

[`examples/chat_openai_client.py`](./examples/chat_openai_client.py). The one
non-obvious requirement is the `thinking` flag:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="dummy")

resp = client.chat.completions.create(
    model="nmitchko/DeepSeek-V4-Flash-0731-Latent-Reasoning",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"chat_template_kwargs": {"thinking": True}},   # REQUIRED
    max_tokens=4096,
    temperature=0.6,
)
print(resp.choices[0].message.content)
```

**`chat_template_kwargs={"thinking": True}` is required.** Without it the
reasoning phase is not enabled. Answer quality drops sharply. All benchmark
numbers above used it.

---

## How to run

The backbone, tokenizer, and DSpark draft block load with a standard
NVFP4-capable inference stack on sm120 hardware. Settings that matter:

| Setting | Value | Why |
|---|---|---|
| speculative decoding | DSpark, 5 draft tokens | matches the 3-layer draft block |
| KV cache dtype | `fp8` | makes long context fit |
| tensor parallel | 2 | measured on 2x 96 GiB |
| stop threshold | 0.5 | `sigmoid(stop_logit) > 0.5` ends the latent phase |
| min / max latent steps | 4 / 256 | floor guarantees reasoning; cap bounds a stop misfire |
| output token budget | >= 4096 | reasoning and the answer share one budget |

Two behaviors are worth knowing before you judge output quality.

- **Give the answer real token headroom.** The latent reasoning phase and the
  answer share the same output-token budget. A tight `max_tokens` can be
  consumed entirely by reasoning. The answer can then be empty or truncated.
- **Warm up before trusting output.** The first request or two after a cold
  start can come back as degenerate repetition. The model then settles and
  stays correct. Send one throwaway request after startup. Treat a single bad
  early answer as unwarmed, not broken.

`compression_factor: 6` in the config records how the head was fit. It is **not
a budget enforced at inference**. The learned stop head, bounded by the min/max
latent steps, is what terminates the reasoning phase.

### Serving requirements

Upstream vllm cannot serve this model. You need the DS4 SM120 vllm fork. Serve
from the fork's **`ds4`** branch. Clone it from the public mirror:

```bash
git clone -b ds4 https://github.com/nickmitchko/vllm-ds4-sm120.git
```

The latent reasoning head lives in the reasoning addon. It is an opt-in vllm
general plugin. Clone it from the public mirror:

```bash
git clone https://github.com/nickmitchko/ds4-reasoning-addon.git
```

The fork exposes an Anthropic-compatible `/v1/messages` endpoint. The latent
reasoning loop is driven entirely by the serving runtime. This repository
contains the weights, not the custom runtime source.

---

## Limitations

- **Requires Blackwell-class hardware** (sm120) for native NVFP4 kernels.
- **Driving the latent loop requires runtime support.** The weights are
  complete. But reading layer-35 hidden states and writing decoded latents back
  into the residual stream mid-generation is not something a stock
  `transformers` forward pass does. Without that support you get the backbone.
  You do not get latent reasoning.
- **Evaluation is BBH-only** at 50 items per subtask. No multi-task or
  long-context benchmark suite is reported here.
- `dyck_languages` at 0.26 is a genuine weak spot. It is not a formatting
  artifact.
- Reasoning happens in latent space. The surfaced trace is not a faithful
  token-level record of the computation that produced the answer.

---

## License

MIT, inherited from
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
