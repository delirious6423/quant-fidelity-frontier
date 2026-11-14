---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - apple-silicon
  - metal
  - tokspur
  - qwen3.8-flash-next
---

# Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF

Quantised and planar-packed weights of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision `34567a4712bc9766c4449e2e98e4468bfa24d915`) for **Tokspur 0.3.0** on Apple Silicon Macs with 128 GB or more unified memory.

Unlike traditional split-GGUF releases that require downloading raw GPU shards and repacking an additional 80 GiB local cache on first run, [`san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF`](https://huggingface.co/san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF) ships the pre-built **Tokspur Planar Pack v3** (`Qwen3.8-Flash-Next-Q4_K.tokspur.gguf`) alongside a compact host-embedding base GGUF (`Qwen3.8-Flash-Next-base.gguf`) and the shared Q8_0 next-token speculative draft head (`MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`). `tokspur adopt` verifies and adopts these files directly in place with zero local repacking and zero duplicate GPU shard storage.

## Provenance & Quantisation Recipe

- **Upstream checkpoint:** [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/34567a4712bc9766c4449e2e98e4468bfa24d915) (`34567a4712bc9766c4449e2e98e4468bfa24d915`, 131 BF16/I64 safetensor shards). Converted directly from Qwen's originals using `tokspur-quant`; no third-party GGUF intermediates or external drafter checkpoints are included.
- **Host-resident embeddings (`Qwen3.8-Flash-Next-base.gguf`):**
  - `per_layer_token_embd.weight` (`IQ4_NL`, `28,800,138,240` B) — n-gram per-layer embedding table kept in host unified memory and gathered asynchronously.
  - `token_embd.weight` (`Q8_0`, `675,430,400` B) — input token embedding table.
  - Complete GGUF model metadata, BPE vocabulary (`248,320` tokens, `247,587` merges), n-gram hash constants, and Qwen's upstream `chat_template.jinja` (SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`).
- **GPU-resident planar pack (`Qwen3.8-Flash-Next-Q4_K.tokspur.gguf`, recipe `tokspur-pack-v3`, `1,254` tensors):**
  - Routed MoE expert projections (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`): `94` `Q4_K`, `2` `Q5_K`, and `43` `Q5_1` tensors laid out in co-located gate/up and down planar `I8` blocks for Apple Silicon Metal / NAX execution.
  - Attention, GDN linear-attention, shared-expert, router, PLE projection, and output head weights (`Q8_0`, `BF16`, `F32` — including exact `binary64 -> binary32` `-exp(a_log)` state-space decay weights).
  - Embedded manifest (`manifest_sha256`: `8fb0a3ac566b51e174e417b659e27f35ae3f36f6f6d95ebed41b5b688aa81906`) recording per-tensor packed SHA-256 digests verified before execution.
- **Speculative next-token head (`MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, `32` tensors):**
  - Layer-48 shared Multi-Token Prediction (MTP) head in `Q8_0` / `BF16` / `F32`.

## File Inventory

| File | Role | Size (bytes) | Size (GiB) | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `Qwen3.8-Flash-Next-base.gguf` | Target metadata + host embeddings | `29,486,514,464` | `27.46` | `6e27f395dc7527953671f79f03b9554c6136062dbdc6128116f82e02d3cee36a` |
| `Qwen3.8-Flash-Next-Q4_K.tokspur.gguf` | Target GPU Planar Pack v3 (`1,254` tensors) | `85,889,703,936` | `80.00` | `0967101337bbadc937e18a66b695a944bb1194f12d32bada66d6c0080fd51049` |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | Shared MTP speculative draft head (`32` tensors) | `2,786,568,256` | `2.60` | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |

Total download footprint: **`118,162,786,656` bytes (`110.05 GiB`)** — zero additional pack build space required on disk.

## Quality Evaluation

Evaluated with `llama-perplexity` (`32` chunks × `2,048` context tokens, paired logit comparison against the `UD-Q4_K_XL` reference quantisation on the same corpus):

| Metric | Value |
| --- | --- |
| Mean `PPL(Q)` (Tokspur `Q4_K`) | `4.539204 ± 0.066090` |
| Mean `PPL(base)` (reference `UD-Q4_K_XL`) | `4.564891 ± 0.066115` |
| Paired `PPL(Q) - PPL(base)` | `-0.025687 ± 0.012622` |
| Paired `ln(PPL(Q) / PPL(base))` | `-0.005643 ± 0.002774` (`98.18%` chunk correlation) |
| Mean KL divergence | `0.079720 ± 0.001575` (median `0.008645`) |
| Top-1 token agreement | `90.518% ± 0.162%` |

## Usage with Tokspur (`>= 0.3.0`)

Download the repository to a local folder (for example `~/Models/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF`), then adopt, configure, and serve in place:

```sh
hf download san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF --local-dir ~/Models/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF

tokspur adopt \
  --target ~/Models/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF \
  --draft ~/Models/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF
tokspur configure
tokspur serve --speed-preset
```

`tokspur adopt` verifies full SHA-256 digests with `F_NOCACHE`, seals `Qwen3.8-Flash-Next-Q4_K.tokspur.gguf` read-only (`0444`), and derives tokenizer and chat-template assets locally without copying model weights.

## Licence & Attribution

Distributed under the **Qwen Community License 1.0** ([`LICENSE`](LICENSE); see also [`NOTICE`](NOTICE)). Review Section 1 (attribution requirements above commercial scale thresholds) and Section 2 (separate licensing requirement for commercial Model as a Service or AI Work Assistant offerings) of the Qwen Community License 1.0 before commercial deployment.
