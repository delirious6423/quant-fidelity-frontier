---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF/blob/main/LICENSE
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

# Qwen3.8 Flash-Next Q4_K for Tokspur

[Website](https://tokspur.com/) | [Get started](https://tokspur.com/get-started/) | [Model guide](https://tokspur.com/docs/models/) | [Results](https://tokspur.com/results/) | [GitHub](https://github.com/san-digital/tokspur) | [DFlash2 drafter](https://huggingface.co/san-digital/Qwen3.8-Flash-Next-DFlash2)

The full [Qwen3.8 Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), quantised to 4 bits and packed ready to run in [Tokspur](https://tokspur.com/), a local inference server for the Apple M5 Max. Tokspur serves these files where they land. Nothing is repacked on first run and no second copy of the weights is written to disk.

| | |
| --- | --- |
| Base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/34567a4712bc9766c4449e2e98e4468bfa24d915) at revision `34567a4`, all 512 experts |
| Precision | Routed experts mostly Q4_K. Attention, routers and output head at Q8_0 or higher |
| Download | 118.2 GB in 3 files |
| Hardware | Apple M5 Max with a 40-core GPU and 128 GB of unified memory, macOS 27 or later |
| Runtime | Tokspur 0.3.1 or later |
| Context | 65,536 tokens by default, 131,072 optional |
| Input and output | Text only |
| Licence | Qwen Community License 1.0 |

## Quick start

Install Tokspur from [tokspur.com/get-started](https://tokspur.com/get-started/), then download, verify and serve the model:

```sh
tokspur pull
tokspur serve
```

`tokspur pull` fetches this repository, checks every file by SHA-256 and seals the GPU pack read-only. The server then answers chat completion requests at `http://127.0.0.1:18081/v1` under the model identifier `tokspur`. Add `--speed-preset` for the faster Metal 4 kernels.

Or download the files yourself and point the server at the folder:

```sh
hf download san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF --local-dir ~/Models/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF
tokspur serve ~/Models/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF
```

## Speed

Recorded on 15th November 2026 on an M5 Max with a 40-core GPU and 128 GB, with cached prompts, speed mode and the optional [DFlash2 drafter](https://huggingface.co/san-digital/Qwen3.8-Flash-Next-DFlash2).

| Measure | Result |
| --- | ---: |
| Complete request, 1,171 tokens of Rust that passed type checking | 165 tokens/sec |
| Generation counted from the first token | Up to 175 tokens/sec |

The [results page](https://tokspur.com/results/) has the method. The drafter card has Japanese and reasoning workloads.

## Files

| File | Size | Contents |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-Q4_K.tokspur.gguf` | 85.9 GB | GPU weights in Tokspur Planar Pack v3: experts, attention, routers and output head |
| `Qwen3.8-Flash-Next-base.gguf` | 29.5 GB | Embedding tables kept in host memory, model metadata, vocabulary and chat template |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2.79 GB | Built-in draft head that proposes tokens for the main model to check |

Keep the folder layout, including the `MTP/` subfolder, and leave the files unchanged after setup. The server rechecks them at every start.

<details>
<summary>Exact sizes and SHA-256 digests</summary>

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-Q4_K.tokspur.gguf` | 85,889,703,936 | `0967101337bbadc937e18a66b695a944bb1194f12d32bada66d6c0080fd51049` |
| `Qwen3.8-Flash-Next-base.gguf` | 29,486,514,464 | `6e27f395dc7527953671f79f03b9554c6136062dbdc6128116f82e02d3cee36a` |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2,786,568,256 | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |

Total 118,162,786,656 bytes.

</details>

## Quality

Compared with the `UD-Q4_K_XL` reference quantisation of the same model using `llama-perplexity`, on the same corpus, over 32 chunks of 2,048 tokens.

| Measure | Value |
| --- | ---: |
| Perplexity, Tokspur Q4_K | 4.539 ± 0.066 |
| Perplexity, reference | 4.565 ± 0.066 |
| Paired difference, Tokspur minus reference | -0.0257 ± 0.0126 |
| Paired log ratio | -0.0056 ± 0.0028 (98.18% chunk correlation) |
| Mean KL divergence | 0.0797 ± 0.0016 (median 0.0086) |
| Top-1 token agreement | 90.52% ± 0.16% |

Perplexity is about 0.6% lower than the reference, a gap of roughly two standard errors, so read the two as equivalent on this corpus. The test compares two 4-bit builds with each other. It does not measure loss against the full-precision model.

## How it was made

Converted with `tokspur-quant` straight from the 131 BF16 safetensors shards of `Qwen/Qwen3.8-Flash-Next` at revision `34567a4712bc9766c4449e2e98e4468bfa24d915`. No third-party GGUF files or outside drafter checkpoints were used.

- **GPU pack**, 1,254 tensors. The routed expert projections are 94 Q4_K, 2 Q5_K and 43 Q5_1 tensors, stored as co-located planar blocks laid out for Metal. Attention, GDN linear attention, shared expert, router, PLE projection and output head weights are Q8_0, BF16 or F32.
- **Base GGUF**. Per-layer token embeddings in IQ4_NL (28.8 GB) and token embeddings in Q8_0 (0.68 GB), with full model metadata, the BPE vocabulary of 248,320 tokens and Qwen's own chat template.
- **Draft head**, 32 tensors. The shared multi-token prediction head from layer 48 in Q8_0, BF16 and F32.

<details>
<summary>Verification details</summary>

- The pack carries an embedded manifest, `manifest_sha256` `8fb0a3ac566b51e174e417b659e27f35ae3f36f6f6d95ebed41b5b688aa81906`, which Tokspur checks before execution.
- The chat template is Qwen's upstream `chat_template.jinja`, SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
- The vocabulary has 247,587 BPE merges, and the base GGUF also holds the n-gram hash constants.
- The state-space decay weights `-exp(a_log)` are stored as F32 with an exact binary64 to binary32 conversion.

</details>

## Limits

- The transformer weights live only in the planar pack, which is a Tokspur format. llama.cpp and other GGUF runtimes cannot run this repository.
- Tokspur refuses to start on a Mac with less than 128 GB of unified memory.
- Text only.

## Licence

The weights are distributed under the Qwen Community License 1.0. See [`LICENSE`](https://huggingface.co/san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF/blob/main/LICENSE) and [`NOTICE`](https://huggingface.co/san-digital/Qwen3.8-Flash-Next-Q4_K-Tokspur-GGUF/blob/main/NOTICE). Before commercial use, read Section 1 on attribution above its scale thresholds and Section 2, which requires a separate licence for Model as a Service and AI Work Assistant businesses. The Tokspur runtime is licensed separately under Apache License 2.0.
