---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx-serve
pipeline_tag: text-generation
tags:
- exl3
- mlx-serve
- turboexl3
- qwen4_exp
- moe
- apple-silicon
---

# Qwen3.8-Flash-Next EXL3 4.05 bpw for mlx-serve

This is turboderp's EXL3 quantization of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at
4.05 bits per weight, converted to the file layout of [mlx-serve](https://github.com/ddalcu/mlx-serve) on Apple
silicon. The source is the branch [`4.05bpw_h6_ng6`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6) of
[turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3), commit `55a732e0`. The conversion copies turboderp's quantized bytes.
It does not quantize the model again.

## Status

- Stock mlx-serve cannot serve this pack. It has no EXL3 path for the trunk (`trunk_quant`) or for the EXL3 n-gram table.
- To serve it, you need two parts:
  1. mlx-serve with full-EXL3 support for Qwen3.8 (the `trunk_quant` block, the EXL3 n-gram table and the EXL3 MTP
     head). Our pull request that adds it to [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) is not open yet. <!-- [[MLX_SERVE_PR]] this line becomes the pull request's link when it is open -->
  2. The turboexl3 quant plugin. It is required: it claims the pack's `expert_quant` and `trunk_quant` blocks. Build
     it from the [`plugin/`](https://github.com/davidtai/turboexl3/tree/main/plugin) directory of
     [davidtai/turboexl3](https://github.com/davidtai/turboexl3).
- We tested this pack on one Apple M5 Max with 128 GB of memory. We did not test other Macs.
- We tested text only. We did not test image or video input.

## What is in this pack

| part | format |
|---|---|
| routed experts: 48 layers x 512 experts, and the 512 experts of the MTP layer | EXL3 K4 |
| attention q/k/v/o, Gated DeltaNet projections, shared experts (trunk and MTP layer) | EXL3 K6 |
| sparse-attention indexer projection (trunk and MTP layer) | EXL3 K4 |
| MTP `fc_hidden` and `fc_embedding` | EXL3 K5 |
| `lm_head` | EXL3 K6 |
| n-gram embedding table | EXL3 K6 (`exl3_ngram_trellis`) |
| MTP hyper-connection mixer | bf16 |
| vision tower | bf16 |
| token embeddings, routers, norms, hyper-connection weights, convolutions and other small tensors | bf16, not quantized |

K is the number of bits per weight of the EXL3 trellis code. All EXL3 tensors use the MUL1 codebook.

| size | GB |
|---|---:|
| whole download | 107.63 |
| without the n-gram table | 68.59 |
| routed experts (49 files) | 62.14 |
| trunk (`trunk-exl3-00` to `trunk-exl3-10`) | 5.52 |
| n-gram table | 39.04 |
| vision tower | 0.90 |

GB = 10^9 bytes.

## What this pack is not

- It is not a new quantization. Every EXL3 byte is turboderp's. The only part that is not his is the vision tower.
  It comes from ddalcu's mlx-serve pack (see below).
- It does not load in exllamav3, transformers, vLLM, SGLang or mlx-lm. For exllamav3, use turboderp's repository.
- It is not an official release of Qwen, of turboderp or of mlx-serve.
- We did not measure its quality. We ran no KLD or perplexity test on this pack. turboderp's
  [model card](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) shows KLD and perplexity against BF16 for his quants. He measured them with exllamav3 on his
  own files.

## Files and where they come from

| files | source | change |
|---|---|---|
| `experts-00.safetensors` to `experts-47.safetensors`, `experts-mtp-00.safetensors` | turboderp: the routed experts | Each layer's 512 per-expert tensors are stacked into one tensor per projection: `switch_mlp.{gate,up,down}_proj.{trellis,suh,svh}`. The bytes do not change. Each file has the stamp `{format: exl3, k: 4, codebook: mul1, window: 16}`. |
| `trunk-exl3-00.safetensors` to `trunk-exl3-10.safetensors` | turboderp: the rest of the text model and the MTP layer | The tensor names change to mlx-serve's names. Each EXL3 linear is stored as `.weight` (the trellis), `.scales` (suh) and `.biases` (svh). Its `mul1` value is checked (0x83DCD12D) and dropped. Unquantized f16 tensors are stored as bf16; the n-gram `head_bias` stays f16. The `conv1d` weights are relabelled from [C, 1, K] to [C, K, 1]; the bytes are the same. |
| `trunk-exl3-carry.safetensors` | turboderp: the MTP hyper-connection mixer and 4 small n-gram tensors | The mixer comes from his model shards, stored as bf16 instead of f16. The n-gram tensors (`head_bias`, the head offsets, the head vocabulary sizes and the layer multipliers) come from the header of his `ngram_embedding.safetensors`, renamed. |
| `ngram_embedding.safetensors` | turboderp: the n-gram table | No change. sha256 `c63613666649014e668cdbd32a4e3f1e8beda697eb4f3a1874300d93a1bfbc75`. His `4.05bpw_h6_ng6` and `5.05bpw_h6_ng6` branches ship this same file. |
| `trunk-model-vision.safetensors` | ddalcu: `model-vision.safetensors` of [ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit) | The same 333 bf16 tensors, written again. Only the file header differs. turboderp's EXL3 vision tower is not used. |
| `config.json` | ddalcu's mlx-serve config for this model | We added the EXL3 blocks and some notes (see [config.json](#configjson)). |
| `model.safetensors.index.json` | ours | It maps all 2,520 tensors to their files. |
| `generation_config.json` | Qwen | We added `default_chat_template_kwargs.enable_thinking: true`. |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `vocab.json`, `merges.txt`, `LICENSE` | Qwen | No change. These are the same bytes as in Qwen's and turboderp's repositories. |

### The n-gram table

Qwen3.8-Flash-Next adds a large n-gram embedding: a table that the model reads with hashed bigrams and trigrams of the
input tokens. It has 320,001,536 rows of 160 values, and Qwen counts it as 51B parameters. turboderp quantized it with
EXL3. mlx-serve maps the file and decodes on the CPU the rows that each token needs. It does not copy the table into GPU
memory: the table costs disk space and OS file cache, not resident memory.

This file is turboderp's, with no change (sha256 `c63613666649…`).

### config.json

- `expert_quant` (`{format: exl3, k: 4, codebook: mul1}`) and `trunk_quant` (`{format: exl3, codebook: mul1,
  window: 16, source: turboderp/Qwen3.8-Flash-Next-exl3@4.05bpw_h6_ng6}`) tell mlx-serve which tensors are EXL3.
- `ngram_table` (`{file: ngram_embedding.safetensors, format: exl3_ngram_trellis, k: 6, row_dim: 160, codebook:
  mul1}`) names the n-gram table.
- `qwen4_norm_convention: "delta"`: turboderp stores the zero-centered RMSNorm weights as Hugging Face does. mlx-serve adds
  1 to them when it loads the pack.
- `quantization` and `quantization_config` (4-bit affine, group 64) come from ddalcu's config, because mlx-serve's loader
  expects them. No tensor in this pack is affine.
- `vision_from: "hybrid"`, `quantization_note` and `mtp_mixer_from: "turboderp"` are notes. "hybrid" is our earlier pack. The vision shard came to this pack through it from ddalcu's
  pack.

## Run it with mlx-serve

These steps need the two parts in [Status](#status). Download the pack (108 GB):

```sh
hf download OpensourceWTF/Qwen3.8-Flash-Next-EXL3-4.05-mlx-serve --local-dir ~/models/Qwen3.8-Flash-Next-EXL3-4.05-mlx-serve
```

Build mlx-serve with the support in [Status](#status) and its plugin kit, then the turboexl3 plugin ([`plugin/`](https://github.com/davidtai/turboexl3/tree/main/plugin)):

```sh
# in the mlx-serve checkout
zig build -Doptimize=ReleaseFast
zig build kit      # stages zig-out/kit/mlx-serve-plugin-kit-<version>/

# in a checkout of github.com/davidtai/turboexl3
cd plugin
ln -s <mlx-serve checkout>/zig-out/kit/mlx-serve-plugin-kit-<version> kit
zig build          # writes zig-out/lib/libturboexl3.dylib
mlx-serve plugin check zig-out/lib/libturboexl3.dylib --model ~/models/Qwen3.8-Flash-Next-EXL3-4.05-mlx-serve
#   expected: claims expert_quant (by_marker), trunk_quant (by_marker)
mlx-serve plugin install zig-out/lib/libturboexl3.dylib
# then set "enabled": true in ~/.mlx-serve/plugins.json
```

Serve:

```sh
MLX_SERVE_NOCACHE_LOAD=1 mlx-serve --model ~/models/Qwen3.8-Flash-Next-EXL3-4.05-mlx-serve --serve
```

- `MLX_SERVE_NOCACHE_LOAD=1` reads the shards past the OS file cache, so the load does not also fill memory with cached
  copies of the shards. Our measurements used it.
- The default numerics use turboexl3's kernels, which reorder some sums. `MLX_SERVE_BIT_EXACT=1` selects exact numerics:
  the routed experts run on Sushi, the EXL3 kernels built into mlx-serve, and no sum is reordered. The two modes give
  different text.
- turboexl3 runs the routed experts only on an Apple M5 or later with macOS 26.2 or later. On other Macs, the plugin leaves
  the routed experts to Sushi. We did not test this.
- The plugin links no MLX of its own. The MLX version of the kit must be the same as the server's.

## Speed on an Apple M5 Max (128 GB)

**PR17B-era numbers:** measured on build PR17B of our mlx-serve tree. Our S20 sweeps (build S20 adds later
decode optimizations) will replace them.

**Method.** One Apple M5 Max with 128 GB, macOS 27.0.1. The server was our mlx-serve tree with turboexl3, before its port
to the current mlx-serve. Each cell is one run, with no repeats. For
each prompt length and mode, a warm-up server first ran the prompt for one token, which puts the prompt's n-gram rows
into the OS file cache. Then the other pack files were evicted from the cache, a new server started, the SoC cooled below
40 °C, and one request ran. The request was the prompt as one user message in Qwen's chat template (thinking on), greedy
(temperature 0), seed 1234, at most 1,024 new tokens. Server flags: `--ctx-size 262144 --max-concurrent 1
--mtp-max-depth 3 --mtp-typical 0.2 --prefix-cache-mem 1GB`, with `MLX_SERVE_GPU_CEILING_MB=114688` (102400 at 256K),
`MLX_SERVE_CACHE_LIMIT=2147483648`, `MLX_SERVE_NOCACHE_LOAD=1` and `MLX_SERVE_NGRAM_WARM=0`. The prompts are fixed texts
(Python source code with one coding task), the same bytes for every pack and mode. The prefill and decode rates come from
the server's log line for the request. TTFT is the client's time from sending the request to the first streamed token.
turboexl3 reorders sums, so the two modes give different answers; with MTP, their decode rates also depend on how many
drafts each answer accepts. Peak memory is the server's peak phys_footprint (Metal memory included); the file cache of
the n-gram table is not in it.

"turboexl3" is the default numerics. "exact" is `MLX_SERVE_BIT_EXACT=1`. Prefill and decode are in tokens per second.
`*` marks a run whose answer reached the 1,024-token cap (0 of the 18 runs).

| prompt | tokens | prefill, turboexl3 | prefill, exact | decode, turboexl3 | decode, exact | TTFT s, turboexl3 | TTFT s, exact | peak GiB, turboexl3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1,031 | 1373.0 | 1025.5 | 141.0 | 119.7 | 0.76 | 1.01 | 71.5 |
| 2K | 2,013 | 1423.7 | 1916.3 | 134.8 | 114.2 | 1.42 | 1.06 | 71.1 |
| 4K | 4,025 | 2116.1 | 1938.8 | 132.3 | 116.5 | 1.91 | 2.09 | 71.3 |
| 8K | 8,100 | 2711.0 | 2488.6 | 135.7 | 116.1 | 3.00 | 3.27 | 73.5 |
| 16K | 16,275 | 2830.0 | 2538.2 | 137.6 | 112.5 | 5.77 | 6.43 | 75.1 |
| 32K | 32,424 | 2759.3 | 2492.8 | 126.4 | 108.7 | 11.77 | 13.03 | 75.7 |
| 64K | 64,987 | 2639.1 | 2485.6 | 119.5 | 102.5 | 24.66 | 26.18 | 77.2 |
| 128K | 129,949 | 2512.4 | 2398.9 | 114.3 | 96.8 | 51.78 | 54.23 | 79.9 |
| 256K | 258,992 | 2344.0 | 2270.0 | 105.5 | 89.6 | 110.60 | 114.20 | 84.8 |

## Credits

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): the model, the tokenizer, the chat template and the license.
- [turboderp](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3): the EXL3 quantization that this pack is made from, made with
  [exllamav3](https://github.com/turboderp-org/exllamav3). EXL3 is based on QTIP by Cornell RelaxML.
- [ddalcu](https://github.com/ddalcu): [mlx-serve](https://github.com/ddalcu/mlx-serve) and its Qwen3.8 model code; the
  pack [ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit), the source of this pack's vision tower and base config; and
  [Sushi](https://github.com/ddalcu/sushi), the EXL3 kernels built into mlx-serve.
- [turboexl3](https://github.com/davidtai/turboexl3): the EXL3 Metal kernels of the default mode.
- The conversion scripts: [OpenSourceWTF/qwen38-exl3-mlx-serve-converter](https://github.com/OpenSourceWTF/qwen38-exl3-mlx-serve-converter).

## License

The weights are derived from Qwen3.8-Flash-Next and stay under the Qwen Community License 1.0: the `LICENSE` file, the
same as in Qwen's repository. turboderp's and ddalcu's repositories use the same license. A short summary (the `LICENSE`
file is the license):

- Keep the copyright notice and the permission notice with every copy.
- A commercial product or service that uses the model and has more than 100,000,000 monthly active users, or more than
  US$20,000,000 monthly revenue, must show the model name in its user interface.
- A licensee that runs, or whose affiliate runs, a "Model as a Service" or "AI Work Assistant" business (as the license
  defines them) needs a separate license from Qwen before any commercial use. Internal use that gives third parties no
  access to the model, its outputs or its capabilities does not need one.
- There is no warranty.

The code (mlx-serve, Sushi, turboexl3, exllamav3 and the conversion scripts) has its own licenses.
