---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- mtp
- speculative-decoding
- qwen4exp
- qwen3.8-flash-next
- strix-halo
- llama.cpp
---

# Qwen3.8-Flash-Next MTP draft head (GGUF, standard quant types)

The multi-token-prediction head that ships inside
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): one full
`qwen4exp` block (attention + 512-expert MoE + hyper-connections, 4B parameters), trained
jointly with the model. The upstream converter (PR 27742) drops it, so the popular GGUFs
have none. This one is exported straight from the bf16 checkpoint and quantized with
plain `llama-quantize`, so it runs on stock llama.cpp kernels (any backend), not a fork's
quant types.

**This is a draft head, not a model.** Use it as `-md` next to any Flash-Next GGUF
target (tested with unsloth's UD-Q4_K_XL and UD-IQ4_XS).

| file | size | what |
|---|---:|---|
| `Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf` | 2.5 GB | the one to use: Q4_K_M with the LM head and token embeddings at Q4_K (reading the draft's LM head is most of each draft step; and a draft quantized like its target agrees with it more often than a more precise one -- a Q8_0 head measured worse) |
| `Qwen3.8-Flash-Next-MTP-BF16.gguf` | 7.8 GB | unquantized export, to requantize as you like |

## Needs a patched llama.cpp (for now)

`qwen4exp` is not upstream yet and neither is its MTP graph. Two options:

- **llama.cpp `7584430716ee` (tag b10612)** + [PR 27742](https://github.com/ggml-org/llama.cpp/pull/27742)
  (`model: add Qwen3.8-Flash-Next`, pinned at `unslothai/llama.cpp@bea3b12d`) +
  [`patches/qwen4exp-mtp-draft-head.patch`](patches/qwen4exp-mtp-draft-head.patch) from this
  repo. The patch is the draft-head graph (JJJYmmm's PR 27739, reconciled by
  LaurentZuijdwijk) and the converter's `--mtp` export, plus one fix (an MTP-only export
  emits no PLE keys). Verified to apply and build on exactly that tree.
- Or build [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp)
  branch `vulkan/qwen4exp-rocmfpx`, which carries the same two commits (b98aa9847,
  ae96a0dc0) on a Vulkan backend tuned for Strix Halo.

Set `LLAMA_ATTN_ROT_DISABLE=1`: upstream's quantized-KV activation rotation (#21038) is
not supported by the qwen4exp attention path and the server aborts at load without it.

## Run

    LLAMA_ATTN_ROT_DISABLE=1 llama-server \
      -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
      -md Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf -ngld 999 \
      --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
      -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 262144

`p-min 0.75` stops drafting when the head is unsure; that matters more than depth. Deeper
fixed drafting (n-max 8, p-min 0.3) measured slower, and so did adaptive drafting: this
head carries its own MoE, so every drafted token is a real forward pass.

The target's 28.8 GB n-gram table (`per_layer_token_embd`) is the model's problem, not the
head's: upstream keeps it in host memory (`-ot per_layer_token_embd=CPU`), Laurent's fork
splits it per head so it fits a Vulkan device. Our numbers below were taken with a patch
that leaves it on disk and reads rows per batch; it changes memory, not speed.

## Attached to unsloth's UD-Q4_K_XL (no `-md` needed)

`unsloth-UD-Q4_K_XL-mtp-shards/` turns
[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)'s
UD-Q4_K_XL into a model that carries its own draft head, the way Qwen3.5/3.6-MoE GGUFs do
(`nextn_predict_layers = 1`, the head as `blk.48.*`). Two files, 1.9 GB, no re-download of
the 111 GB:

1. keep unsloth's shards 2, 3 and 4 and rename them `-0000N-of-00005.gguf`
2. drop in `…-00001-of-00005.gguf` (their shard 1 with the metadata changed: `block_count 49`,
   `nextn_predict_layers 1`, `compress_ratios` extended, `split.count 5`) and
   `…-00005-of-00005.gguf` (the 29 head tensors, the same bytes as the Q4_K_M head above)
3. run with `--spec-type draft-mtp` and no `-md`; the draft context is created against the
   target and shares its memory (~0.5 GB less than the separate head)

Made with [`patches/merge-mtp-shard.py`](patches/merge-mtp-shard.py), which does the same to
any split qwen4exp GGUF given a head file. Measured: within a few percent of the `-md` form
(34.5 vs 36.7 t/s on code, same on prose) -- the convenience is the point, not speed.

## Measured (AMD Strix Halo / Radeon 8060S, 128 GB unified, temperature 0, 300 tokens)

| backend, target | bare | + MTP | acceptance (code / prose) |
|---|---:|---:|---|
| ROCm, UD-Q4_K_XL | 20.3 t/s | **35.8** code / 22.6 prose | 0.90 / 0.74 |
| ROCm, UD-IQ4_XS | 18.0 / 18.6 | 32.8 / 22.1 | 0.84 / 0.68 |
| Vulkan (RADV, Laurent's fork), UD-IQ4_XS | 24.2 / 24.3 | **37.2** code / **30.3** prose | 0.88 / 0.82 |

Where the remaining time goes on this GPU, profiled: a target pass is ~47 ms for one
token and only ~4.4 ms per extra verified token, a head step ~3.4 ms, and about a third
of each pass is kernel-launch gaps (the target graph is ~8,000 nodes). Vulkan launches
cheaper, which is the whole of its lead.

## How it was made

    # the 31 bf16 shards that hold mtp.* plus embed_tokens / lm_head / the PLE constants (58 GB)
    python convert_hf_to_gguf.py --mtp --outtype bf16 --outfile Qwen3.8-Flash-Next-MTP-BF16.gguf src/
    llama-quantize --output-tensor-type Q4_K --token-embedding-type Q4_K \
        Qwen3.8-Flash-Next-MTP-BF16.gguf Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf Q4_K_M

34 tensors: `blk.48.*` (attention, indexer, MoE, hyper-connections), `blk.48.nextn.{eh_proj,enorm,hnorm}`,
`token_embd`, `output`, `output_hc_*`; `nextn_predict_layers = 1`, `block_count = 49`.

## Credits

`qwen4exp` support: Daniel Han (PR 27742). MTP graph: JJJYmmm (PR 27739), reconciled and
the converter export by [LaurentZuijdwijk](https://github.com/LaurentZuijdwijk/llama.cpp).
The "match the target's quant" observation: agentionai. Base model: the Qwen team,
Qwen Community License 1.0 (included).
