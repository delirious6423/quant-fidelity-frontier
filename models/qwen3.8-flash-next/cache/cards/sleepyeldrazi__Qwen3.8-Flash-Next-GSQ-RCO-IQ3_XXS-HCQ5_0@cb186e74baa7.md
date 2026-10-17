---
license: apache-2.0
language:
- en
library_name: llama.cpp
tags:
- gguf
- gsq
- rco
- quantization
- mixed-precision
- moe
- speculative-decoding
- mtp
- strix-halo
- qwen
base_model: Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS — HC-Q5_0 ("HCQ5_0")

**Qwen3.8-Flash-Next** (arch `qwen4exp`) on the **GSQ-RCO IQ3_XXS** quantization pack, with one
measured model change: the hyper-connection (HC) trunk requantized **BF16 -> Q5_0**, plus
**MTP draft sidecars** (a full-vocab one and a reduced 65,280-token LM-head one) for
speculative decoding.

- Base pack: [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (IQ3_XXS)
- Model change: 194 tensors (all `hc_{attn,ffn}_{up,down}.weight` + `output_hc_{up,down}.weight`)
  BF16 -> Q5_0. **Every other tensor is byte-identical** to the base pack (verified by spot-check).
- Draft: `mtp-draft-q3-v65k.gguf` — the model's own MTP (nextn) layer, Q3 mix, with the draft
  LM head sliced to the 65,280 most frequent tokens (99.6% of vocab).

Measured on the target hardware (Ryzen AI MAX+ 395 / Radeon 8060S, `gfx1151`, 60 GB LPDDR5X,
Vulkan/RADV):

| config (spec off unless noted) | d0 | 16K | 128K |
| --- | ---: | ---: | ---: |
| base IQ3_XXS pack (reference) | 29.1 t/s | 26.4 t/s | ~24.5 t/s |
| **HCQ5_0** (this trunk) | **34.6 t/s (+18.9%)** | **30.7 t/s (+16.0%)** | — |
| HCQ5_0 + MTP (n_max 3, full draft) | 39.5 t/s | — | — |
| HCQ5_0 + MTP (n_max 3, **v65k draft**) | +7% over full draft at 32K | +1-4% at 128K | +0.6 GiB RAM headroom at 256K |

Decode at depth 0 is bandwidth-bound on this UMA machine (34.6 t/s x ~3.4 GB/token is at the
measured ~115 GB/s DRAM read roofline); the HC remap removes 0.845 GB from every forward pass
(the HC trunk was ~1.28 GB/pass of BF16, ~33% of all per-token weight bytes, while the experts
were already at ~2.5 bpw).

## What changed in the weights (and why)

| file | what | why |
| --- | --- | --- |
| `...-HCQ5_0-00001-of-00002.gguf` | 194 HC tensors BF16 -> Q5_0 (5.25 bpw); saves 0.83 GB on disk and 0.845 GB read per token. All 1,030 other tensors byte-copied (quantize run: 1,223 copied + 194 converted, zero warnings). | The HC up/down GEMMs are read **every token, every layer** (48 layers x 2 modules). They were the only large dense block left at 16 bpw in the pack. Q5_0 was chosen because every HC matrix has `ne0 % 32 == 0` (Q5_K's 256-wide superblock is not valid for the `[320, 10240]` up matrices). |
| `...-HCQ5_0-00002-of-00002.gguf` | The 54.4 GB PLE / n-gram embedding table (`per_layer_token_embd.weight`, Q8_0, 320,001,536 x 160). Byte-identical to the base pack; it streams from SSD, never resident. | Shipped as shard 2 so the model stays a normal 2-shard GGUF. |
| `mtp-draft-q3-v65k.gguf` | 34 tensors: block 48 (the MTP/nextn layer) + `token_embd` + `output_hc_*`. `output.weight` sliced 248320 -> **65280** columns (byte-exact Q5_K prefix, no requantization); `token_embd` and the tokenizer stay full-vocab. Extra key `qwen4exp.draft_vocab_size = 65280`. | The draft head reads its LM head every draft step. Keeping only the K lowest token ids (BPE merge order as a frequency proxy) cuts the head from 0.44 GB to 0.11 GB; the engine pads the K logits back to full vocab with `-inf`, so the target still verifies over the whole vocabulary. |

Quality gates for the HCQ5_0 remap (run before shipping):
- 128-token greedy generation: identical for the first 47 tokens, then a benign word-choice
  branch (a requant of the residual-mixing path flips one logit tie; not corruption).
- 33K-token needle (marker at 16K depth): found.
- 1K-token coherence: clean.
- **Perplexity was not measured** (needs a calibration set + long CPU/GPU run); GSQHalo.cpp
  reports 3.9044 for the base IQ3_XXS pack on wikitext if you want a reference point.

## Files

| file | size | notes |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ5_0-00001-of-00002.gguf` | 46.2 GiB | trunk: 48 layers, 1,223 tensors, mixed GSQ pack (IQ2_XXS..Q6_K) + 194x Q5_0 |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ5_0-00002-of-00002.gguf` | 54.4 GiB | PLE n-gram table (shard 2, streamed) |
| `mtp-draft-q3-v65k.gguf` | 1.2 GiB | MTP draft sidecar, **reduced** head (fastest; needs `feat/strix-prod`-class build) |
| `mtp-draft-q3.gguf` | 1.5 GiB | same MTP draft, **full** 248,320-token head (the original sidecar; use if a build predates the reduced-vocab feature) |

Model facts (from the GGUF): arch `qwen4exp`, `block_count 48`, `embedding_length 2560`,
`attention.head_count 24`, `expert_count 512` / `expert_used_count 10` (MoE "A3B" class),
`context_length 262144`, vocab 248,320 (ggml `gpt2` tokenizer, `qwen35` pre-tokenizer),
PLE: n-gram size 3, 16 heads, 160-wide per-layer embeddings. All quantization types in the
files are **standard ggml types** (F32, BF16, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XXS/XS/S,
IQ3_XXS/S, IQ4_XS/NL, Q5_0, F16) — no custom/proprietary formats.

## How to run

> **Engine note (verified 2026-10-08 against ggml-org/llama.cpp master `08246a2`):**
> the `qwen4exp` architecture itself is upstream llama.cpp (merged 2026-08), and this trunk
> **loads and runs on stock llama.cpp** — verified by a stock `llama-quantize` full-tensor
> dry-run of both shards (all 1,224 tensors accepted, no errors; only standard types, no
> fork-only quant formats). The caveats are about the *sidecar features*, not the weights:
>
> | feature | stock llama.cpp | recommended fork |
> | --- | --- | --- |
> | trunk (2 shards) | loads, runs | loads, runs (fastest) |
> | PLE table (shard 2) | `-lzm on` (mmap, on-demand rows) or `-lm dio` | `--ngram-on-disk` (O_DIRECT + 64 reader threads, 256 MiB row cache) or `-lzm on-direct` |
> | MTP draft (both files here) | **not supported**: the files load, but stock's MTP graph requires
> `blk.48.nextn.hc_head_norm.weight`, which these sidecars do not carry (they name the head-mixer
> norm `blk.48.nextn.shared_head_norm.weight`), and `qwen4exp.draft_vocab_size` is a fork-only key |
> `--spec-type draft-mtp` works with both drafts |
>
> **Use a fork for MTP.** The fork this model was developed and measured on is
> [`sleepyeldrazi/llama.cpp-strix`](https://github.com/sleepyeldrazi/llama.cpp-strix),
> branch [`feat/strix-prod`](https://github.com/sleepyeldrazi/llama.cpp-strix/tree/feat/strix-prod)
> (a fork of [`LaurentZuijdwijk/llama.cpp`](https://github.com/LaurentZuijdwijk/llama.cpp);
> Vulkan/Strix-Halo; carries the sidecar-MTP loader that accepts these drafts, the reduced-vocab
> draft support, `--ngram-on-disk`, QSA gather, drafter arbiter).
>
> **GSQHalo.cpp** ([`Aristo94/GSQHalo.cpp`](https://github.com/Aristo94/GSQHalo.cpp), ROCm/HIP,
> fork of `halo-box/strix-llama.cpp`) runs the **trunk** fine (it has GSQ-RCO prefill kernels and
> `--ngram-on-disk`), but **neither draft in this repo loads there**: its sidecar loader requires
> the final head mixer as either `nextn.hc_head_{norm,down,up}` or the trunk's
> `output_hc_{norm,down,up}` set, while these sidecars ship `nextn.shared_head_norm` +
> `output_hc_{down,up}` only (load aborts with "MTP block has no final mixer"), and the v65k file's
> 65,280-wide head is a dimension mismatch anyway. A GSQHalo-compatible draft is regenerable from
> the model's nextn layer with the converter's expected names.

### Stock llama.cpp (trunk only, no MTP)

```sh
llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ5_0-00001-of-00002.gguf \
  -c 262144 -ngl 999 -fa on --kv-unified \
  -ctk q4_0 -ctv q4_0 \
  -lzm on \            # PLE table streamed from disk (mmap); or -lm dio
  -b 512 -ub 512 -t 16 --parallel 1
```

On a 60 GB UMA machine, 256K context with q4_0 KV + full weights fits with ~3-7 GiB to spare
(measured: 50.1 GiB in the driver + ~2 GiB process + ~1.3 GiB kernel).

### Custom fork (`feat/strix-prod`) — full config with MTP (what the numbers above used)

```sh
llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ5_0-00001-of-00002.gguf \
  -c 262144 -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 \
  -fa on --kv-unified --n-gpu-layers 999 --device vulkan0 \
  --ngram-on-disk --fit off -ctxcp 2 \
  -b 512 -ub 512 -t 16 --parallel 1 \
  -md mtp-draft-q3-v65k.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.8
```

Measured settings that mattered (see `PORT-LOG.md` of the development repo for the full A/Bs):
- **MTP `n_max 3`**: best at every depth. Deeper drafts cost an extra full-MoE draft pass
  without enough acceptance gain; adaptive n_max was slower than fixed on this drafter.
- **q4_0 KV (main and draft)**: 256K steady state with ~5.7 GiB headroom; q8_0 is the
  alternative at 128K (+4% faster than f16).
- **MTP is content-dependent**: with high-acceptance text it is a big win (up to ~1.9x on the
  base pack at depth 0), but on low-acceptance prose it can be *slower* than spec-off — the
  drafter is a full MoE layer. The fork ships a drafter cost arbiter
  (`--spec-draft-arbiter`, off by default) that stops drafting when it stops paying.
- `-ctxcp` (context checkpoints) is ~112 MiB+ each and only needed for prefix caching.

### GSQHalo.cpp (public alternative; trunk only)

```sh
# from https://github.com/Aristo94/GSQHalo.cpp (ROCm 7.14 / gfx1151) — trunk, no MTP
llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ5_0-00001-of-00002.gguf \
  -c 262144 -ngl 999 -fa on -lm dio -lzm on -b 8192 -ub 8192 \
  --cache-ram 2048 -ctxcp 8
```

(The trunk runs there; MTP with the drafts in this repo does not — see the engine table above.)

## Provenance

- Weights: GSQ-RCO IQ3_XXS pack (ISTA DASLab) -> HC trunk BF16 -> Q5_0 remap with stock
  `llama-quantize --tensor-type-file` (1,030 tensors byte-copied, 194 converted).
- Drafts: the model's MTP head exported as a sidecar (`mtp-draft-q3.gguf`), and a reduced
  version of it with the head sliced to 65,280 columns (`mtp-draft-q3-v65k.gguf`,
  `port-tools/slice-draft-vocab.py`, FR-Spec style, cf. vLLM `QWEN4EXP_DRAFT_VOCAB`).
- Development + measurements: engine work in
  [`sleepyeldrazi/llama.cpp-strix`](https://github.com/sleepyeldrazi/llama.cpp-strix)
  branch `feat/strix-prod` (Vulkan/Strix-Halo fork of `LaurentZuijdwijk/llama.cpp`).
  The full port log, baselines, and per-feature A/B reports (`PORT-LOG.md`, `port-notes/`,
  `port-tools/`) live in the author's working directory; the serve script used for the numbers
  above is `serve-qwen-hcq5-mtp.sh` from the same tree.

## License

Apache-2.0, like the base [GSQ-RCO pack](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
and [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). This repo contains a
derived quantization (HC trunk BF16 -> Q5_0, everything else byte-identical) and a derived MTP
draft sidecar.
