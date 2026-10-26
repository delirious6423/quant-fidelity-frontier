---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- deepseek
- exl3
- trellis
- qtip
- quantized
- strix-halo
---

# DeepSeek-V4.1-Flash-1017 — Strix trellis (EXL3) dense packs K3 / K4 / K5 / K6

These are **trellis-coded (EXL3 / QTIP-style) quantizations of the dense linears** of DeepSeek-V4.1-Flash-1017:
attention `wq_a`, `wkv`, `wq_b`, `wo_a`, `wo_b`, the shared expert `w1`/`w3`/`w2`, and the output head. They come at
3, 4, 5 and 6 bits per weight. They are **not a standalone model**: there are no routed experts, embeddings or norms.
They are decode-site weights for the DeepSeek engine in
[github.com/hyukudan/strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab). That engine runs them
on top of
[kyuz0/deepseek-v4.1-flash-gguf](https://huggingface.co/kyuz0/deepseek-v4.1-flash-gguf) `DeepSeek-V4.1-Flash-1017-Q2.gguf`
together with the U6 dense overlay from
[hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6).

**Production (mix-B, identity `ds41-trellis-mixB-v1`) = `ours-K4` for the dense classes + `ours-K5` for the head.**
`ours-K3` and `ours-K6` are provided as size and quality alternatives. They are not production.

## Files

| Path | Files | Size (bytes) | SHA-256 of the pack manifest |
|---|---:|---:|---|
| `ours-K3/` + `ours-K3.sha256` | 43 | 2,688,762,054 | `40556bb37a7323385d959d38f448bbfa698760fdb58f240a71b920a8e199e4aa` |
| `ours-K4/` + `ours-K4.sha256` | 43 | 3,581,409,171 | `3b09b5ea4cc66ddc152b940e97e7c03270329d380292e8a5cf107fb8de122c68` |
| `ours-K5/` + `ours-K5.sha256` | 43 | 4,474,173,923 | `33efdd46a880be3f6a982e69f23702c5b5aeceb9b8f4a81c215e153009be677d` |
| `ours-K6/` + `ours-K6.sha256` | 43 | 5,367,055,488 | `16656afba2ec598828b4d241e664d47c0e0e10c7beab2da74bc02b1aefd891f5` |
| `mixB.txt` | 1 | 196 | mix-B manifest for `STRIX_DS41_TRELLIS_MIX` (see below) |

- **Pack contents.** Each pack has `dense.L0`…`dense.L39.safetensors`, `dense.head.safetensors`, `index.json` and
  `proxy_err.tsv` (the quantizer's per-tensor proxy error).
- **Checking.** Run `sha256sum -c SHA256SUMS` from the repo root; it covers every file of the repo.
  `ours-K<k>.sha256` lists the SHA-256 of every file of its pack. The K4 and K5 manifests separate hash and path with
  one space, so GNU `sha256sum -c` reads only the K3 and K6 ones directly.
- **Keep the manifests where they are.** Leave each `ours-K<k>.sha256` next to its `ours-K<k>/` directory, unchanged.
  The engine fingerprints a pack by `<dir>.sha256`, and that fingerprint feeds the namespaces of its prefix cache.
- **`mixB.txt`.** It is the production mapping with relative pack directories. Its two lines are `default=ours-K4` and
  `head=ours-K5`.

## Format

Each dense linear is stored as three tensors:
- `<name>.trellis`: I16 `[I/16][N/16][16·K]`;
- `<name>.suh`: F16 `[I]`;
- `<name>.svh`: F16 `[N]`.

The codebook is `mul1` (marker `0x83DCD12D`) and there is no bias. The effective weight is
W = diag(suh) · H_I · Q · H_O · diag(svh), with H the normalized 128-block Sylvester Hadamard. Names follow the original
checkpoint: `layers.L.attn.wq_a`, `…wkv`, `…wq_b`, `…wo_a.slice.<g>` (one linear per output group), `…wo_b`,
`layers.L.ffn.shared_experts.w1|w3|w2`, and `head`. The engine reads every shard header, so the partial `index.json` is
not used.

## How it was made

- **Quantizer:** exllamav3 **1.6.3** `quantize_exl3` (turboderp, MIT), driven by the engine's trellis reference
  script. Settings: codebook mul1, seed 787, LDLQ.
- **Source weights:** the dense weights are dequantized from the **original FP8 checkpoint**
  (`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`), not from a GGUF.
- **Calibration:** the Hessians come from the original model's own forward pass over 39 calibration segments
  (≈65K tokens), disjoint from the evaluation set.
- **Packs:** one pack per K = 3, 4, 5, 6, with every dense linear at that K.

## How to use (strix-halo-engine-lab)

You need two things:
1. An engine build with the trellis sites enabled: link
   [github.com/hyukudan/strix-trellis](https://github.com/hyukudan/strix-trellis) with
   `-DSTRIX_DS41_TRELLIS_LIBRARY=<archive> -DSTRIX_DS41_TRELLIS_INCLUDE_DIR=<dir>`. Details are in
   `docs/models/deepseek-v4.1/trellis-library-api.md`.
2. The Q2 GGUF, the DSpark drafter GGUF and the U6 overlay. The full commands are in the U6 repo's card.

Production environment for both ranks (`docs/models/deepseek-v4.1/production-defaults.md`):

```sh
A=/path/to/DeepSeek-V4.1-Flash-1017-strix-dense-U6
B=/path/to/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3      # this repo
export STRIX_DS41_OVERLAY=$A/c-u6.gguf STRIX_DS41_CB4=1
export STRIX_DS41_TRELLIS=$B/ours-K4 STRIX_DS41_TRELLIS_MIX=head=$B/ours-K5
export STRIX_DS41_TRELLIS_SITES=shexp,head STRIX_DS41_TRELLIS_HEADOUT=1
```

Then start `strix-deepseek41-serve` (or `strix-deepseek41-rank`) on both nodes as usual.

- **Using the manifest instead.** `cd $B && STRIX_DS41_TRELLIS_MIX=$B/mixB.txt` gives the same mapping. Pack
  directories in a manifest are resolved from the working directory.
- **Site classes.** `STRIX_DS41_TRELLIS_SITES` selects where the trellis weights are used: `q_b`, `o_a`, `o_b`,
  `shexp`, `head`, or `all`. Production uses `shexp,head`. If it is unset, the engine uses no trellis site.
- **Other mixes.** A mix can give each class its own pack, for example
  `STRIX_DS41_TRELLIS_MIX="default=$B/ours-K4,head=$B/ours-K5,shexp_w2=$B/ours-K3"`. The classes are `q_a`, `kv`, `q_b`,
  `o_a`, `o_b`, `shexp_w13`, `shexp_w2`, `shexp` and `head`.

## Exactness and quality notes

- **Trellis sites change the logits.** They are a different weight format, not a repack of the U6 or Q8_0 tensors.
  The engine announces the identity it runs: `ds41-trellis-k<K>-v1:<sites>` for one pack, or
  `ds41-trellis-mix-v1{…} (= ds41-trellis-mixB-v1)` for mix-B. Its bitwise gates are GPU against GPU within that
  identity: R1 against verify windows, batch, and both ranks. The library's GPU output matches its CPU reference only
  numerically, not bitwise.
- **Measured quality.** Teacher-forced over 103,908 positions, against the original FP8 checkpoint, with Q2 experts and
  A8 activations as in production:
  - **mix-B (K4 dense + K5 head): KL 0.0446 [0.0417, 0.0476], top-1 94.0 %.** Paired against the U6 overlay:
    −0.0001 [−0.0004, +0.0002], i.e. the same quality. Worst category: long English documents, +0.0045.
  - **K5 everywhere: KL 0.0444**, paired against U6 −0.0003.
  - For reference: U6 overlay 0.0447, Q8_0 dense 0.0441.
  - **All-K3 dense (K5 head): +0.0046 against U6.** This fails the +0.003 quality gate. A K3 head is clearly worse.
- **Bytes per token per node.** About 2.27 GB for mix-B, against 2.79 GB for K5 and 3.63 GB for the U6 dense.
- **Speed.** The trellis library has no K3/K4-tuned kernels yet, so the lower K saves bytes but little time.

## License and credits

- **License: MIT**, same as the base model. `LICENSE` reproduces DeepSeek's MIT notice ("Copyright (c) 2023 DeepSeek").
- **Base model:** [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
  revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, by DeepSeek (MIT).
- **Q2 GGUF:** the target these packs run with is
  [kyuz0/deepseek-v4.1-flash-gguf](https://huggingface.co/kyuz0/deepseek-v4.1-flash-gguf), revision
  `20408f699619df7071aa7cda6034e9cef5bf804b` (MIT).
- **Quantizer:** [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT). EXL3 is based on QTIP
  (Tseng et al.).
- **Engine and trellis library:** [strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab),
  [strix-trellis](https://github.com/hyukudan/strix-trellis).
