---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - vllm
  - safetensors
  - mirai
  - 2-bit
  - experimental
  - text-generation
---

# Qwen3.8-Flash-Next, Mirai S small, experimental

**This build replaces the v8 build** (Qwen3.8-Flash-Next-S-small-v8-experimental). It is the same size as v8 (30.7 GB
of trellis matrices against 30.6 GB). Its ROMA KL is 25% lower (0.0742 against 0.0993 in vLLM), and it is level with v8
on our agent circuits. It uses the recipe of
[trymirai/Qwen3.8-Flash-Next-S-experimental](https://huggingface.co/trymirai/Qwen3.8-Flash-Next-S-experimental)
(v11p, 38.7 GB of trellis matrices) at a smaller budget.

Qwen3.8-Flash-Next has 48 layers, 512 routed experts, an MTP drafter and an n-gram embedding table. Every large matrix is
stored as a QTIP-style trellis with a computed codebook (Mirai S).

**Rates.**
- **Routed experts:** 1.85 bits per weight on average. Each expert's gate_up and down matrices get their own rate,
  from 1.5 to 5 bits, set by the expert's share of routing weight on 28M calibration tokens. Gate_up averages 1.73 bits
  and down 2.03.
- **Per-layer budgets:** every layer gets the same expert budget, except that layers 12-23 get 0.15 bit more per expert
  weight and layers 36-47 0.15 bit less.
- **Everything else:** the attention, DeltaNet and shared-expert projections and the LM head use 5 bits, and the token
  embedding 4 bits.

**Training.** After the fit, quantization-aware training adjusted the row scales and norms, with the codes frozen. It
trained on the bf16 model's outputs over our trace mix, which includes agent sessions in custom tool-call formats. The
MTP drafter layer is Qwen's FP8 tensor from
[Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8), unchanged.

**How it differs from v11p:** 28.0 GB of routed experts instead of 35.9 GB, and a per-layer shift of 0.15 bit instead
of 0.3. Everything else is the same: calibration, backbone and LM head rates, and training.

This is an experimental build. It runs only in vLLM 0.31.0 through the plugin in `plugin/`, on one NVIDIA GPU, and it
has been tested on B200 only.

## Two folders, one model

The repo holds the same weights twice, with two versions of the n-gram embedding table. Download one folder.

| folder | n-gram table | size of the folder | host RAM for the table |
|---|---|---|---|
| `fp8-table/` | Qwen's FP8 table, unchanged | 86.2 GB | about 52 GB |
| `int4-table/` | our int4 rows | 61.2 GB | about 27 GB |

Our agent circuits ran with Qwen's FP8 table. The ROMA numbers below cover both tables. The repo root holds this card,
the LICENSE, `FORMAT.md`, `trellis_ref.py` and the plugin.

## Sizes

- Trellis matrices: 30.7 GB (`trellis-*.safetensors`, 125.0B weights, 1.97 bits per weight with row scales).
- Everything else vLLM loads, including the 2.7 GB FP8 MTP layer: 4.2 GB.
- n-gram table: 51.2 GB as FP8 (`ngram-*.safetensors` plus five tensors in `model-00002-of-00002.safetensors`) or
  26.2 GB as int4 (`ngram-int4/`). With `--engram-config '{"cpu_offload": true}'` it stays in host RAM.
- GPU memory for weights, from vLLM's load report: 36.5 GiB with MTP-3, with either table. The plugin keeps the
  backbone projections and the LM head decoded to int8 bytes after loading.

Our size table lists this build at 32.0 GB and v8 at 31.9 GB. That count leaves out the MTP layer and the n-gram table,
and it stores row scales in 16 bits; the files store them in 32 bits.

## Quickstart (one B200 or H100-class card)

```bash
hf download trymirai/Qwen3.8-Flash-Next-S-small-experimental --include "fp8-table/*" --include "plugin/*" \
  --local-dir qwen-next-s-small
uv venv --python 3.12 qns-venv
source qns-venv/bin/activate
uv pip install "vllm==0.31.0" ./qwen-next-s-small/plugin
vllm serve ./qwen-next-s-small/fp8-table --served-model-name qwen-next-s-small --language-model-only \
  --enable-prefix-caching --reasoning-parser qwen3 --engram-config '{"cpu_offload": true}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' --max-cudagraph-capture-size 64
```

For the int4 table, use `int4-table` in place of `fp8-table` in both the download and the serve command.

Keep the venv activated: vLLM's kernel builds need its `ninja` on `PATH`. The first start compiles the plugin's kernels
and captures CUDA graphs, which takes a few minutes. Then:

```bash
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen-next-s-small", "messages": [{"role": "user", "content": "What is 17 * 23?"}]}'
```

The model thinks by default; the answer is in `content` and the thinking in `reasoning_content`.

**Memory.** On one GPU, plan for about 37 GiB of weights plus the KV cache and activations. In host RAM, plan for the
n-gram table: about 52 GB with `fp8-table/`, 27 GB with `int4-table/`.

**Smaller cards.** This build has run only on B200s with the whole card available. Its weights take 36.5 GiB with
MTP-3, against 41.9 GiB for Qwen3.8-Flash-Next-S-experimental. That build serves on a 48 GiB budget with
`--max-model-len 61440 --max-num-seqs 4 --max-num-batched-tokens 64` and `--gpu-memory-utilization` set to 46 GiB of
the card. We have not tested those settings, or any card smaller than a B200, with this build.

Keep `--max-num-batched-tokens 64` on small cards. Larger prefill chunks decode all 512 experts of a layer to bf16 at
once, which needs 3.4 GB more.

**If the engine segfaults in `cuGraphLaunch`**, start it with `VLLM_USE_BREAKABLE_CUDAGRAPH=0`. vLLM 0.31 enables its
breakable CUDA graphs for this architecture, and earlier revisions of the plugin crashed intermittently with them. The
fallback costs about 3% of MTP-3 throughput and about 100 ms of time to first token.

## Quality

ROMA KL is the mean KL divergence from the bf16 model's next-token distribution over 37,620 positions of our
evaluation set; lower is better.

| build | ROMA KL | top-1 agreement |
|---|---|---|
| this repo (`fp8-table/`) | 0.0742 | 91% |
| this repo (`int4-table/`) | 0.0692 | 91% |
| v8, the build this one replaces (`fp8-table/`) | 0.0993 | 90% |
| for comparison: Qwen3.8-Flash-Next-S-experimental (v11p) | 0.0397 | 93% |

The int4 table gives a lower ROMA KL than Qwen's FP8 table here, as it did for v11p (0.0370 against 0.0397). The
torch reference of these weights (bf16 throughout, bf16 n-gram table) scored 0.0739. ROMA through vLLM and through
torch can differ by a few percent in either direction (v11p: 0.0397 in vLLM, 0.0371 in torch).

Our agent circuits ran on the package in this repo (`fp8-table/`), served by vLLM with MTP-3. The v8 row ran on v8's FP8
twin, which decodes the same 2-bit weights into an FP8 checkpoint for stock vLLM.

| build | hard | creative | assistant | AppWorld 80 x3 | AppWorld full | marathon | in-context |
|---|---|---|---|---|---|---|---|
| this repo | 51/55 | 8/8 | 120/136 | 70% | 63% | 100% (362/363) | 73% |
| v8 (FP8 twin) | 51/55 | 4/4 | 125/136 | 68% | not run | 100% (384/385) | not run |
| Qwen3.8-Flash-Next-S-experimental | 53/55 | 7/8 | 113/136 | 76% | 67% | 100% (323/323) | 78% |
| Qwen3.8-Flash-Next-FP8 | 52/55 | 8/8 | 127/136 | 76% | 67% | 96% (345/359) | 87% |

- **hard:** 11 exact-arithmetic and coding problems, 5 samples each. All four of this build's misses are on the same
  expression-evaluator problem, which the v8 row also misses four times.
- **creative:** implement an invented interpreter and binary format from a spec in a file-editing agent loop, graded
  by hidden tests: 2 tasks x 4 seeds (2 x 2 for v8). One of this build's 8 runs ended at step 31, when our harness
  failed to run a test script that contained a NUL byte. Its code at that point passed all 45 hidden tests.
- **assistant:** a simulated 10-day work week (calendar, mail, jobs, travel), 34 checked tasks x 4 seeds. Seed-to-seed
  spread is several tasks; over 8 seeds, Qwen3.8-Flash-Next-S-experimental scored 240/272 against 249/272 for its FP8
  twin.
- **AppWorld:** the 80-task contract run 3 times, and the full 585-task contract once.
- **marathon:** 6 agent sessions, each grown to about 380k tokens with YaRN (factor 1.5, 393k context); share of tasks
  fully passed.
- **in-context:** 190 generated items (long ledger logs, rule induction, an invented DSL spec), 40k-token budget.
  49 of this build's 52 misses ran out of tokens while thinking. It answered 3 items wrong.

## Speed

Not measured on a quiet machine yet. This build uses the same plugin and kernels as Qwen3.8-Flash-Next-S-experimental,
which decodes at 156 tok/s (216 tok/s with MTP-3) on one B200 at one request at a time.

## Known limits

- vLLM 0.31.0 only: the plugin subclasses vLLM's Qwen4Exp classes. One GPU only, no tensor parallelism.
- Text only (`--language-model-only`); the vision tower is not included.
- Prefill is slower than the FP8 twin's: a prefill pass longer than 64 tokens decodes all 512 experts of every layer.
- That long-prefill path computes the routed experts in bf16, so its numerics differ slightly from the decode path.
  Outputs depend a little on how a prompt is chunked.
- The MTP drafter proposes only token ids below 98,304. The output is unaffected; only draft acceptance can drop.
- vLLM's tool-call parsers have not been tested with this build. Our circuits parse tool calls from text.

## Files

In each of `fp8-table/` and `int4-table/`:

- `config.json`: Qwen's config, plus `quantization_config` (`quant_method: qwen_next_s`, format
  `qwen-next-s-trellis-v1`).
- `trellis-NN.safetensors`, `trellis-head.safetensors`, `trellis.safetensors.index.json`: the trellis matrices,
  identical in both folders. `FORMAT.md` specifies their bytes, and `trellis_ref.py` is an executable reference
  (pack, unpack, decode).
- `model-*.safetensors`, `model.safetensors.index.json`: every other tensor under Qwen's names.
- `ngram-*.safetensors` (`fp8-table/`) or `ngram-int4/` (`int4-table/`): the n-gram table.
- Tokenizer, chat template, generation config and LICENSE: from Qwen/Qwen3.8-Flash-Next.

`plugin/` is the `qwen-next-s` vLLM plugin (pip-installable; it registers itself through vLLM's plugin entry point).
It is the same plugin as in Qwen3.8-Flash-Next-S-experimental.

## License

The [Qwen Community License 1.0](LICENSE) covers this repo, including the FP8 MTP and n-gram tensors copied from
Qwen's checkpoint. It has conditions for commercial use (for products above 100M monthly users or US$20M monthly
revenue, and for model-as-a-service or AI work assistant businesses); read it before using the model.
