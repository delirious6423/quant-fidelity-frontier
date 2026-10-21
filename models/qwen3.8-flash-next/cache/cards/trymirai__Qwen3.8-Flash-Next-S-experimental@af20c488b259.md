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

# Qwen3.8-Flash-Next, Mirai S, experimental

Qwen3.8-Flash-Next (48 layers, 512 routed experts, MTP drafter, n-gram embedding table) with every large matrix stored
as a QTIP-style trellis with a computed codebook (Mirai S, recipe v11p). The routed experts average 2.4 bits per
weight: each expert matrix has 1 to 5 bits (1 bit for 24% of the expert weights, 2 for 47%, 3 for 27%, 4-5 for 2%).
The attention, DeltaNet and shared-expert projections and the LM head use 5 bits, the token embedding 4. The MTP
drafter layer is Qwen's FP8 layer from [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8).

The 51.2B-weight n-gram embedding table comes in two versions, each in its own complete model folder; download one:

| folder | n-gram table | table size | download | ROMA KL |
|---|---|---|---|---|
| `fp8-table/` | Qwen's FP8 table, unchanged | 51.2 GB | 94.1 GB | 0.0397 |
| `int4-table/` | our int4 fit of Qwen's bf16 table, one scale per 160-value row | 26.2 GB | 69.2 GB | 0.0370 |

Both use the same trellis weights. ROMA (defined under Quality) puts the two tables level: their difference is within
the measurement's noise. The circuits and speed numbers below are for `fp8-table/` unless a row says otherwise.

This is an experimental build. It runs only in vLLM 0.31.0 through the plugin in `plugin/`, on one NVIDIA GPU, and it
has been tested on B200 only.

## Sizes

- Trellis matrices: 38.7 GB (`trellis-*.safetensors`, 125.0B weights, 2.47 bits per weight with row scales).
- Everything else vLLM loads as stored, including the FP8 MTP layer: 4.2 GB (`model-*.safetensors`).
- n-gram table, FP8: 51.2 GB, of which 49.6 GB is in `ngram-*.safetensors` and 1.6 GB (four shards and the table's
  scale, as in Qwen's checkpoint) in `model-00002-of-00002.safetensors`. int4: 26.2 GB in `ngram-int4/` (format in
  `FORMAT.md`). With `--engram-config '{"cpu_offload": true}'` the table stays in host RAM: about 51 GB or 26 GB of
  free RAM.
- GPU weights: 38.75 GiB as stored. The plugin keeps the backbone projections and the LM head decoded to int8 bytes
  after loading, so the GPU holds 40.1 GiB of weights (vLLM's load report, which includes its buffers, says 41-44 GiB).
  The FP8 twin of the same weights needs 126.8 GiB.

## Quickstart (one B200 or H100-class card)

With the int4 table (use `--exclude "int4-table/*"` and `fp8-table` for the FP8 one):

```bash
hf download trymirai/Qwen3.8-Flash-Next-S-experimental --exclude "fp8-table/*" --local-dir qwen-next-s
uv venv --python 3.12 qns-venv
source qns-venv/bin/activate
uv pip install "vllm==0.31.0" ./qwen-next-s/plugin
vllm serve ./qwen-next-s/int4-table --served-model-name qwen-next-s --language-model-only --enable-prefix-caching \
  --reasoning-parser qwen3 --engram-config '{"cpu_offload": true}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' --max-cudagraph-capture-size 64
```

Keep the venv activated: vLLM's kernel builds need its `ninja` on `PATH`. The first start compiles the plugin's kernels
and captures CUDA graphs, which takes a few minutes. Then:

```bash
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen-next-s", "messages": [{"role": "user", "content": "What is 17 * 23?"}]}'
```

The model thinks by default; the answer is in `content` and the thinking in `reasoning`.

**A card with 48 GiB of memory.** Add these flags to the same command:

```bash
  --max-model-len 61440 --max-num-seqs 4 --max-num-batched-tokens 64 --gpu-memory-utilization 0.958
```

- `--gpu-memory-utilization` is a share of the card's total memory. vLLM's budget has to be 46.0 GiB, and the
  engine uses about 1.6 GiB beyond it. On a card with C GiB in total, use 46 / C: 0.958 for 48 GiB.
- `--max-num-batched-tokens 64` keeps prefill on the routed-expert path that decodes only the experts in use.
  Larger prefill chunks decode all 512 experts of a layer to bf16 at once, which needs 3.4 GB more.

We tested this with `fp8-table/` on a B200 with all but 48 GiB of its memory held by another process
(`--gpu-memory-utilization 0.258`, the same 46.0 GiB budget); the table is in host RAM, so `int4-table/` needs the
same GPU memory. It has not run on real 48 GB hardware. An L40S (45 GiB) does not have enough memory, and
GPUs older than Hopper are untested.

**If the engine segfaults in `cuGraphLaunch`**, start it with `VLLM_USE_BREAKABLE_CUDAGRAPH=0`. vLLM 0.31 enables its
breakable CUDA graphs for this architecture. At earlier revisions of the plugin they crashed intermittently in
in-process engines; this release did not crash in 6 in-process runs or in any server session. The fallback costs
about 3% of MTP-3 throughput (lower draft acceptance) and about 100 ms of time to first token.

## Quality

ROMA KL is the mean KL divergence from the bf16 model's next-token distribution over 37,620 positions of our
evaluation set; lower is better.

| build | ROMA KL | top-1 agreement |
|---|---|---|
| this repo, `fp8-table/` | 0.0397 | 93% |
| this repo, `int4-table/` | 0.0370 | 94% |
| FP8 twin (the same 2-bit weights decoded into an FP8 checkpoint, stock vLLM) | 0.0411 | 93% |

The two tables are level. Qwen's FP8 table is closer to the bf16 table than our int4 fit (error weighted by the
layer's key and value projections: 0.0005 against 0.0067), but the two builds' outputs differ by only KL 0.0054 (98%
top-1 agreement), about the size of run-to-run numeric changes in this build, so the lower ROMA of `int4-table/` is
noise rather than a better table.

Our agent circuits, served by vLLM with MTP-3 (the FP8 twin row is the same weights in FP8, the teacher row is Qwen's
FP8 checkpoint):

| build | hard | creative | assistant | AppWorld 80 x3 | AppWorld 585 | marathon | in-context |
|---|---|---|---|---|---|---|---|
| this repo, `fp8-table/` | 53/55 | 7/8 | 113/136 | 76% | 67% | 100% (323/323) | 78% |
| this repo, `int4-table/` | 50/55 | | | | | | |
| FP8 twin | 51/55 | 8/8 | 122/136 | 74% | 68% | 99% (335/339) | 79% |
| Qwen3.8-Flash-Next-FP8 | 52/55 | 8/8 | 127/136 | 76% | 67% | 96% (345/359) | 87% |

- **hard:** 11 exact-arithmetic and coding problems, 5 samples each.
- **creative:** implement an invented interpreter and binary format from a spec in a file-editing agent loop, 2 tasks x 4
  seeds, graded by hidden tests. The miss passed 39 of 45 tests.
- **assistant:** a simulated 10-day work week (calendar, mail, jobs, travel), 34 checked tasks x 4 seeds. Over 4 more
  seeds both this repo and the FP8 twin scored 127/136; over all 8 seeds this repo has 88% and the twin 92%, a
  difference within seed noise.
- **AppWorld:** the 80-task contract run 3 times, and the full 585-task contract once.
- **marathon:** 6 agent sessions, each grown to about 380k tokens with YaRN (factor 1.5, 393k context); share of tasks
  fully passed.
- **in-context:** 190 questions over 32k-230k-token documents (ledgers, rule books, specs). Most wrong answers are
  thinking traces that ran out of tokens: 35 of 42 for this repo, 38 of 39 for the FP8 twin, 22 of 25 for the teacher.

No circuit showed a format or parser failure. `int4-table/` has run the hard battery only so far: one sample fewer
than `fp8-table/` on each of three problems, within the battery's run-to-run spread (the FP8 twin scored 51/55).

## Speed

One B200, vLLM 0.31.0, one request at a time, `fp8-table/` against the FP8 twin of the same weights:

| | this repo | FP8 twin |
|---|---|---|
| decode | 156 tok/s | 148 tok/s |
| decode with MTP-3 | 216 tok/s | 216 tok/s |
| time to first token, short prompt | ~90 ms | ~60 ms with MTP, ~145 ms without |
| time to first token, ~4,000-word prompt | ~500 ms | ~315 ms |

The MTP-3 row is a 512-token essay answer. Over five prompts, including two long ones, this repo averages 217 tok/s
and the twin 221.

With the 48 GiB settings, decode with MTP-3 is the same (217 tok/s). Prefill runs in 64-token chunks at about 1,400
tokens/s, so a 56k-token prompt takes 41 s to the first token.

## Known limits

- vLLM 0.31.0 only: the plugin subclasses vLLM's Qwen4Exp classes. One GPU only, no tensor parallelism.
- Text only (`--language-model-only`); the vision tower is not included.
- Prefill is slower than the FP8 twin's: a prefill pass longer than 64 tokens decodes all 512 experts of every layer
  (about 2 ms per layer per pass).
- That long-prefill path computes the routed experts in bf16, so its numerics differ slightly from the decode path.
  Outputs depend a little on how a prompt is chunked.
- The MTP drafter proposes only token ids below 98,304 (99.8% of generated tokens in English prose, code and
  reasoning). The output is unaffected; only draft acceptance can drop.
- vLLM's tool-call parsers have not been tested with this build. Our circuits parse tool calls from text.

## Files

Each of `fp8-table/` and `int4-table/` is a complete model directory:
- `config.json`: Qwen's config, plus `quantization_config` (`quant_method: qwen_next_s`, format
  `qwen-next-s-trellis-v1`).
- `trellis-NN.safetensors`, `trellis-head.safetensors`, `trellis.safetensors.index.json`: the trellis matrices.
- `model-*.safetensors`, `model.safetensors.index.json`: every other tensor under Qwen's names.
- The n-gram table: `ngram-*.safetensors` (plus five tensors in `model-00002-of-00002.safetensors`) in `fp8-table/`,
  `ngram-int4/shard_NNN.safetensors` in `int4-table/`. The plugin uses the int4 table when `ngram-int4/` exists.
- Tokenizer, chat template, generation config and LICENSE: from Qwen/Qwen3.8-Flash-Next.

At the top level: `FORMAT.md` specifies the trellis and int4-table bytes, `trellis_ref.py` is an executable reference
(pack, unpack, decode), and `plugin/` is the `qwen-next-s` vLLM plugin (pip-installable; it registers itself through
vLLM's plugin entry point).

## License

The [Qwen Community License 1.0](LICENSE) covers this repo, including the FP8 MTP layer and n-gram table copied from
Qwen's checkpoint and our int4 table derived from Qwen's table. It has conditions for commercial use (for products
above 100M monthly users or US$20M monthly revenue, and for model-as-a-service or AI work assistant businesses); read
it before using the model.
