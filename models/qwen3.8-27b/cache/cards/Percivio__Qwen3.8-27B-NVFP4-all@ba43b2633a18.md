---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- qwen3.8
- nvfp4
- fp4
- modelopt
- speculative-decoding
- mtp
- vllm
- blackwell
---

# Qwen3.8-27B-NVFP4-all

Qwen3.8-27B with the attention, MLP and main Gated DeltaNet projections and the output head in NVFP4, for single-stream decoding with MTP speculative decoding on one NVIDIA RTX PRO 6000 Blackwell (SM120).

It starts from NVIDIA's [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), which has its MLP and `lm_head` in NVFP4 but keeps the attention and Gated DeltaNet projections in per-tensor FP8. Two changes:

- **The 208 attention and Gated DeltaNet projections** (q/k/v/o in the 16 full-attention layers; `in_proj_qkv`, `in_proj_z` and `out_proj` in the 48 linear-attention layers) are re-quantized to NVFP4 from Qwen's BF16 weights ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)). Block scales come from an MSE search over amax fractions, as in ModelOpt's NVFP4 MSE calibrator. Projections that vLLM fuses share one global scale. Activation scales are NVIDIA's calibrated FP8 input scales divided by 6.
- **The MTP draft layer's seven projections**, which NVIDIA leaves in BF16, are replaced with the 128×128 block-FP8 weights of Qwen's official FP8 release ([Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)), in `mtp-fp8.safetensors`. Only the draft reads them.

Everything else, including the vision encoder, tokenizer and chat template, is byte-identical to NVIDIA's checkpoint. Total size is 18.4 GB.

As in NVIDIA's checkpoint, the token embedding, the vision encoder, the MTP layer's `fc` and the two small per-head gate projections of each Gated DeltaNet layer (`in_proj_a` and `in_proj_b`, 24M parameters in all) stay in BF16, which is why the Hub's tensor summary lists BF16. That summary also counts each U8 byte as one parameter, but each byte packs two NVFP4 weights, so it shows about 15B parameters for this 27B model.

## Speed on stock vLLM

Served with unmodified vLLM and the command under [Usage](#usage) (vLLM's default NVFP4 kernels, FP8 KV cache, MTP with five draft tokens) on one RTX PRO 6000 Blackwell Server Edition (AWS g7e.2xlarge), against the official FP8 release with MTP5 on the same host. Means of three benches on fresh engine starts (four for the FP8 release):

| Context (prompt tokens) | 64K (76.7K) | 128K (126.1K) | 256K (223.7K) |
|---|---:|---:|---:|
| Official FP8 release (`Qwen/Qwen3.8-27B-FP8`): decode tok/s | 77.8 | 72.4 | 68.8 |
| This checkpoint: decode tok/s | 152.5 | 135.6 | 125.7 |
| Speed-up in decode | 1.96× | 1.87× | 1.83× |
| Official FP8 release: full request, s (of which prefill) | 37.5 (11.2) | 51.2 (23.0) | 86.5 (56.6) |
| This checkpoint: full request, s (of which prefill) | 23.3 (9.8) | 37.3 (22.2) | 77.3 (61.0) |

A full request (prefill plus 2,048 output tokens) takes 38%, 27% and 11% less time than with the FP8 release.

## Quality

| | Official FP8 release | This checkpoint, stock vLLM | This checkpoint, research stack |
|---|---:|---:|---:|
| GSM8K, all 1,319 test problems | 1,231 | 1,240 (24 lost, 33 gained) | 1,231 (23 lost, 23 gained) |
| GSM8K, first 200 test problems | 186 | 188 | 189 |
| Needle retrieval in 64K–200K-token prompts | 18/18 | 18/18 | 18/18 |
| Six long-context agentic tasks, strict tools with vLLM's unfixed grammar, three runs | 16/18 | not run | 18/18 |
| The same, strict tools with the grammar fix, three runs | 14/18 | not run | 12/18 |
| The same, strict tools off, three runs | 18/18 | not run | 16/18 |
| Agentic tasks, all nine runs | 48/54 | not run | 46/54 |
| Reasoning loops in those runs | 0 | – | 0 |

Greedy decoding with thinking on; GSM8K answers are capped at 8,192 tokens, and "lost" and "gained" count problems against the FP8 release's answers. "Stock vLLM" is the command under [Usage](#usage). The research stack is the configuration the agentic tasks were run on: vLLM's Marlin kernels plus the patched draft head described below. With greedy decoding, speculative decoding returns the target model's own tokens whatever the draft proposes, but the Marlin and default kernels compute slightly different numbers, so the two paths do not produce identical text. On the release host, the published draft-head files scored 190/200 on GSM8K and 18/18 on needles with the Marlin kernels, and 187/200 and 18/18 with the default kernels.

The agentic tasks are two question-answering, two bug-repair and two proof-of-concept tasks over 128K–200K-token code contexts, served with tool calling and prefix caching and graded by our fixture tests and fact and citation checks. This is a small internal battery, not a public benchmark. The first agentic row used vLLM's unfixed strict-tool grammar (see [Limitations](#limitations)). The other two rows ran later on another RTX PRO 6000 host, each run on a fresh engine start: one with our [grammar fix](https://github.com/Percivio-Ltd/sm120-serving/tree/master/strict-tool-grammar) and one with the tools' `strict` flag off. Repeat runs of one configuration have moved by up to two tasks, so the differences between the columns are within run-to-run variation.

## Faster with a patched draft head

A vLLM patch, which is not part of this release, has the MTP draft score a pruned FP8 copy of the output head (65,536 of 248,320 vocabulary rows) instead of the whole head. Verification is unchanged. The patch and its settings are in [`pruned-draft-head/`](https://github.com/Percivio-Ltd/sm120-serving/tree/master/pruned-draft-head) of [Percivio-Ltd/sm120-serving](https://github.com/Percivio-Ltd/sm120-serving). Same host and settings as above, three benches:

| Context (prompt tokens) | 64K (76.7K) | 128K (126.1K) | 256K (223.7K) |
|---|---:|---:|---:|
| This checkpoint with the patched draft head: decode tok/s | 163.8 | 142.8 | 132.8 |
| Speed-up in decode over the FP8 release | 2.10× | 1.97× | 1.93× |
| Full request, s (of which prefill) | 22.3 (9.8) | 36.6 (22.2) | 76.5 (61.0) |

[`serve.sh draft-head`](https://github.com/Percivio-Ltd/sm120-serving/blob/master/qwen3.8-27b-nvfp4-all/serve.sh) mounts it into the vLLM image under [Usage](#usage).

## Limitations

- **Prefill at 256K is about 8% slower than the FP8 release's** (61.0 s against 56.6 s); at 64K and 128K it is slightly faster. The whole request is still faster at every length measured.
- **Do not force the Marlin kernels.** With `--kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a4":"marlin","fp8_w8a8":"marlin"}}'` decode runs 3–4% more verification rounds per second, but prefill takes 1.5–1.6× as long as the FP8 release's, so a 256K request takes 18% longer than with the FP8 release (102.1 s; 145.9 / 138.4 / 122.5 tok/s, three benches).
- **Strict tool calling needs a fix on this vLLM build.** With strict tools, the `qwen3_coder` and `qwen3_xml` tool parsers use an xgrammar grammar that rejects the chat template's close for a tool without arguments, so such a call can run on in hidden whitespace up to the context limit ([vllm-project/vllm#50989](https://github.com/vllm-project/vllm/issues/50989)). This affects every model served with those parsers, not only this checkpoint. Mount the [strict-tool grammar fix](https://github.com/Percivio-Ltd/sm120-serving/tree/master/strict-tool-grammar) or send tools without `strict` until vLLM ships a fixed xgrammar.
- **Single-stream only.** Batched throughput was not measured.
- **Text only.** The vision encoder is NVIDIA's, unchanged and not evaluated. The benchmarks disable image and video input.
- **Limited quality evidence:** GSM8K, needle retrieval and a six-task internal agentic battery. Other tasks and languages were not tested.

## Usage

Tested with vLLM nightly `0.32.1rc1.dev435+gc50aca144` (`vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480`):

```bash
vllm serve Percivio/Qwen3.8-27B-NVFP4-all \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --mamba-ssm-cache-dtype float32 \
  --attention-config '{"backend":"FLASHINFER"}' \
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --reasoning-parser qwen3
```

On this GPU vLLM serves the NVFP4 layers with `FlashInferCutlassNvFp4LinearKernel`. For tool calling, add `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, and with strict tools mount the [strict-tool grammar fix](https://github.com/Percivio-Ltd/sm120-serving/tree/master/strict-tool-grammar). `AGENT=1` [`serve.sh`](https://github.com/Percivio-Ltd/sm120-serving/blob/master/qwen3.8-27b-nvfp4-all/serve.sh) does both and turns on prefix caching.

## How the numbers were measured

Single stream, greedy, thinking on, 2,048 output tokens per request. The prompts are built from the Django 5.1 source at 64K, 128K and 256K (76.7K / 126.1K / 223.7K prompt tokens), with three tasks per context (review, tests, explain). The servers ran without prefix caching, with `--gpu-memory-utilization 0.90 --seed 42 --generation-config vllm` and image and video input disabled; the FP8 release kept its default BF16 KV cache. Decode tok/s is output tokens over decode time, pooled over the three tasks. Request time is the client's end-to-end time, and prefill is the time to first token. Speed-ups compare each entry's mean with the FP8 release's mean on the same host. The quality scores of the FP8 release and of the research stack come from earlier runs of the same evaluations on other hosts. The serving script ([`serve.sh`](https://github.com/Percivio-Ltd/sm120-serving/blob/master/qwen3.8-27b-nvfp4-all/serve.sh)) and the benchmark ([`bench/`](https://github.com/Percivio-Ltd/sm120-serving/tree/master/bench)) are in [Percivio-Ltd/sm120-serving](https://github.com/Percivio-Ltd/sm120-serving).

## Reproduce the checkpoint

`reproduce/` holds the two build scripts and the SHA-256 manifest. Three independent builds from these pinned sources produced byte-identical files:

| Source | Revision |
|---|---|
| `nvidia/Qwen3.8-27B-NVFP4` | `482ca0f3832238542f8f5295dde86b5f22711d80` |
| `Qwen/Qwen3.8-27B-FP8` | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

```bash
python3 reproduce/build_hybrid.py q27-nvfp4 q27-fp8 q27-nvfp4-mtp8          # FP8 MTP layer, CPU
python3 reproduce/nvfp4_convert.py q27-nvfp4-mtp8 q27-bf16 q27-nvfp4-all --device cuda
```

Both ran inside the vLLM image above (PyTorch and safetensors only). The manifest also lists NVIDIA's `README.md` and `.quant_summary.txt` and an empty `.complete` marker, which this repository does not include, and NVIDIA's `.gitattributes`, which the Hub's default replaces here. Its `.quant_summary.txt` still describes the FP8 layers it replaced.

## License

Apache-2.0, the license of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) and [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4). `LICENSE` is the copy shipped with NVIDIA's checkpoint. The weights are modified as described above.
