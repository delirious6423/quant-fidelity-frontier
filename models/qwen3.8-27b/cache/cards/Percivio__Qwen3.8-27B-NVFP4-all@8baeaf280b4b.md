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

Qwen3.8-27B with every linear layer of the language model in NVFP4, for single-stream decoding with MTP speculative decoding on one NVIDIA RTX PRO 6000 Blackwell (SM120).

It starts from NVIDIA's [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), which has its MLP and `lm_head` in NVFP4 but keeps the attention and Gated DeltaNet projections in per-tensor FP8. Two changes:

- **The 208 attention and Gated DeltaNet projections** (q/k/v/o in the 16 full-attention layers; `in_proj_qkv`, `in_proj_z` and `out_proj` in the 48 linear-attention layers) are re-quantized to NVFP4 from Qwen's BF16 weights ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)). Block scales come from an MSE search over amax fractions, as in ModelOpt's NVFP4 MSE calibrator. Projections that vLLM fuses share one global scale. Activation scales are NVIDIA's calibrated FP8 input scales divided by 6.
- **The MTP draft layer's seven projections**, which NVIDIA leaves in BF16, are replaced with the 128×128 block-FP8 weights of Qwen's official FP8 release ([Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)), in `mtp-fp8.safetensors`. Only the draft reads them.

Everything else, including the vision encoder, tokenizer and chat template, is byte-identical to NVIDIA's checkpoint. Total size is 18.4 GB.

## Speed on stock vLLM

Served with unmodified vLLM and the command under [Usage](#usage) (Marlin kernels, FP8 KV cache, MTP with five draft tokens) on one RTX PRO 6000 Blackwell Server Edition (AWS g7e.2xlarge), against the official FP8 release with MTP5 on the same host. **These are single benches;** repeat benches are running, and this table will be replaced by three-bench means.

| Context (prompt tokens) | 64K (76.7K) | 128K (126.1K) | 256K (223.7K) |
|---|---:|---:|---:|
| Official FP8 release (`Qwen/Qwen3.8-27B-FP8`), MTP5: decode tok/s | 77.6 | 71.8 | 73.3 |
| This checkpoint: decode tok/s | 145.6 | 136.2 | 120.7 |
| Speed-up in decode | 1.88× | 1.90× | 1.65× |
| Official FP8 release: full request, s (of which prefill) | 37.6 (11.2) | 51.5 (23.0) | 84.6 (56.7) |
| This checkpoint: full request, s (of which prefill) | 32.3 (18.2) | 51.0 (35.9) | 102.3 (85.4) |

## Quality

These results come from the research stack described below. With greedy decoding, speculative decoding returns the target model's own tokens. The draft-head patch can change the output only through floating-point differences in how tokens are batched for verification. The target runs the same Marlin kernels as in the stock-vLLM command above.

| | Official FP8 release (`Qwen/Qwen3.8-27B-FP8`) | This checkpoint |
|---|---:|---:|
| GSM8K, all 1,319 test problems (greedy, thinking on) | 1,231 | 1,231 (23 lost, 23 gained) |
| Needle retrieval in 64K–200K-token prompts | 18/18 | 18/18 |
| Six long-context agentic tasks, three runs each (our grader) | 16/18 | 18/18 |
| Reasoning loops in those runs | 0 | 0 |

The agentic tasks are two question-answering, two bug-repair and two proof-of-concept tasks over 128K–200K-token code contexts, graded by fixture tests and fact and citation checks. This is a small internal battery, not a public benchmark.

On stock vLLM with the command under [Usage](#usage), the checkpoint scored 190/200 on the first 200 GSM8K test problems (greedy, thinking on, 8,192-token cap; the research stack scored 189/200) and 18/18 on needle retrieval.

## Speed with our research stack

Our fastest lossless setup adds a vLLM patch that is not part of this release: the MTP draft uses a pruned FP8 output head (65,536 of 248,320 vocabulary rows). Measured on one RTX PRO 6000 Blackwell Server Edition (AWS g7e.2xlarge), means of three benches on fresh engine starts:

| Context (prompt tokens) | 64K (76.7K) | 128K (126.1K) | 256K (223.7K) |
|---|---:|---:|---:|
| Official FP8 release, stock vLLM, MTP5: decode tok/s | 80.5 | 74.7 | 69.2 |
| This checkpoint with the patched draft head: decode tok/s | 156.9 | 150.5 | 131.8 |
| Speed-up in decode | 1.95× | 2.01× | 1.91× |
| Official FP8 release: full request, s (of which prefill) | 36.5 (11.1) | 50.1 (22.7) | 85.3 (55.7) |
| This checkpoint: full request, s (of which prefill) | 31.3 (18.2) | 49.5 (35.9) | 100.9 (85.3) |

## Limitations

- **Prefill is about 1.5–1.6× slower than the official FP8 release** on this GPU with the Marlin kernels. At 256K the whole request (prefill plus 2,048 output tokens) takes longer than with the FP8 release. Workloads with long prompts and short answers may be better served by the FP8 release or NVIDIA's checkpoint.
- **Single-stream only.** Batched throughput was not measured.
- **Text only.** The vision encoder is NVIDIA's, unchanged and not evaluated. The benchmarks disable image and video input.
- **Limited quality evidence:** GSM8K, needle retrieval and a six-task internal agentic battery. Other tasks and languages were not tested.

## Usage

Tested with vLLM nightly `0.32.1rc1.dev435+gc50aca144` (`vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480`):

```bash
vllm serve Percivio/Qwen3.8-27B-NVFP4-all \
  --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a4":"marlin","fp8_w8a8":"marlin"}}' \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --mamba-ssm-cache-dtype float32 \
  --attention-config '{"backend":"FLASHINFER"}' \
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --reasoning-parser qwen3
```

Keep `--kernel-config`: without it, vLLM uses its default NVFP4 kernels on this GPU, a path we have not yet tested with this checkpoint.

## How the numbers were measured

Single stream, greedy, thinking on, 2,048 output tokens per request. The prompts are built from the Django 5.1 source at 64K, 128K and 256K (76.7K / 126.1K / 223.7K prompt tokens), with three tasks per context (review, tests, explain). The server ran without prefix caching, with `--gpu-memory-utilization 0.90 --seed 42 --generation-config vllm`. Decode tok/s is output tokens over decode time. Request time is the client's end-to-end time, and prefill is the time to first token. Speed-ups compare each entry's mean with the FP8 release's mean on the same host.

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
