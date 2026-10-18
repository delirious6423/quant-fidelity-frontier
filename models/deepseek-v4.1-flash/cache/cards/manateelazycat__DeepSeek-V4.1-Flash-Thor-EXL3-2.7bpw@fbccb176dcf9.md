---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
tags:
- exl3
- vllm
- dgx-spark
- gb10
- quantized
---

# DeepSeek-V4.1-Flash, EXL3 routed experts for two DGX Sparks

A quantization of [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) @ `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`, made to be served at tensor parallel 2 on two NVIDIA DGX Sparks (GB10) by upstream vLLM with the patches and plugin in [github.com/sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41).

- The 15,360 backbone routed experts are EXL3 trellis tensors at 2, 3, 4 or 5 bits per weight, chosen per expert from measured sensitivity: 2.70 bpw in memory, 2.90 bpw per decoded token.
- The dense linears of all 40 layers are EXL3 too: `wq_a`, `wq_b`, `wo_b` and the shared expert at K4, `wo_a` at K5; the LM head at K4.
- The DSpark drafter: its dense linears at EXL3 K5, its routed experts at EXL3 K3.
- The Engram tables are the official ones, read from disk at lookup time. Everything else (attention norms, routers, embeddings, the vision tower) is the checkpoint's own.
- This repo does not carry the two official Engram shards (203 GB); the serving instructions below download them from the official repo into the same folder.

This checkpoint loads only with that engine: vLLM main `6e517b15` + the repo's patch series + the `vllm_exl3_moe` plugin (the release image). Stock vLLM does not know its quant method (`deepseek_v41_exl3`). Other engines: not tested.

## Results

Measured on two DGX Sparks. Method below.

| | this build | native |
|---|---|---|
| KLD vs native on the panel (mean, nats; the 8 domains weighted equally) | 0.0243 (0.0398) | reference |
| top-1 agreement | 95.57% | reference |
| perplexity on the panel | 1.5167 | 1.4978 |
| HumanEval+ (164), base / plus | 97.0 / 90.9 | 97.6 / 92.1 |
| MBPP+ (378), base / plus | 91.3 / 78.6 | 91.0 / 77.5 |
| IFEval (541), prompt strict / instruction strict | 87.6 / 91.6 | 89.5 / 92.9 |
| decode, `kit_bench.py` set b, code / prose / structured, DSpark on (k = 5, per-round depth) | 103.0 / 58.3 / 141.3 tok/s | |
| context per request | 262,144 tokens; needles 3/3 at 8K, 32K, 64K, 128K and 256K (a 255,709-token prompt) | |
| MemAvailable after load, head / worker | 9 / 9 GiB | |

No suite differs significantly from native (IFEval prompt strict vs native: paired p = 0.13).

For comparison on the same panel with the same reference implementation: 0xSero's J268 build (2.43 bpw in memory, 2.77 per decoded token) scores KLD 0.0327 (domains weighted equally 0.0516), top-1 95.00%, perplexity 1.5277; Mia-AiLab's 2.9 bpw pack scores KLD 0.0295 (domains weighted equally 0.0443), top-1 94.99%, perplexity 1.5221. This build is closer to native on 7 of the 8 domains and with the domains weighted equally; Mia-AiLab's pack is closer on multilingual (per-domain table in the README).

## Files

| file | what |
|---|---|
| `config.json` | the official config plus `quantization_config`: `quant_method: deepseek_v41_exl3` and an `exl3_moe` block (plan file, decode kernel `coop`, `tp_layout: channels`, `exllamav3_version` 1.6.0, `head: exl3` with `head_k` 4, and `dense`: the K of each dense module, backbone and drafter) |
| `exl3_moe_plan.json` | the bit width of every routed expert: per layer, one character per expert; a `drafter` section for the drafter's experts |
| `model-exl3-L00-p0.safetensors` to `model-exl3-L39-p1.safetensors` | each layer's EXL3 routed experts in two halves, as the two Sparks encoded them: `layers.L.ffn.experts.E.w{1,2,3}.{trellis,suh,svh,mul1}` |
| `model-exl3-dense-L00.safetensors` to `model-exl3-dense-L39.safetensors` | each layer's EXL3 dense linears: `wq_a`, `wq_b`, `wo_a` (8 head-group slices), `wo_b`, the shared expert |
| `model-exl3-head.safetensors` | the EXL3 K4 head, `head.{trellis,suh,svh,mul1}` |
| `model-exl3-mtp.safetensors`, `model-exl3-mtp-experts.safetensors` | the DSpark drafter's EXL3 dense linears (K5) and routed experts (K3) |
| `model-NNNNN-of-00046.safetensors` | every other official tensor, unchanged: attention norms and compressors, routers, embeddings, vision tower, the drafter's remaining tensors. Copied from the official shards minus the tensors EXL3 replaces. |
| `model.safetensors.index.json` | tensor to file map, including the Engram tensors in `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors` |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `encoding/` | official, unchanged |
| `LICENSE` | DeepSeek's MIT license, unchanged |
| `MANIFEST.sha256` | sha256 of every top-level file of the assembled folder, the two Engram shards included |

Download size: 196 GB (182.8 GiB), plus 203 GB for the two official Engram shards.

## Serve

Both Sparks need the full folder at the same absolute path, the serving image built from the GitHub repo, and passwordless ssh from a control machine. Full steps, image build and launcher settings: the [GitHub README](https://github.com/sudoingX/dgx-spark-deepseek-v41#quickstart).

```bash
# on both Sparks
hf download sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw --local-dir ~/models/dsv41-exl3
hf download deepseek-ai/DeepSeek-V4.1-Flash model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027 --local-dir ~/models/dsv41-exl3
(cd ~/models/dsv41-exl3 && sha256sum -c MANIFEST.sha256)

# from the control machine, in a clone of the GitHub repo
HEAD=<ssh-host-of-rank-0> WORKER=<ssh-host-of-rank-1> MODEL_DIR=/absolute/path/to/dsv41-exl3 \
  ./scripts/launch/dsv41_tp2.sh up
```

The launcher runs `vllm serve` in one container per Spark with `--tensor-parallel-size 2 --nnodes 2`, NCCL over RoCE on the ConnectX link, `VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel` (Marlin for every MXFP8 linear), DSpark speculative decoding, Engram disk offload, FULL_DECODE_ONLY CUDA graphs, FlashInfer autotune off, the V4.1 tool-call and reasoning parsers, and text-only input by default. OpenAI API on the head node, port 8000, model name `dsv41`.

## How it was made

- **Source**: the official checkpoint, every shard checked against the Hub's sha256 metadata.
- **Calibration**: 766 sequences, 853,106 tokens. The base set: 374 sequences, 548,231 tokens of native DeepSeek-V4.1-Flash conversations (398,727 tokens; code, agentic tool loops, long-context reading, chat, math, knowledge, multilingual, writing; thinking and chat modes) plus exllamav3's standard calibration text (133,120) and random tokens (16,384), sha256 `83f5b3b78a9e220b468d5ab68c979aaedd4743c014d13a86532c781009cfe41f`. A supplement of 392 native conversations (304,875 tokens): instruction following with format, case, length and language constraints, chat, writing, and 30 languages; no prompt shares an 8-gram with IFEval.
- **Allocation**: importance of an expert = the measured sensitivity of its layer (KL at the output per unit of routed-output noise) x the expert's per-token share of the routed output, as a mix over domains (the eight panel domains weight 1, instruction following 1.5, exllamav3's standard text 0.5, random tokens 0.1). Super experts (20% or more of their layer's routed energy) stay at K5; layers 37-39, which the DSpark drafter reads, stay at K3 or more. Counts: K2 6,157, K3 7,689, K4 1,479, K5 35.
- **Encode**: exllamav3 1.6.0 EXL3 trellis, mul1 codebook, one layer at a time on the hidden states coming out of the already-quantized layers, one Hessian per expert from its routed tokens weighted by routing weight, plus an all-token prior for rarely routed experts. The dense linears and the head from input Hessians on the same stream; the drafter's from Hessians captured in a serve.
- **Assembly**: the encoded files hardlinked into one folder with the official non-expert tensors (`scripts/quantize/p4_link_variant.py` and the variant tools). Everything is reproducible from the GitHub repo; see its RECIPE.md.

## Quality method

- **Panel**: 199 native DeepSeek-V4.1-Flash conversations from prompts disjoint from the calibration set, 316,944 tokens, 254,846 scored assistant positions, sha256 `39e0db2f0b58aa432a512aed7b425e3feb1b424c9e02cc315be39bc9b7b71cf2`.
- **Reference**: DeepSeek's own `inference/model.py`, streamed layer by layer on a Spark, for the native weights and for every quantized build alike; each build's own tensors are loaded into it, everything else stays official.
- **KLD**: mean KL(native || quantized) over scored positions, in nats. **Top-1**: share of positions where both argmax tokens match.
- **Suites**: HumanEval+, MBPP+ (evalplus) and IFEval, greedy, thinking off, 2,048 max tokens, paired per item against the native model served on the official weights. A suite fails if the drop is larger than 1 point at p < 0.05.
- **Speed**: bertholomus's `kit_bench.py` set b: single stream, greedy, thinking off, 384-token replies, median of 9 reps a prompt.

## Limits

- Two DGX Sparks at TP=2; there is no single-box configuration.
- Text-only by default; image input on this build has not been tested.
- Up to 2 sequences at a time, 262,144 tokens each.
- Quantization changes the model's outputs; the KLD and the suites above measure by how much. Everything else about the model's behaviour is DeepSeek's: read the [base model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

## License

MIT, as the base model. These weights are a derivative of DeepSeek-V4.1-Flash; DeepSeek's `LICENSE` file ships unchanged. The serving code, patches and plugin in the GitHub repo are Apache-2.0.

## Credits

DeepSeek for the model and its reference implementation. turboderp for exllamav3 (MIT): the EXL3 quantizer and kernels. The vLLM project, with upstream PRs #56509 (zeenat28-ui) and #58560 (bluemelov1) carried as patches. bertholomus for `kit_bench.py`. 0xSero's and Mia-AiLab's DeepSeek-V4.1-Flash builds for two Sparks were measured as baselines; nothing of theirs is in these weights.
