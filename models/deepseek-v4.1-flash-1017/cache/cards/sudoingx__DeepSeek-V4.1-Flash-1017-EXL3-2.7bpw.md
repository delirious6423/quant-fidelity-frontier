---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- exl3
- vllm
- dgx-spark
- gb10
- quantized
---

# DeepSeek-V4.1-Flash-1017, EXL3 routed experts for two DGX Sparks

A quantization of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, made to be served at tensor parallel 2 on two NVIDIA DGX Sparks (GB10) by upstream vLLM with the patches and plugin in [github.com/sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41). Same recipe, plan settings and composition as the build of the launch checkpoint (`2cba9e42`, [sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-EXL3-2.7bpw)), re-run end to end on 1017's weights.

- The 15,360 backbone routed experts are EXL3 trellis tensors at 2, 3, 4 or 5 bits per weight, chosen per expert from measured sensitivity: 2.70 bpw in memory, 2.90 bpw per decoded token.
- The dense linears of all 40 layers are EXL3 too: `wq_a`, `wq_b`, `wo_b` and the shared expert at K4, `wo_a` at K5; the LM head at K4.
- The DSpark drafter: its dense linears at EXL3 K5 (`main_proj` stays FP8), its routed experts at EXL3 K3.
- The Engram tables are the official ones, read from disk at lookup time. Everything else (attention norms, routers, embeddings, the vision tower) is the checkpoint's own.
- This repo does not carry the two official Engram shards (203 GB); the serving instructions below download them from DeepSeek's 1017 repo into the same folder. 1017's Engram shards differ from the launch checkpoint's, so the 1017 ones are required.

This checkpoint loads only with that engine: vLLM main `6e517b15` + the repo's patch series + the `vllm_exl3_moe` plugin (the release image). Stock vLLM does not know its quant method (`deepseek_v41_exl3`). Other engines: not tested.

## Results

Measured on two DGX Sparks against native 1017. Method below.

| | this build | native 1017 |
|---|---|---|
| KLD vs native on the panel (mean, nats; the 8 domains weighted equally) | 0.0243 (0.0405) | reference |
| top-1 agreement | 95.47% | reference |
| perplexity on the panel | 1.6159 | 1.6005 |
| HumanEval+ (164), base / plus | 96.3 / 90.2 | 97.0 / 90.2 |
| MBPP+ (378), base / plus | 89.7 / 76.2 | 91.5 / 77.2 |
| IFEval (541), prompt strict / instruction strict | 89.1 / 92.4 | 90.4 / 93.6 |
| decode, `kit_bench.py` set b, code / prose / structured, DSpark on (k = 5, per-round depth) | 100.7 / 59.2 / 141.3 tok/s | |
| context per request | 262,144 tokens; needles 3/3 at 8K, 32K, 64K and 128K | |
| MemAvailable after load, head / worker | 8 / 9 GiB | |

No suite differs significantly from native 1017 (smallest paired p: MBPP+ base, 0.14).

For comparison on the same panel with the same reference implementation: LibertAIDAI's 1017 EXL3 2.9 bpw pack ([LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) @ `c178a012`, every EXL3 tensor it ships loaded) scores KLD 0.0304 (domains weighted equally 0.0454), top-1 94.76%, perplexity 1.6155. This build is closer to native on 7 of the 8 domains and with the domains weighted equally; LibertAIDAI's pack is closer on multilingual (0.0571 vs 0.0697) and on a panel of 300 IFEval prompts with native replies (KLD 0.1135 vs 0.1199).

Other 1017 packs made with this recipe, on the same panel: object-oriented-architects' 2.7 bpw pack (LibertAIDAI's tensors around this recipe's experts), measured here, 0.0232 / 95.65%, IFEval-reply 0.1201; and by their own numbers HeNryous's 2.75 bpw build with the dense set at K5 and the head at K6, 0.0201 / 95.96%, and 0xSero's 2.7 bpw build, 0.0304 / 94.95%. This build's experts with the dense set at K5 and the head at K6 measure 0.0219 / 95.75%; this build keeps K4 / K4 for decode speed (on the same boxes and session, that composition decoded at 95.1 / 56.7 / 135.3 tok/s against this build's 100.4 / 59.0 / 142.1; the recipe repo's BENCHMARKS.md has both variants).

## Files

| file | what |
|---|---|
| `config.json` | the official config plus `quantization_config`: `quant_method: deepseek_v41_exl3` and an `exl3_moe` block (plan file, decode kernel `coop`, `tp_layout: channels`, `exllamav3_version` 1.6.0, `head: exl3` with `head_k` 4, and `dense`: the K of each dense module, backbone and drafter) |
| `exl3_moe_plan.json` | the bit width of every routed expert: per layer, one character per expert; a `drafter` section for the drafter's experts |
| `model-exl3-L00-p0.safetensors` to `model-exl3-L39-p1.safetensors` | each layer's EXL3 routed experts as the two Sparks encoded them, in two halves; layers 16-18 in one file each (`-p0`), encoded on one Spark while the other was down |
| `model-exl3-dense.safetensors` | the EXL3 dense linears of all 40 layers: `wq_a`, `wq_b`, `wo_a` (8 head-group slices), `wo_b`, the shared expert |
| `model-exl3-head.safetensors` | the EXL3 K4 head, `head.{trellis,suh,svh,mul1}` |
| `model-exl3-mtp.safetensors`, `model-exl3-mtp-experts.safetensors` | the DSpark drafter's EXL3 dense linears (K5) and routed experts (K3) |
| `model-NNNNN-of-00046.safetensors` | every other official tensor, unchanged: attention norms and compressors, routers, embeddings, vision tower, the drafter's remaining tensors. Copied from the official shards minus the tensors EXL3 replaces. |
| `model.safetensors.index.json` | tensor to file map, including the Engram tensors in `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors` |
| `tokenizer.json`, `tokenizer_config.json`, `encoding/` | official, unchanged (1017 ships no `chat_template.jinja`; vLLM formats chats with its DeepSeek V4.1 encoder) |
| `LICENSE` | DeepSeek's MIT license, unchanged |
| `MANIFEST.sha256` | sha256 of every file in this repo except this card and the manifest itself, plus DeepSeek's two Engram shards (their official sha256) |

Download size: 196 GB (182.8 GiB), plus 203 GB for the two official Engram shards.

## Serve

Both Sparks need the full folder at the same absolute path, the serving image built from the GitHub repo, and passwordless ssh from a control machine. Full steps, image build and launcher settings: the [GitHub README](https://github.com/sudoingX/dgx-spark-deepseek-v41#quickstart).

```bash
# on both Sparks
hf download sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw --local-dir ~/models/dsv41-1017-exl3
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir ~/models/dsv41-1017-exl3
(cd ~/models/dsv41-1017-exl3 && sha256sum -c MANIFEST.sha256)

# from the control machine, in a clone of the GitHub repo
HEAD=<ssh-host-of-rank-0> WORKER=<ssh-host-of-rank-1> MODEL_DIR=/absolute/path/to/dsv41-1017-exl3 \
  ./scripts/launch/dsv41_tp2.sh up
```

The launcher runs `vllm serve` in one container per Spark with `--tensor-parallel-size 2 --nnodes 2`, NCCL over RoCE on the ConnectX link, `VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel` (Marlin for every MXFP8 linear), DSpark speculative decoding, Engram disk offload, FULL_DECODE_ONLY CUDA graphs, FlashInfer autotune off, the V4.1 tool-call and reasoning parsers, and text-only input by default. OpenAI API on the head node, port 8000, model name `dsv41`. The per-round depth table measured for this build: `VLLM_DSPARK_ROUND_MS=28.99,32.76,36.11,39.34,42.43,45.06`.

## How it was made

- **Source**: the official 1017 checkpoint, every shard checked against the Hub's sha256 metadata on both Sparks.
- **Calibration**: the launch build's set, unchanged: 766 sequences, 853,106 tokens (374 base sequences of native DeepSeek-V4.1-Flash conversations, exllamav3's standard text and random tokens, plus 392 instruction, chat, writing and multilingual conversations; no prompt shares an 8-gram with IFEval).
- **Allocation**: per-expert statistics and the per-layer sensitivity measured again on 1017, then the launch build's exact settings: importance = layer sensitivity x the expert's per-token share of the routed output, as a mix over domains; super experts stay at K5, layers 37-39 at K3 or more. Counts: K2 6,183, K3 7,637, K4 1,505, K5 35 (762 of the 15,360 experts on a different K than the launch build's plan).
- **Encode**: exllamav3 1.6.0 EXL3 trellis, mul1 codebook, one layer at a time on the hidden states coming out of the already-quantized layers, one Hessian per expert from its routed tokens weighted by routing weight, plus an all-token prior. The dense linears and the head from input Hessians on the same stream; the drafter's from Hessians captured in a serve of this build.
- **Selection**: the dense K4 set and the K4 head were kept by the same rules as the launch build, with the bars taken from LibertAIDAI's 1017 pack on the same panel: dense K4 within 10% of the K5 set and under that pack's KLD, the K4 head under 0.932 x it.
- **Assembly**: the encoded files hardlinked into one folder with the official non-expert tensors (`scripts/quantize/p4_link_variant.py` and the variant tools). Everything is reproducible from the GitHub repo; see its RECIPE.md.

## Quality method

- **Panel**: 199 native conversations from prompts disjoint from the calibration set, 254,846 scored assistant positions (the launch build's panel, sha256 `39e0db2f0b58aa432a512aed7b425e3feb1b424c9e02cc315be39bc9b7b71cf2`), scored against native 1017's logits.
- **Reference**: DeepSeek's own `inference/model.py`, streamed layer by layer on a Spark, for the native weights and for every quantized build alike; each build's own tensors are loaded into it, everything else stays official.
- **KLD**: mean KL(native || quantized) over scored positions, in nats. **Top-1**: share of positions where both argmax tokens match.
- **Suites**: HumanEval+, MBPP+ (evalplus) and IFEval, greedy, thinking off, 2,048 max tokens, paired per item against native 1017 served on the official FP8 weights. A suite fails if the drop is larger than 1 point at p < 0.05.
- **Speed**: bertholomus's `kit_bench.py` set b: single stream, greedy, thinking off, 384-token replies, median of 9 reps a prompt.

## Limits

- Two DGX Sparks at TP=2; there is no single-box configuration.
- Text-only by default; image input passed one basic test on this build (a generated two-color image described correctly) and has no broader measurement.
- Up to 2 sequences at a time, 262,144 tokens each; the 256K needle rung has not been run on this build yet.
- Quantization changes the model's outputs; the KLD and the suites above measure by how much. Everything else about the model's behaviour is DeepSeek's: read the [base model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

## License

MIT, as the base model. These weights are a derivative of DeepSeek-V4.1-Flash-1017; DeepSeek's `LICENSE` file ships unchanged. The serving code, patches and plugin in the GitHub repo are Apache-2.0.

## Credits

DeepSeek for the model and its reference implementation. turboderp for exllamav3 (MIT): the EXL3 quantizer and kernels. The vLLM project, with upstream PRs #56509 (zeenat28-ui) and #58560 (bluemelov1) carried as patches. bertholomus for `kit_bench.py`. LibertAIDAI's 1017 pack was measured as the baseline; nothing of theirs is in these weights.
