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

# DeepSeek-V4.1-Flash-1017, EXL3 2.8 bpw for two DGX Sparks

A quantization of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170`, served at tensor parallel 2 on two NVIDIA DGX Sparks (GB10) by upstream vLLM with the patches and plugin of [github.com/sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) v2.0. Of the 1017 packs we measured, it is the closest to native 1017 on two panels of native replies, and on the v2.0 engine it decodes faster than TensorFold on code and structured output in three same-box sessions, with prose level.

- The 15,360 backbone routed experts are EXL3 trellis tensors at 2, 3, 4 or 5 bits per weight, chosen per expert from measured sensitivity: 2.80 bpw in memory, 3.10 bpw per decoded token (K2 5,709, K3 7,056, K4 2,553, K5 42).
- The dense linears of all 40 layers are EXL3: `wq_a`, `wo_a`, `wo_b` and the shared expert at K5, `wq_b` at K4; the LM head at K5.
- The DSpark drafter: its dense linears at EXL3 K5 (`main_proj` and `wo_a` stay FP8), its routed experts at K3, and its own K4 copy of the LM head (331 MB against the target's 414 MB K5 head; the target keeps K5).
- `dspark_conf_feat_head.npz`: the per-round draft depth model fitted to this pack's own acceptance (four weights per draft position over the confidence head's logit and each draft's logit margin and top-1 probability, from 11,357 fixed-depth rounds). The v2.0 launcher reads it from the model folder.
- The Engram tables are the official ones, read from disk at lookup time. Everything else (attention norms, compressors, routers, embeddings, the vision tower) is the checkpoint's own.
- This repo does not carry the two official Engram shards (203 GB); the serving steps below download them from DeepSeek's 1017 repo into the same folder.

This checkpoint loads only with the v2.0 engine: vLLM main `6e517b15` + the repo's patches 0001-0038 + the `vllm_exl3_moe` plugin (the image `install.sh` builds, `dsv41-serve:v5`). The v1.x image does not know the drafter's own head. Stock vLLM does not know the quant method (`deepseek_v41_exl3`). Other engines: not tested.

## Results

Measured on two DGX Sparks against native 1017. Method below.

| | this build | native 1017 |
|---|---|---|
| KLD vs native, our panel (mean, nats; the 8 domains weighted equally) | 0.0198 (0.0331) | reference |
| top-1 agreement, our panel | 95.94% | reference |
| KLD vs native, a second panel of native 1017 replies (domains weighted equally) | 0.0213 (0.0413) | reference |
| top-1 agreement, second panel | 95.97% | reference |
| KLD on a panel of 300 IFEval prompts with native replies | 0.0980 | reference |
| perplexity, our panel / second panel | 1.6117 / 1.5869 | 1.6005 / 1.5720 |
| HumanEval+ (164), base / plus | 97.0 / 89.6 | 97.0 / 90.2 |
| MBPP+ (378), base / plus | 90.7 / 77.8 | 91.5 / 77.2 |
| IFEval (541), prompt strict / instruction strict | 90.8 / 93.8 | 90.4 / 93.6 |
| decode, `kit_bench.py` set b, code / prose / structured, DSpark on (k = 5, per-round depth) | 103.8 / 57.8 / 144.1 tok/s | |
| context per request | 262,144 tokens; needles 3/3 at 8K, 32K, 64K and 128K | |
| MemAvailable after load, head / worker, vision on | 6 / 6 GiB (4 / 4 after a 128K prompt) | |

No suite differs significantly from native 1017 (paired McNemar p 0.61 to 1.00).

Every pack on the same panels, scored by the same reference implementation with each pack's own tensors loaded (all measured here):

| pack | our panel, KLD / top-1 | second panel, KLD / top-1 | IFEval-reply KLD |
|---|---|---|---|
| **this build** (2.80 bpw, 3.10 per decoded token) | **0.0198** / 95.94% | **0.0213** / 95.97% | 0.0980 |
| [HeNryous 2.75 bpw](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark) @ `ede209db` (dense K5, head K6) | 0.0201 / **95.96%** | 0.0216 / **95.97%** | **0.0974** |
| [object-oriented-architects 2.7 bpw](https://huggingface.co/object-oriented-architects/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-TensorFold) @ `cdb167c4` | 0.0232 / 95.65% | 0.0249 / 95.65% | 0.1201 |
| [our v1.2 2.7 bpw](https://huggingface.co/sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw) | 0.0243 / 95.47% | 0.0262 / 95.49% | 0.1199 |
| [LibertAIDAI 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) @ `c178a012` | 0.0304 / 94.76% | 0.0322 / 94.70% | 0.1135 |

This build has the lowest KLD on both panels and is closer to native than HeNryous's pack on 6 of the 8 domains of each (HeNryous is closer on math and knowledge on our panel, on math and long context on the second). Top-1 agreement is level with HeNryous's (95.94% vs 95.96%, 95.967% vs 95.969%), and HeNryous's pack is a little closer on the IFEval-reply panel (0.0974 vs 0.0980). HeNryous's and object-oriented-architects' packs are made with this recipe's pipeline.

Speed, one session on the same two Sparks, alternating fresh boots with the same memory preparation before each, the same client (set b, 9 reps a prompt): TensorFold `8e0966e2` serving object-oriented-architects' 2.7 bpw pack with lucataco's recipe settings, 98.4 / 58.0 / 141.3 and 98.6 / 58.4 / 140.9 tok/s; this build 103.9 / 57.9 / 144.0 and 103.6 / 57.7 / 144.1 tok/s. On the means, code +5.3%, structured +2.1%, prose -0.6%. A second session (the release gates, one boot each): TensorFold 97.7 / 57.5 / 140.4, this build 103.7 / 57.8 / 144.4 tok/s (+6.1%, +0.5%, +2.8%). On prose TensorFold verifies 1.99 tokens a round in 34.1-34.6 ms, this build 2.10 tokens in 36.3 ms. TensorFold loads with 12-13 GiB free a box and serves a 1M window. `8e0966e2` is the commit lucataco's 1017 recipe pins. A third session against TensorFold's newest DeepSeek TP2 commit (`af26d42b`, its v0.6 release plus two fixes), alternating fresh boots: TensorFold 98.6 / 58.5 / 140.7 and 98.2 / 57.4 / 140.3, this build 104.2 / 57.8 / 143.6 and 103.6 / 57.7 / 143.1 tok/s: code +5.6%, structured +2.0%, prose -0.3%.

## Files

| file | what |
|---|---|
| `config.json` | the official config plus `quantization_config`: `quant_method: deepseek_v41_exl3` and an `exl3_moe` block (plan file, decode kernel `coop`, `tp_layout: channels`, `exllamav3_version` 1.6.0, `head: exl3` with `head_k` 5, `dense`: the K of each dense module, backbone and drafter, and `drafter_head_k` 4) |
| `exl3_moe_plan.json` | the bit width of every routed expert: per layer, one character per expert; a `drafter` section for the drafter's experts |
| `model-exl3-L00-p0.safetensors` to `model-exl3-L39-p1.safetensors` | each layer's EXL3 routed experts as the two Sparks encoded them, in two halves; layers 16-18 in one file each (`-p0`) |
| `model-exl3-dense-L00.safetensors` to `model-exl3-dense-L39.safetensors` | each layer's EXL3 dense linears: `wq_a`, `wq_b`, `wo_a` (8 head-group slices), `wo_b`, the shared expert |
| `model-exl3-head.safetensors` | the EXL3 K5 head, `head.{trellis,suh,svh,mul1}` |
| `model-exl3-mtp.safetensors`, `model-exl3-mtp-experts.safetensors`, `model-exl3-mtp-head.safetensors` | the DSpark drafter's EXL3 dense linears (K5), routed experts (K3) and its own K4 head (`mtp.head.*`) |
| `model-NNNNN-of-00046.safetensors` | every other official tensor, unchanged: attention norms and compressors, routers, embeddings, vision tower, the drafter's remaining tensors |
| `model.safetensors.index.json` | tensor to file map, including the Engram tensors in `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors` |
| `dspark_conf_feat_head.npz` | the draft depth model (above); data and fitter in the GitHub repo (`data/dspark/`, `scripts/dspark/fit_draft_features.py`) |
| `tokenizer.json`, `tokenizer_config.json`, `encoding/` | official, unchanged (1017 ships no `chat_template.jinja`; vLLM formats chats with its DeepSeek V4.1 encoder) |
| `LICENSE` | DeepSeek's MIT license, unchanged |
| `MANIFEST.sha256` | sha256 of every file in this repo except this card and the manifest itself, plus DeepSeek's two Engram shards (their official sha256) |

Download size: 204 GB (189.9 GiB), plus 203 GB for the two official Engram shards.

## Serve

Both Sparks need the full folder at the same absolute path, the v2.0 serving image built from the GitHub repo, and passwordless ssh from a control machine. Full steps, image build and launcher settings: the [GitHub README](https://github.com/sudoingX/dgx-spark-deepseek-v41#quickstart).

```bash
# on both Sparks
hf download sudoingx/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw --local-dir ~/models/dsv41-1017-exl3-28
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir ~/models/dsv41-1017-exl3-28
(cd ~/models/dsv41-1017-exl3-28 && sha256sum -c MANIFEST.sha256)

# from the control machine, in a clone of the GitHub repo (v2.0)
HEAD=<ssh-host-of-rank-0> WORKER=<ssh-host-of-rank-1> MODEL_DIR=/absolute/path/to/dsv41-1017-exl3-28 \
  ./scripts/launch/dsv41_tp2.sh up
```

The launcher runs `vllm serve` in one container per Spark with `--tensor-parallel-size 2 --nnodes 2`, NCCL over RoCE on the ConnectX link, Marlin for every MXFP8 linear, DSpark speculative decoding with the per-round depth from this folder's `dspark_conf_feat_head.npz` and the round-cost table measured on this build (`27.6,31.9,35.6,38.9,42.2,44.8` ms, the v2.0 default), Engram disk offload, FULL_DECODE_ONLY CUDA graphs, FlashInfer autotune off, the V4.1 tool-call and reasoning parsers, and text-only input by default. OpenAI API on the head node, port 8000, model name `dsv41`.

## How it was made

- **Source**: the official 1017 checkpoint, every shard checked against the Hub's sha256 metadata on both Sparks.
- **Calibration**: the set of our earlier builds, unchanged: 766 sequences, 853,106 tokens (374 base sequences of native DeepSeek-V4.1-Flash conversations, exllamav3's standard text and random tokens, plus 392 instruction, chat, writing and multilingual conversations; no prompt shares an 8-gram with IFEval).
- **Allocation**: v1.2's per-expert statistics and per-layer sensitivity on 1017 and v1.2's settings (importance = layer sensitivity x the expert's per-token share of the routed output, super experts at K5, layers 37-39 at K3 or more), with a 2.80 bpw memory budget and a cap of 3.10 bpw per decoded token in place of 2.70 / 2.90. The allocator reproduces this plan exactly from the files in the GitHub repo.
- **Encode**: exllamav3 1.6.0 EXL3 trellis, mul1 codebook, one layer at a time on the hidden states coming out of the already-quantized layers, one Hessian per expert from its routed tokens weighted by routing weight, plus an all-token prior. The experts came from three runs of that encoder: v1.2's full encode, a run at 2.80 / 3.00 that encoded the 1,536 experts gaining a bit, and this plan's run, which encoded 893 experts and took the rest from the first build holding them at their new K. The dense linears and the head from input Hessians on v1.2's stream; the drafter's dense linears and experts from Hessians captured in a serve (v1.2's files); its head is v1.2's K4 encode of the same LM head.
- **Selection**: dense K5 with the K6 head measured 0.0194 on these experts; `wq_b` at K4 and the head at K5 cost 0.0004 of KLD for shorter decode rounds, which kept the build under 0.0200 on our panel.
- **Draft depth model**: a fixed-depth logging session on this build (patch 0034), then `fit_draft_features.py --export`; the fitter reproduces the shipped file from the shipped log.

## Quality method

- **Panels**: our panel, 199 native conversations from prompts disjoint from the calibration set, 254,846 scored assistant positions (sha256 `39e0db2f0b58aa432a512aed7b425e3feb1b424c9e02cc315be39bc9b7b71cf2`); the second panel, the same 199 prompts answered by native 1017 through OpenRouter, 237,457 scored positions (sha256 `f22a47c238f0926ffa22c2b5f42f9276eabdbec2358949ac6bc3d50420fe0da0`); the IFEval-reply panel, 300 IFEval prompts with native replies, 116,318 positions. Every panel is scored against native 1017's logits.
- **Reference**: DeepSeek's own `inference/model.py`, streamed layer by layer on a Spark, for the native weights and for every quantized pack alike; each pack's own tensors are loaded into it, everything else stays official.
- **KLD**: mean KL(native || quantized) over scored positions, in nats. **Top-1**: share of positions where both argmax tokens match.
- **Suites**: HumanEval+, MBPP+ (evalplus) and IFEval, greedy, thinking off, 2,048 max tokens, paired per item against native 1017 served on the official FP8 weights. A suite fails if the drop is larger than 1 point at p < 0.05.
- **Speed**: bertholomus's `kit_bench.py` set b: single stream, greedy, thinking off, 384-token replies, median of 9 reps a prompt.

## Limits

- Two DGX Sparks at TP=2; there is no single-box configuration.
- Text-only by default; image input loads with `TEXT_ONLY=0` (the numbers above were measured that way) and has no broader measurement.
- Up to 2 sequences at a time, 262,144 tokens each; the 256K needle rung has not been run on this build yet.
- MemAvailable after load is lower than v1.2's (6 / 6 GiB with vision on against 8 / 9, and 4 / 4 after a 128K prompt), from the heavier experts, dense set and head; keep other GPU work off both boxes while it serves.
- Quantization changes the model's outputs; the KLD and the suites above measure by how much. Everything else about the model's behaviour is DeepSeek's: read the [base model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

## License

MIT, as the base model. These weights are a derivative of DeepSeek-V4.1-Flash-1017; DeepSeek's `LICENSE` file ships unchanged. The serving code, patches and plugin in the GitHub repo are Apache-2.0.

## Credits

DeepSeek for the model and its reference implementation. turboderp for exllamav3 (MIT): the EXL3 quantizer and kernels. The vLLM project, with upstream PRs #56509 (zeenat28-ui) and #58560 (bluemelov1) carried as patches. bertholomus for `kit_bench.py`; TensorFold (Apache-2.0) for designs credited in the GitHub repo's NOTICE. HeNryous for the RoPE-table and indexer-buffer memory patches the v2.0 engine reworks (patches 0024 and 0025). HeNryous's, object-oriented-architects' and LibertAIDAI's packs were measured as baselines; nothing of theirs is in these weights.
