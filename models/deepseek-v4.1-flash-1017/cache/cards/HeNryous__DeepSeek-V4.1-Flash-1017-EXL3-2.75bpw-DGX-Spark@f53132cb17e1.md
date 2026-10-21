---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- exl3
- exllamav3
- deepseek
- deepseek_v41
- moe
- quantized
- dgx-spark
- gb10
---

# DeepSeek-V4.1-Flash-1017 · EXL3 2.75 bpw (2× DGX Spark)

**TL;DR** — EXL3 quantization of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
with a **per-expert bit plan** (routed experts 2–5 bpw, 2.75 bpw on average), made with sudoingX's pipeline at a slightly
larger bit budget. **KLD 0.0201 / top-1 95.96 %** against the native 1017 (DeepSeek's reference code) on sudoingX's KLD
panel. Built and tested for tensor parallelism 2 on two DGX Spark / GB10 (128 GB each): 320K context, image input,
tool calling, DSpark speculative decoding.
Ready-to-run serving image and launch example:
[github.com/HeNryous/deepseek-v41-1017-dgx-spark](https://github.com/HeNryous/deepseek-v41-1017-dgx-spark).
The two Engram tables are **not** in this repo: take them from the 1017 base repo (see below).

| | |
|---|---|
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT) |
| Format | EXL3 trellis, `mul1` codebook ([exllamav3](https://github.com/turboderp-org/exllamav3)) |
| Routed experts | own bitrate per expert: 2 bpw 40.8 % · 3 bpw 43.8 % · 4 bpw 14.9 % · 5 bpw 0.4 % → **2.75 bpw** mean (3.10 bpw weighted by decode usage) |
| Attention, shared experts | EXL3 5 bpw |
| Output head | EXL3 6 bpw |
| DSpark drafter | dense EXL3 5 bpw, routed experts 2 bpw |
| Size | 198.8 GB in this repo (+ 203.1 GB Engram from the base repo) |
| KLD vs native 1017 | **0.0201**, top-1 agreement 95.96 % (sudoingX panel) |
| Vision tower | included, BF16 |
| Engram tables | not included: shards 47 + 48 of the 1017 base repo |

## Quality

KLD and top-1 agreement against the native 1017 checkpoint run through DeepSeek's reference code, measured with the KLD
panel of the [sudoingX recipe](https://github.com/sudoingX/dgx-spark-deepseek-v41) (199 conversations, 254,846 scored
positions). Panel perplexity 1.6005 (native) → 1.6119 (this quant).

| Model | Routed experts | KLD ↓ | top-1 ↑ |
|---|---|---|---|
| **this repo** | 2.75 bpw, per expert | **0.0201** | **95.96 %** |
| same, dense K4 / head K4 | 2.75 bpw, per expert | 0.0225 | — |
| [0xSero 1017 EXL3 2.7 bpw](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw) (as reported, same panel, 95 % CI 0.0268–0.0343) | 2.7 bpw | 0.0304 | 94.95 % |

By domain: code 0.0125 · math 0.0152 · agentic 0.0183 · long 0.0258 · knowledge 0.0382 · chat 0.0482 · writing 0.0514 ·
multilingual 0.0601.

Read the gap to 0xSero as budget and calibration, not method: it is the same pipeline, but we put more bits into the
target model (experts 2.75 vs 2.70 bpw, dense/head K5/K6 vs K4/K4; our K4 variant has his dense layout) and pay for it
with a smaller drafter; both repos are ~199 GB. Our calibration texts are sudoingX's, written by the previous V4.1
checkpoint like the panel's replies; his are native 1017 replies, which this panel may reward less. Other 1017 packs
report KL on other text sets, so their numbers don't compare with these.

The plan spends bits where a sensitivity pass measured the largest output error, weighted by how often the router picks
an expert (allocator cap 3.10 bpw per decoded token). Every expert's bitrate is listed in `exl3_moe_plan.json`.

## Tensor groups

| Group | Tensors | Stored as |
|---|---|---|
| Routed experts | `layers.N.ffn.experts.E.w1/w2/w3`, 384 per layer | EXL3 2/3/4/5 bpw per expert (`exl3_moe_plan.json`) |
| Shared expert | `layers.N.ffn.shared_experts.w1/w2/w3` | EXL3 5 bpw |
| Attention | `wq_a`, `wq_b`, `wkv`, `wo_a` (per-group slices), `wo_b` | EXL3 5 bpw |
| KV compressor, indexer, Engram projection | `attn.compressor.*`, `attn.indexer.*`, `engram.wkv` … | unchanged from the base checkpoint (BF16 / FP8) |
| Engram n-gram tables | `layers.1/14.engram.embed` | **not in this repo**, read from the base shards |
| Output head | `head` | EXL3 6 bpw |
| DSpark drafter, dense | attention, shared expert, projections of `mtp.0-2` | EXL3 5 bpw (rest as in the base) |
| DSpark drafter, routed experts | `mtp.0-2.ffn.experts.*`, 128 per block | EXL3 2 bpw |
| Token embedding, router, norms | `embed`, `ffn.gate.*`, RMSNorm weights | BF16 / FP32 as in the base |
| Vision tower + aligner | `vision.*`, `aligner.*` | BF16, unchanged |

## Engram tables (required)

DeepSeek-V4.1 reads two Engram n-gram tables (`layers.1.engram.embed`, `layers.14.engram.embed`, ~95 GiB each)
as-is (FP8, not re-quantized) from the base checkpoint. **Use the 1017 shards.** The older `deepseek-ai/DeepSeek-V4.1-Flash` has files with
the same names and the same sizes but different content (checked by sha256).

| File from deepseek-ai/DeepSeek-V4.1-Flash-1017 | Bytes | sha256 |
|---|---|---|
| `model-00047-of-00048.safetensors` | 101,535,150,936 | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| `model-00048-of-00048.safetensors` | 101,537,926,640 | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |

Download everything into **one folder**, on both nodes (or download once and copy over the fabric link):

```bash
DIR=~/models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark      # needs ~420 GB free
hf download HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark --local-dir "$DIR"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors --local-dir "$DIR"
```

`model.safetensors.index.json` in this repo already points at the two shards; nothing else to edit. The GitHub repo has
a one-line script that does both downloads and checks free space and file sizes.

## Serving

Needs the serving stack of the sudoingX recipe (v1.1 = `cc59c36`: vLLM pin `6e517b15` + patches 0001–0021 + the
`vllm_exl3_moe` plugin) plus our patches 0030–0074 (list and measurements in the GitHub repo). All of it is in one
image; build the plugin's JIT kernels once before the first start (building them next to the loaded weights can run the
node out of memory):

```bash
docker pull ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v22
mkdir -p ~/.cache/dsv41-serve && docker run --rm --gpus all -v $HOME/.cache/dsv41-serve:/root/.cache \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions -e MAX_JOBS=4 -e EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1 \
  --entrypoint python3 ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v22 \
  -c 'from vllm_exl3_moe import tf_experts as t, mrow, rdma, mhc, router, smoe, l2pf; t._ext(); mrow.ext(); rdma.ext(); mhc.ext(); router.ext(); smoe.ext(); l2pf.ext()'
```

Key settings (TP2 across two Sparks over RoCE; full launch example with the environment switches in the GitHub repo,
`serve/serve_example.sh`):

```
--tensor-parallel-size 2 --max-model-len 327680 --max-num-seqs 4 --max-num-batched-tokens 4096
--kv-cache-memory-bytes 4026531840        # 3.75 GiB per rank = 1,531,258 tokens with VLLM_DSV41_KV_PAIR=3 (0057 + 0060)
--long-prefill-token-threshold 1024       # with our patch 0031c: caps background requests only
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5]]}'
--scheduling-policy priority --enable-prompt-tokens-details
--limit-mm-per-prompt '{"image":4}' --mm-processor-cache-gb 0
--enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
```

Image v22 (21.10.) adds patches 0064, 0066, 0067 and 0071–0074 to v12; the weights and the quality numbers above are
unchanged. MoE calls of 33–63 rows run without host syncs (with the vLLM processes pinned to the Cortex-X925 cores:
4 streams sustained +9 %), the DSpark drafter round is cheaper (−0.66 % per decode round), and requests that the
client marks as background (vLLM's `priority` field) wait while an interactive request runs: next to one or two
background requests with 60K-token prompts the interactive stream decodes at 58.5 / 59.1 tok/s (alone 57.2), first
token 1.3–2.4 s.

Sampling as recommended by DeepSeek: temperature 1.0, top_p 1.0. Thinking is on by default; the deepseek_v41 reasoning
parser reads the mode only from `chat_template_kwargs`: `{"thinking": false}` or `{"reasoning_effort": "none" | "low" |
"high" | "xhigh" | "max"}` (a top-level `reasoning_effort` does not reach the parser).

Measured on 2× GB10 (TP2, GPU clock locked at 2000 MHz on both nodes, no other clients; medians of three suite / two
bench runs):

| | |
|---|---|
| decode, 1 stream, prose / code (Elevatormusic's TensorFold test scripts, suite) | 54.6 / 91.5 tok/s |
| decode, 4 streams aggregate, prose / code (same scripts, bench) | 94.4 / 155.9 tok/s |
| prefill, cold 8K (same scripts) | 2,203 tok/s |
| time to first token, cold 18.5K-token prompt ¹ | 8.4 s (~2,200 tok/s) |
| time to first token, cold 314K-token prompt ¹ | 152 s (2,066 tok/s); 3 of 3 needles found (10/50/90 % depth, T = 0) |
| KV pool at 3.75 GiB per rank (image v12, patches 0057 + 0060) | 1,531,258 tokens (4.67 full 320K requests at once; v5: 989,460) |

¹ Measured with the earlier image v4i; the later changes are mainly in decode-time paths.

Full comparison with the TensorFold recipe for this model (measured with its own test scripts at 2000 and 2400 MHz) in the
[GitHub README](https://github.com/HeNryous/deepseek-v41-1017-dgx-spark#expected-results); contexts above 320K are not
measured here (TensorFold: [forum thread 386125](https://forums.developer.nvidia.com/t/386125)).

## How it was made

With the quantization pipeline of the sudoingX recipe, on the two Sparks themselves:
calibration statistics (pass 0) → per-expert sensitivity → bit plan for 2.75 bpw (`plan_1017_275`) → EXL3 encode of all
15,360 routed experts with their own Hessians (pass 1) → dense pass at 5 bpw (chosen over 4 bpw by KLD) → DSpark
drafter (dense 5 bpw, experts 2 bpw; the drafter only changes acceptance, never the target's output) → assembly
with the Engram shards from the base. Drafter experts at 3 bpw gave +1.9 % tokens per step but −1 % steps per second
(decode +0.9 %, 3 streams −2.8 %), so 2 bpw stayed.

## Files

| Files | Content |
|---|---|
| `model-exl3-L00…L39-p0/p1.safetensors` | routed experts, EXL3 (two parts per layer) |
| `model-exl3-dense-L00…L39.safetensors` | attention and shared experts, EXL3 5 bpw |
| `model-exl3-head.safetensors` | output head, EXL3 6 bpw |
| `model-exl3-mtp.safetensors`, `model-exl3-mtp-experts.safetensors` | DSpark drafter (dense 5 bpw, experts 2 bpw) |
| `model-00001…00046-of-00046.safetensors` | everything kept from the base: embedding, vision tower, router, norms, compressor, indexer, Engram projection |
| `exl3_moe_plan.json` | bitrate of every routed expert (layers 0–39) and of the drafter experts |
| `model.safetensors.index.json`, `config.json` | index incl. the two Engram shards; model config |
| `tokenizer.json`, `tokenizer_config.json`, `encoding/` | tokenizer and DeepSeek's chat encoding (from the base repo) |

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model and reference code; [sudoingX](https://github.com/sudoingX/dgx-spark-deepseek-v41)
for the recipe, quantization pipeline, KLD panel and calibration data; [turboderp](https://github.com/turboderp-org/exllamav3)
for EXL3; ashhart, bertholomus and the TensorFold contributors for the RDMA ring, L2 prefetch and EXL3 prompt kernels;
0xSero for the 1017 reference point on the same panel; Elevatormusic for the TensorFold test scripts and run logs;
christopher_owen, jayleaton, shayanx45 and the DGX Spark forum for many measurements.

## License

MIT, inherited from DeepSeek-V4.1-Flash-1017 (see `LICENSE`). Community quantization, no warranty.
