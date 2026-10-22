---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
- en
- zh
- ko
tags:
- deepseek
- deepseek-v4.1
- gguf
- vector-quantization
- vq
- imatrix
- importance-matrix
- calibration
- mixture-of-experts
- engram
- mtp
- speculative-decoding
- vision
- 1m-context
- dgx-spark
- gb10
- ds4
- ds4-dfm-rs
---

# DeepSeek-V4.1-Flash-1017 iVQ GGUF

**The [VQ release](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF) of DeepSeek's October 17 V4.1 Flash with its routed experts refitted to measured activations: the same layout, size and runtime as VQ.**
The VQ release, like YoungAi's recipe it follows, fits its codebooks to the weights
alone. iVQ first runs the original model over 3.65M calibration tokens. It records
how strongly real inputs drive each input channel of each of the 384 experts in all
40 layers, then refits the shared codebooks, the code assignments and the row gains
so that the error lands where the activations are small. This is the VQ counterpart
of llama.cpp's importance matrix (imatrix), hence the "i".

Everything that fixes the file layout is held fixed: 13-bit indices on layers 0–13
and 12-bit on 14–39, one codebook per layer, and every byte outside the 40
routed-expert blobs. The comparison with VQ therefore measures the calibration-weighted
fit alone ([RECIPE.md](RECIPE.md) lists what differs in the fit), and the files drop
into a setup that already serves VQ.

> **Status, 2026-10-22:** converted, verified and uploaded. The base has VQ's exact
> tensor inventory and byte layout (1,000 tensors, 113,556,639,424 bytes; 8.36 GB
> outside the expert blobs compared byte for byte) and passes the ds4-dfm-rs
> DeepSeek V4.1 loader check (`bind: slots=1000 bound=1000 required-missing=0`).
> **Measured against the original FP4/FP8 model:** with the original's own inputs,
> the routed-expert output error of a typical token falls by **7% per layer**
> (13-bit layers 10%, 12-bit layers 6%). End to end, teacher-forced over
> 58,803 held-out tokens, the gain is smaller: KL **0.369 → 0.350 (−4.9%)**, the routed
> expert sets stay closer to the original's in every layer, and next-token agreement
> is unchanged at 82.4% → 82.5% ([below](#measured-against-the-original)).
> Serving on DGX Spark has not been measured for these files yet; the bytes per
> token are VQ's, so speed should match.

> [!WARNING]
> **Experimental low-bit release.** The routed experts, which hold most of the
> weights, are stored at about 1.6 bits per weight, far below common 4-bit
> GGUF quants. You may see failures the source model rarely shows: reasoning
> that loops or never reaches an answer, text that breaks mid-character (shown
> as `�`) or switches language mid-word, and weaker answers in general. This
> is exploratory quantization research, not a drop-in replacement for the
> source model.
>
> **Problems or odd output?** Feel free to post them in the
> [Community tab](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF/discussions),
> even as rough notes. The prompt, sampling settings and runtime commit help
> most.

## At a glance

| | |
|---|---|
| Source | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) at `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Starting point | [Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF) (YoungAi `vq8sh14-q4k-mtpnative`) |
| What changes | The 40 routed-expert VQ blobs: codebooks, indices and row gains, fitted with calibration statistics |
| What does not change | Bit allocation (layers 0–13 13-bit K=8192, layers 14–39 12-bit K=4096), one E4M3 codebook per layer shared by all 384 experts and gate/up/down, F16 row gains, file layout and size. The q4_K skeleton, the MTP draft towers, the vision sidecar and the Engram tables are byte-identical to the VQ release |
| Calibration | 3,095 sequences, 3,653,893 tokens: English 51%, Korean 14%, Chinese 13%, code 13%, Spanish and Japanese about 2% each, DeepSeek agent prompts 5%. Raw text 41%, thinking chats 31%, non-thinking chats 28%. Up to 31,721 tokens per sequence |
| Fidelity (reference forward, vs the original) | Routed-expert error per layer, typical token, **−7%** vs VQ; KL 0.369 → 0.350 (−4.9%); same next token 82.4% → 82.5% ([details](#measured-against-the-original)) |
| Size | Base GGUF 113,556,639,424 bytes (40 parts), the same as VQ. Routed experts about 1.6 bits per weight |
| Context | 1,048,576 tokens, from the model metadata |
| Runtime | [ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs) on DGX Spark (CUDA), the same as VQ. YoungAi `ds4-server` reads the same layout |

## What the importance matrix changes

A routed expert's weight matrix is cut into 8-weight vectors along each row. Each
row is divided by its gain, and every vector is replaced by the nearest of K
shared codewords. The VQ release picks codewords that minimize the plain squared
error of the weights:

$$\textstyle\sum_d (v_d - c_d)^2$$

The output error of an expert is the error of `W·x`, not the error of `W`. A weight
error on input channel `k` therefore costs as much as that channel's activation
energy. iVQ minimizes a diagonal estimate of the output error instead:

$$\textstyle\sum_d \omega_{d}\,(v_d - c_d)^2,\qquad \omega_{r,k} = g_r^2 \cdot h_{e,p}[k]^{0.75}$$

- `h[k]` is the measured per-token mean of `x_k²` for the expert's gate and up
  inputs, and of `a_k²` (`a = silu(gate)·up`) for its down input. It is
  normalized per tensor. Rare experts are shrunk toward the layer mean (prior 256
  tokens), and every weight is floored at 2% of the mean.
- The same weights enter all three steps. The weighted Lloyd fits the shared
  codebook, each vector is assigned to the codeword nearest under its expert's
  column weights, and each row gain is refit by importance-weighted least
  squares.

The method and its tools are described in
[ds4-dfm-rs #108](https://github.com/Baekpica/ds4-dfm-rs/pull/108)
(`docs/vq-calibration.md`). They were first used for
[A.X-K2-VQ-GGUF](https://huggingface.co/Baekpica/A.X-K2-VQ-GGUF), whose ablation
picked this weighting over unweighted, row-energy, frequency-weighted and
output-space variants. They were then used for the
[Beam-1.0](https://huggingface.co/Baekpica/Beam-1.0-501B-A23B-VQ-GGUF) drop-in
re-conversion. This release uses the same weighting with no further search.

What this does to the expert weights (mean over the layers of each group; every
layer sums all 384 experts × 3 matrices; the activation-weighted error weights each
weight's error by its input channel's measured energy and its expert's routed tokens):

| MoE layers | Plain weight error (rel. MSE), VQ → iVQ | Activation-weighted error, VQ → iVQ |
|---|---:|---:|
| 0–13 (13-bit, K=8192) | 0.144 → 0.148 (+3.0%) | 0.144 → **0.131 (−8.6%)** |
| 14–39 (12-bit, K=4096) | 0.171 → 0.173 (+1.7%) | 0.168 → **0.121 (−27.7%)** |
| All 40 | 0.161 → 0.165 (+2.1%) | 0.159 → **0.125 (−21.7%)** |

The plain weight error goes up a little, by design: precision moves away from input
channels that real inputs barely drive.

## Measured against the original

All numbers come from the reference forward in `tools/vq_calib/ds41_ref.py`. It
follows DeepSeek's `inference/model.py` on the prefill path (hyper-connections,
Engram, compressed sparse attention with its indexer, the sqrtsoftplus router) and
emulates its FP8/FP4 kernels; it ran on Runpod 8× A100 80GB. The original is the
FP4/FP8 checkpoint. VQ and iVQ replace only the routed experts with their dequantized
blobs, so the q4_K skeleton, identical in both, is not part of these numbers. The
held-out set is 26 cases, 58,803 tokens: WikiText-2, English/Korean/Chinese/Japanese/
Spanish web, math, code files, chat in all reasoning modes, DSML tool calls, SWE,
competitive programming and two 8K documents. None of it was used for calibration or
for any choice.

| Teacher-forced vs the original (58,803 held-out tokens) | Numerical floor¹ | VQ | **iVQ** | Change |
|---|---:|---:|---:|---:|
| Routed-expert output error per layer, per token (median, mean of 40 layers; original's inputs and routes) | — | 0.309 | **0.288** | **−7%** |
| Routed-expert set overlap with the original (mean of layers 1–39) | — | 75.1% | **76.5%** | **+1.4 pt** |
| Final hidden state cosine | — | 0.777 | **0.798** | **+0.021** |
| KL, all positions | 0.026 | 0.369 | **0.350** | **−4.9%** |
| KL, positions the model generates² | 0.010 | 0.295 | 0.291 | −1.4% |
| Same next token as the original, all positions | 96.2% | 82.41% | 82.47% | +0.06 pt |
| NLL of the actual text, generated positions (original 1.236) | 1.234 | 1.334 | 1.344 | +0.010 |

¹ The same original model run twice, the held-out cases batched differently: the
GEMM row counts change and with them the rounding. This is the reference forward's
own noise; differences below it mean nothing.
² Raw-text cases, and the assistant turns of chat cases. Post-trained chat models are
never trained to predict system prompts, tool schemas or user turns, and the original
gives some of those tokens probabilities near 1e-8, so a quantized model can even have
lower NLL there. KL and agreement over those positions say little about generation.

**Per layer.** With the original's own inputs and routes, the routed output
`Σ w_k E_k(x)` of every held-out token is computed with the original experts, VQ's
and iVQ's, and its relative error `‖Δy‖² / ‖y‖²` taken per token:

| MoE layers | Per token (median), VQ → iVQ | Change | Layers improved | Energy-weighted, VQ → iVQ |
|---|---:|---:|---:|---:|
| 0–13 (13-bit) | 0.239 → **0.215** | **−10%** | 14 of 14 | 0.220 → 0.164 (−26%) |
| 14–39 (12-bit) | 0.348 → **0.327** | **−6%** | 26 of 26 | 0.345 → 0.075 (−78%) |
| All 40 | 0.309 → **0.288** | **−7%** | 40 of 40 | 0.302 → 0.106 (−65%) |

The per-token median counts every token once. The energy-weighted ratio
(`Σ‖Δy‖² / Σ‖y‖²`) is dominated by a few tokens whose routed outputs are orders of
magnitude larger than the rest, and calibration fixes those most, so it overstates
what a typical token gains; it is shown for reference. By the per-token measure the largest drops are on layers 0 (−29%), 39 (−18%), 1 (−18%) and the
smallest on 30 (−3%), 27 (−3%), 29 (−3%); 40 of 40 layers improve.

**End to end** the per-layer gain shrinks further: KL falls by 4.9% over all
positions and by 1.4% over the positions a model generates, next-token agreement does
not move, and the NLL of the actual text on generated positions is 0.01 nat worse.
By kind of text:

| Held-out text | Tokens | KL, VQ → iVQ | Change | Same top-1, VQ → iVQ | NLL, original → VQ → iVQ |
|---|---:|---:|---:|---:|---:|
| WikiText-2 | 4,096 | 0.505 → **0.487** | −4% | 78.3% → 78.1% | 1.12 → 1.43 → 1.41 |
| Web en / ko / zh / ja / es | 10,346 | 0.379 → 0.404 | +6% | 75.0% → 74.6% | 1.90 → 2.05 → 2.10 |
| Math web | 1,476 | 0.213 → **0.198** | −7% | 92.3% → 93.4% | 0.09 → 0.29 → 0.28 |
| Code files (Python, C) | 4,094 | 0.113 → **0.108** | −5% | 96.4% → 96.1% | 0.10 → 0.17 → 0.16 |
| Chat ko / zh / ja | 6,047 | 0.266 → **0.237** | −11% | 84.2% → 84.9% | 1.11 → 1.10 → 1.09 |
| Reasoning (think) | 6,142 | 0.158 → **0.140** | −11% | 88.1% → 88.2% | 0.87 → 0.81 → 0.83 |
| Instruction following | 1,517 | 0.333 → 0.333 | +0% | 80.4% → 80.9% | 1.01 → 0.98 → 1.01 |
| Tool calls / agents (DSML) | 2,659 | 1.297 → **1.025** | −21% | 71.1% → 72.3% | 2.40 → 1.26 → 1.41 |
| SWE agentless | 1,923 | 1.067 → **1.003** | −6% | 71.4% → 72.8% | 2.21 → 1.35 → 1.37 |
| Competitive programming (think) | 4,095 | 0.387 → **0.374** | −3% | 79.8% → 81.1% | 1.19 → 1.03 → 1.01 |
| Long 8K (web, finance) | 16,382 | 0.288 → 0.287 | −0% | 84.9% → 84.3% | 1.23 → 1.29 → 1.29 |

Chat, reasoning, tool calls and code improve; Chinese and Japanese web text gets
worse end to end. Per token, with the original's inputs, the routed-expert error on
those cases is lower too (median over all layers, VQ → iVQ: zh_web_a 0.325 → 0.310, zh_web_b 0.322 → 0.313, ja_web 0.317 → 0.301, en_web 0.313 → 0.293), so the regression
comes from how the remaining error propagates on that text. Chinese made up 12.8% of
the calibration tokens (web 5.8%, chat 7.0%).

**Why the gain is small here.** DeepSeek-V4.1 has extreme massive activations in its
routed experts. From layer 10 on, the top 1% of held-out tokens carry 94–100% of the
routed-output energy, and the largest token's output energy is up to 11 million times
the median token's. On layers 20–25 the 26 sequence-initial tokens (BOS) alone carry
about 95%. The importance matrix sums `x²` and `a²` over tokens, so it mostly describes
those few tokens, and the fit protects them: the energy-weighted error falls by 65%,
the typical token's by 7%. That explains the gap between the two per-layer
measures, but not why the typical token gains so little: two follow-up conversions,
measured the same way, land at the same place. One weights every token equally in the
statistics, the other also uses a rebalanced, more multilingual calibration set (3.75M
tokens, no domain above 8%, Chinese 16%, Japanese 8%). Both give −7.0% and −6.8% per
layer for the typical token, KL 0.349 and 0.352, and the same Chinese and Japanese web
regression, so these files were kept. The remaining suspect is the diagonal
approximation itself: the importance-weighted weight error falls by 22%, the actual
per-token output error by 7%, which points to correlated input channels that only a
covariance-aware fit (GPTQ-style) would account for.

## Calibration data

The set re-uses the families of the A.X-K2 and Beam calibration mixes and of
[Baekpica/Solar-Open2-120B-A15B-REAM-148E-Healing-Mix](https://huggingface.co/datasets/Baekpica/Solar-Open2-120B-A15B-REAM-148E-Healing-Mix).
Chinese is given more weight here because Chinese is one of DeepSeek's main
languages. Every chat is rendered with the model's own `encoding/encoding.py`,
including thinking mode and V4.1 DSML tool calls. Sources are pinned by revision.

| Domain | Tokens | Share | Source |
|---|---:|---:|---|
| English web | 451,787 | 12.4% | fineweb-edu |
| Instruction following | 356,264 | 9.8% | Nemotron IF-Chat-v2, Cascade-2 IF |
| Code files | 345,504 | 9.5% | github-code-clean |
| Korean chat (math, STEM, code) | 272,050 | 7.4% | Nemotron-SFT-Multilingual-v2 ko |
| Chinese chat | 255,693 | 7.0% | Chinese-DeepSeek-R1-Distill-110k, Multilingual-v1 zh, COIG-CQIA |
| Korean / Chinese web | 252,836 / 211,518 | 6.9 / 5.8% | FineWeb-2 |
| Tool calling (DSML) | 211,250 | 5.8% | Nemotron-SFT-Agentic-v2 |
| DeepSeek agent prompts | 180,765 | 4.9% | ds4-dfm-rs imatrix prompts, re-rendered for V4.1 |
| Reasoning: math, general, science, code | 432,724 | 11.8% | Nemotron-Cascade-SFT Stage 1/2 |
| Long finance (10-K contexts) | 139,943 | 3.8% | Nemotron-SpecializedDomains-Finance-v1 |
| Math web | 137,539 | 3.8% | finemath-4plus |
| SWE agentless | 134,510 | 3.7% | Nemotron-SFT-SWE-v2 |
| Competitive programming | 116,462 | 3.2% | Nemotron-SFT-Competitive-Programming-v2 |
| Japanese and Spanish web and chat | 155,048 | 4.2% | FineWeb-2, Nemotron Multilingual |

No domain exceeds 12.4% of the tokens. YoungAi found that calibrating on a narrow
domain costs general text, so the mix stays broad. 19.6% of the tokens are in
sequences of 4,096+ tokens, so the compressed sparse attention and its indexer
see long contexts. WikiText is held out of calibration. The held-out set was
reserved first, and any calibration row sharing a 64-token window with it
(beyond template boilerplate) was dropped.

## Files

The file names and layout are the VQ release's, so a directory or script set up
for that release serves this one once its path is changed. Do not mix files
between the two directories. The part names are the same, and `SHA256SUMS`
catches a mix.

| Path | Contents | Size |
|---|---|---:|
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf.part01-of-40` … `part40-of-40` | iVQ base GGUF, including the MTP drafter, as 40 byte-range parts (24 × 2,838,915,986 B, then 16 × 2,838,915,985 B) | 113,556,639,424 B |
| `DeepSeek-V4.1-Flash.vision.gguf` | Vision sidecar for `--vision-gguf` (identical to the VQ release) | 971,066,016 B |
| `deepseek-engram/model-00047-of-00048.safetensors` | 1017 Engram shard (official file, unmodified) | 101,535,150,936 B |
| `deepseek-engram/model-00048-of-00048.safetensors` | 1017 Engram shard (official file, unmodified) | 101,537,926,640 B |
| `SHA256SUMS` | SHA-256 of the assembled base, every part and every other file | |
| [`RECIPE.md`](RECIPE.md) | What changes against VQ, statistics, commands, pinned inputs | |
| [`SOURCE.md`](SOURCE.md) | Provenance and licenses | |
| `LICENSE`, `LICENSE-ds4-dfm-rs`, `LICENSE-DeepSeek` | YoungAi tools and format (MIT, with the GGML notice), ds4-dfm-rs calibration tools (MIT), DeepSeek weights (MIT) | |

SHA-256 of the assembled base (all parts are in `SHA256SUMS`):

```
746fb52dac7de7b4ef9ffb32dbc103e2439a89f52aad3757a2ed595b75a8f4c0  DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
```

The download is about 318 GB. Plan for about 114 GB more while the parts are
assembled, or use the append-and-delete loop below. If you already have the VQ
release, only the 40 parts are new. The vision sidecar and the Engram shards are
the same files.

## Quick start on DGX Spark

```sh
# 1. Download and verify (parts, vision sidecar, Engram, checksums)
hf download Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF --local-dir ds41-1017-ivq
cd ds41-1017-ivq
sha256sum -c --ignore-missing SHA256SUMS

# 2. Assemble the base. This loop needs only one extra part of free disk.
B=DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
for i in $(seq -w 1 40); do cat "$B.part$i-of-40" >> "$B" && rm "$B.part$i-of-40"; done
sha256sum -c --ignore-missing SHA256SUMS      # the assembled .gguf must report OK

# 3. Build ds4-dfm-rs and serve
git clone https://github.com/Baekpica/ds4-dfm-rs.git && (cd ds4-dfm-rs && make cuda-spark)
./ds4-dfm-rs/ds4-server --cuda -m "$B" \
    --engram-dir deepseek-engram \
    --vision-gguf DeepSeek-V4.1-Flash.vision.gguf \
    --mem-budget-mb 110000 --host 0.0.0.0 --port 8000
```

If you already have the VQ release or the official 1017 checkpoint, add
`--exclude "deepseek-engram/*" --exclude "*.vision.gguf"` to the download. Then
point `--engram-dir` and `--vision-gguf` at the files you have. They are
byte-identical.

The server provides OpenAI-style `/v1/chat/completions`, `/v1/completions`
and `/v1/responses`, and Anthropic-style `/v1/messages`. The embedded drafter
is controlled with `--mtp-mode off|auto|on`. DeepSeek recommends
`temperature` 1.0 and `top_p` 1.0 for 1017. Send them explicitly in each
request, because sampling on this path follows the request. See the
[ds4-dfm-rs DeepSeek V4.1 guide](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/deepseek41-flash.md)
for the full flag set and serving limits. Standard llama.cpp, Ollama and
Transformers loaders do not read this format.

## Use the 1017 Engram, not the V4.1 one

The same caveat as the VQ release applies. 1017 changed the two Engram shards
without changing their names or sizes. An engine pointed at the older
`deepseek-ai/DeepSeek-V4.1-Flash` tables loads them without error and runs a
mismatched model. Use the copies in `deepseek-engram/` (SHA-256 `76a05106…` and
`9dc51433…`), or the same files from the 1017 checkpoint. Do not mount YoungAi's
V4.1 domain sidecars (zchain / GR-RB) either. They were solved against other
weights, and any GR sidecar fitted to the VQ release's indices does not fit
these.

## Speed

![DeepSeek V4.1 Flash VQ prefill and decode on one DGX Spark, 2K–64K](docs/img/v41-flash-vq-spark-2k-64k-2026-10-13.png)

*DeepSeek V4.1 Flash with the same `vq8sh14-q4k-mtpnative` layout and YoungAi's
code GR sidecar, not this artifact. iVQ reads the same bytes per token, so decode
speed should match. Drafter acceptance depends on the generated text and can
differ. Graph and record from
[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs/blob/a51ec2c0b4f8d55c5fee0e794977d95ea277481f/docs/benchmarks/ds41-spark-64k-2026-10-13/README.md).*

## Limits

- This is lossy at about 1.6 bits per routed-expert weight. The calibration-weighted
  fit lowers every layer's expert error, by 7% for a typical token, and end to end
  iVQ is only slightly closer to the original than VQ (KL −4.9%, next-token agreement
  unchanged); it is worse on Chinese and Japanese web text.
- The measurements are teacher-forced with a reference forward on A100s. Generation
  quality, serving memory and GB10 speed have not been measured for these files yet.
- The MTP draft towers are the VQ release's, unchanged. They were not refitted with
  calibration data.
- YoungAi reports that this weight recipe under-weights stop decisions (`</think>`,
  EOS), so long thinking segments can fail to close. Its server therefore defaults
  to non-thinking mode. Assume the same until it is measured here.
- The custom VQ format needs ds4-dfm-rs or YoungAi's engine.

## Credits and sources

- **[YoungAi](https://github.com/fodelf/YoungAi)** by Wenzhou Wu (吴文周,
  [fodelf](https://github.com/fodelf) /
  [wenzhouwu](https://huggingface.co/wenzhouwu)), MIT. YoungAi supplies the
  `vq8sh14-q4k-mtpnative` recipe and the DQVL v3 payload format, and its tools
  built the VQ base this release patches: the q4_K skeleton, the MTP towers and
  the GGUF layout. The weighted kernels keep YoungAi's assignment and FP64 Lloyd
  rules (`v41_vq.cu`). Its notice is kept in [`LICENSE`](LICENSE).
- **[llama.cpp / GGML](https://github.com/ggml-org/llama.cpp)**: the importance-matrix
  idea this release carries over to vector quantization, the GGUF container and the
  q4_K layout.
- **[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)**: the runtime and the
  calibration tools (`tools/vq_calib/`,
  [#108](https://github.com/Baekpica/ds4-dfm-rs/pull/108)), MIT
  ([`LICENSE-ds4-dfm-rs`](LICENSE-ds4-dfm-rs)). Its DeepSeek V4.1 support was
  ported by [JLFN](https://github.com/JLFN)
  ([#80](https://github.com/Baekpica/ds4-dfm-rs/pull/80), vision in
  [#83](https://github.com/Baekpica/ds4-dfm-rs/pull/83)).
- **[antirez/ds4](https://github.com/antirez/ds4)** (DwarfStar) by Salvatore
  Sanfilippo, the DeepSeek V4 engine YoungAi started from.
- **[DeepSeek-AI](https://huggingface.co/deepseek-ai)**: the model, its Engram
  tables and the reference inference code, MIT
  ([`LICENSE-DeepSeek`](LICENSE-DeepSeek)).
- **Calibration text**: NVIDIA Nemotron post-training datasets, FineWeb-Edu,
  FineWeb-2 and FineMath (Hugging Face), github-code-clean (codeparrot),
  Chinese-DeepSeek-R1-Distill-data-110k and COIG-CQIA. Only statistics derived
  from them are in the weights.

## License

MIT. The weights derive from DeepSeek-V4.1-Flash-1017 (MIT,
[`LICENSE-DeepSeek`](LICENSE-DeepSeek)). The VQ format, recipe and base tools are
YoungAi's (MIT, [`LICENSE`](LICENSE), which includes the GGML notice). The
calibration tools are ds4-dfm-rs's (MIT, [`LICENSE-ds4-dfm-rs`](LICENSE-ds4-dfm-rs)).
