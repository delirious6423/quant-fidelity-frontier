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

> **Status, 2026-10-22 (KST): card first.** The calibration statistics have been
> collected (3.65M tokens through the original FP4/FP8 model on 8× A100), and the
> 40 routed-expert blobs are being converted. The GGUF parts, `SHA256SUMS` and the
> measurements below will be added to this repository when they are done. Until
> then, use the [VQ release](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF).

**The [VQ release](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF) of DeepSeek's October 17 V4.1 Flash, refitted with an importance matrix: the same layout, size and runtime, fitted to stay closer to the original model.**

The VQ release, like YoungAi's recipe it follows, trains its codebooks on the
weights alone. Before fitting, iVQ runs the original model over 3.65M tokens of
calibration text. It records how strongly real inputs drive each input channel of
each of the 384 experts in all 40 layers, then spends the shared codebooks'
precision where those activations are. This is the VQ counterpart of llama.cpp's
importance matrix (imatrix), hence the "i".

Everything else is held fixed, so the comparison with VQ measures the
calibration-weighted fit. That covers the bit allocation (13-bit on layers 0–13,
12-bit on 14–39), the file layout, and every byte outside the 40 routed-expert
blobs. [RECIPE.md](RECIPE.md) lists exactly what differs in the fit. The files
drop into a setup that already serves the VQ release.

**How it will be measured.** Both VQ and iVQ are scored teacher-forced against the
original FP4/FP8 model on 26 held-out cases (58,803 tokens, never used for
calibration or for any choice). The scores are KL divergence, next-token top-1
agreement and NLL, overall and per domain. The results go here when the files are
uploaded.

## At a glance

| | |
|---|---|
| Source | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) at `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Starting point | [Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF) (YoungAi `vq8sh14-q4k-mtpnative`) |
| What changes | The 40 routed-expert VQ blobs: codebooks, indices and row gains, fitted with calibration statistics |
| What does not change | Bit allocation (layers 0–13 13-bit K=8192, layers 14–39 12-bit K=4096), one E4M3 codebook per layer shared by all 384 experts and gate/up/down, F16 row gains, file layout and size. The q4_K skeleton, the MTP draft towers, the vision sidecar and the Engram tables are byte-identical to the VQ release |
| Calibration | 3,095 sequences, 3,653,893 tokens: English 51%, Korean 14%, Chinese 13%, code 13%, Spanish and Japanese about 2% each, DeepSeek agent prompts 5%. Raw text 41%, thinking chats 31%, non-thinking chats 28%. Up to 31,721 tokens per sequence |
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

In A.X-K2 and Beam, the plain weight error went up a little, by design, while the
error weighted by the measured activation energy went down. That is the error
that matters for the experts' outputs. The same weight-space comparison against
the VQ release will be added here for every layer.

## Measurements

All numbers come from the reference forward in `tools/vq_calib/ds41_ref.py`. It
follows DeepSeek's `inference/model.py`, emulates its FP8/FP4 kernels, and runs on
Runpod 8× A100 80GB. The teacher is the original FP4/FP8 checkpoint. Each student
replaces only the routed experts with the dequantized VQ or iVQ blobs, so the
q4_K skeleton, identical in both, is not part of these numbers. KL is
KL(teacher ‖ student) in nats per token over the full 129,280-token vocabulary;
top-1 is how often the student's next token matches the teacher's.

*Pending: overall and per-domain KL / top-1 / NLL for VQ and iVQ, and the
routed-expert overlap with the original per layer.*

Different GPUs, CUDA builds and kernels move these numbers slightly. Compare them
statistically, not digit by digit. These are not serving measurements: no DGX
Spark load, speed or generation test has been run on these files yet. Speed should
match the VQ release, because decoding reads exactly the same bytes per token.

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

The GGUF parts and `SHA256SUMS` are uploaded once the conversion is finished and
verified.

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

- This is lossy at about 1.6 bits per routed-expert weight. The importance matrix
  is meant to narrow the gap to the original, not to close it.
- The planned measurements are teacher-forced, with a reference forward on A100s.
  Generation quality, serving memory and GB10 speed will not be measured on
  Runpod.
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
