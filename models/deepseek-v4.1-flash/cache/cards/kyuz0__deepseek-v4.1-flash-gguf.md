---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash
- deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- gguf
- deepseek
- metal
- rocm
---

# DeepSeek V4.1 Flash GGUF for DwarfStar

## DeepSeek V4.1 Flash 1017

The 1017 Q2 and matching BF16 vision encoder come from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170). They use the DwarfStar GGUF format. Pair the 1017 encoder with the 1017 language model; the original release remains available below.

| Artifact | File | Bytes |
| --- | --- | ---: |
| 1017 Q2, 1,046 tensors | `DeepSeek-V4.1-Flash-1017-Q2.gguf` | 365,713,686,528 |
| 1017 BF16 vision encoder, 266 tensors | `DeepSeek-V4.1-Flash-1017-vision-BF16.gguf` | 970,555,552 |

```sh
hf download kyuz0/deepseek-v4.1-flash-gguf DeepSeek-V4.1-Flash-1017-Q2.gguf --local-dir gguf
hf download kyuz0/deepseek-v4.1-flash-gguf DeepSeek-V4.1-Flash-1017-vision-BF16.gguf --local-dir gguf
```

### Tested ROCm runtime

The 1017 files were tested on AMD gfx1151 with ROCm 10.0, using [kyuz0/ds4](https://github.com/kyuz0/ds4/tree/f46631e483e19d13f23d3c4bced7ade0d086a3ad) with these source overlays in order:

1. [Bounded decoder replay](https://github.com/kyuz0/ds4/commit/38d9b735c96ca64ccee1a463d361416a782bd75b).
2. [Raw completion prompts](https://github.com/antirez/ds4/commit/bda86bcbca2409c0a768e5c5ff15a58037cca0f0).
3. [Loaded-model modalities](https://github.com/antirez/ds4/commit/67f2ca912c1d402747b58afbf090e6e2698f04a2).

The combined source tree is `57e71ca100ac7080716eb56e767263bca6983544`. For 1017 vision, apply the included [checkpoint compatibility patch](compatibility/vision-checkpoint-1017.patch) to that tree before building. It selects the matching encoder revision from the language checkpoint metadata. The image preprocessing and inference equations are unchanged.

A short text invocation with the tested options is:

```sh
./ds4 --rocm --quality \
  --model gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  --ctx 4096 --tokens 128 --threads 4 --nothink \
  --temp 1 --top-p 0.95 --min-p 0 --seed 1017 \
  --prompt 'Calculate 17 * 19. Give the result, then explain your calculation in two short sentences.'
```

Text generation and the public multimodal session API were exercised with these weights. The vision API comparison covers 32 cases and 40 image occurrences with the matching 1017 encoder. The reported distribution comparisons use the runtime's quality mode.

### 1017 quantization and calibration

Q2 follows the public antirez recipe: IQ2_XXS routed gate/up experts, Q2_K routed down experts, Q8_0 attention/shared/output weights, and F16/F32 tensors elsewhere. Native Engram data is retained losslessly and read from the GGUF on disk. The separate vision encoder retains the source BF16 tensors.

The tested resident runtime prepared 151.76 GiB of language-model tensor spans, before context and runtime buffers. The file also contains 188.83 GiB of Engram data. Keep it on a fast local SSD and allow additional memory for the selected context and vision encoder.

The importance matrix was collected from the native 1017 checkpoint using 49,152 text tokens and 6,246 vision-conditioned prefill tokens from 7 images. Of 15,360 routed experts, 15,268 were observed; 92 use the public converter fallback. This is a new 1017 calibration matrix. The Q2 was converted directly from the original native weights using this matrix.

### 1017 paired quality measurements

These comparisons teacher-force identical frozen token histories through the original native 1017 checkpoint and Q2. The original weights are mixed FP8/FP4. Forward KL uses full-vocabulary distributions at temperature 1. Results include quantization and runtime arithmetic; image results include the complete existing image pipeline.

NLL has the same definition as the original DwarfStar scorer’s `avg_nll`: average negative log probability of the reference continuation tokens. Lower is better on the same frozen targets; positive ΔNLL means Q2 assigns them less probability. The short panel reuses the V4.1 Flash September 19 continuations with official 1017 token IDs and newly collected 1017 native logits. Scores from different checkpoints, tokenizers or panels are not interchangeable. Top-token agreement is not answer accuracy.

| Compared panel | Cases | Scored positions | Native NLL | Q2 NLL | ΔNLL | Forward KL | Top-token agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short text | 100 | 6,159 | 0.363260 | 0.456892 | +0.093632 | 0.129874 | 90.03% |
| Context compatibility | 12 | 763 | 0.182847 | 0.211355 | +0.028508 | 0.048796 | 95.81% |
| Images, whole pipeline | 32 | 252 | 1.047979 | 0.818835 | -0.229144 | 0.006310 | 99.60% |
| Native generation pilot | 1 | 128 | 0.522755 | 0.576433 | +0.053678 | 0.072756 | 88.28% |
| Prefill control, same pilot history | 1 | 128 | 0.511912 | 0.576433 | +0.064521 | 0.072916 | 88.28% |
| Expanded text | 900 | 36,697 | 0.926836 | 0.915063 | -0.011773 | 0.090616 | 94.54% |
| Native generation histories, 4 × 4,096 tokens | 4 | 16,384 | 0.751280 | 0.807564 | +0.056284 | 0.059656 | 90.08% |
| Fixed long continuations | 8 | 49,152 | 0.130657 | 0.151656 | +0.020999 | 0.037186 | 98.04% |
| Deep lookup/multihop subset, 7 of 36 cases | 7 | 2,688 | 0.284009 | 0.280946 | -0.003063 | 0.047818 | 97.99% |

Deep coverage is partial: six lookup cases at 4K, 8K, 16K, 32K, 64K and 128K, plus one 4K multihop case. The remaining 29 cases are untested. Of the 2,688 scored positions, 127 are teacher-forced answer tokens and the rest are prompt probes. Answer-only forward KL is 0.024531, with matching top choices at all 127 positions. This does not establish independently generated answer accuracy or coverage of the other deep task types.

#### Expanded text groups

| Group | Cases | Scored positions | ΔNLL | Forward KL | Top-token agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| expanded-code | 200 | 9,348 | +0.012151 | 0.078385 | 95.79% |
| expanded-logic | 200 | 536 | -0.946078 | 0.100304 | 93.84% |
| expanded-math | 200 | 15,226 | -0.044691 | 0.057887 | 93.48% |
| expanded-multilingual | 100 | 113 | -1.856009 | 0.776334 | 61.06% |
| expanded-prose | 100 | 8,274 | +0.092138 | 0.186485 | 93.45% |
| expanded-structured | 100 | 3,200 | +0.027909 | 0.008354 | 100.00% |

Groups have different token denominators. The multilingual group contains short frontiers rather than full-answer accuracy tests. Negative ΔNLL on fixed targets does not establish better task performance.

Natural image-aligner outputs were compared before prompt-sentinel insertion: 32 cases, 40 image occurrences, mean row cosine 0.998737 and RMSE 0.001771 against native BF16 features.

The generation-history panels score saved native continuations, not independently generated Q2 task success. All four long native traces reached their 4,096-token caps. The pilot and prefill control share a history. There is no combined quality percentage, universal acceptance threshold or subtraction of native numerical controls as an error floor.

Source revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`. The included upstream MIT license applies.

## Original V4.1 Flash release

Calibrated weights for the Metal implementation in
[DwarfStar](https://github.com/antirez/ds4), developed on the `ds4.1flash` branch.
These are not interchangeable with DeepSeek V4 Flash weights or its DSpark drafter.

| Download target | File size | Main weights |
| --- | ---: | ---: |
| `ds41f-q2` | 340.60 GiB | 151.77 GiB |
| `ds41f-q4` | 482.98 GiB | 294.15 GiB |
| `ds41f-vision` | 0.90 GiB | Separate vision encoder |

The language GGUFs each include 188.83 GiB of native FP8 Engram tables.
DwarfStar reads the needed rows directly from disk; it never makes the whole
table resident. Keep the GGUF on a fast local SSD. The main-weight sizes above
do not include context or runtime buffers.

## Download and Run

From a DwarfStar checkout with V4.1 support:

```sh
./download_model.sh ds41f-q2
./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming --ctx 32768
```

Use SSD streaming on a single 128 GB Mac. Q2 also fits resident tensor
parallelism across two 128 GB Macs with RDMA, or full residency on a larger
Mac. Q4 needs SSD streaming on smaller Macs; the resident target is a 512 GB Mac.
It does not fit resident TP across two 128 GB Macs.

```sh
./download_model.sh ds41f-q4
./download_model.sh ds41f-vision
```

Q4 exceeds Hugging Face's single-file limit, so it is stored in two binary
parts. The downloader joins them into `DeepSeek-V4.1-Flash-Q4.gguf`, verifies
the complete file and removes the temporary parts. Allow another 37 GiB of
free disk space while joining. Rerun the command after an interruption.
The individual parts are not runnable GGUFs.

For images, add `--vision gguf/DeepSeek-V4.1-Flash-Vision.gguf` to the language
model command. The same model and memory options work with `ds4-agent` and
`ds4-server`.

## Quantization

Q2 uses IQ2_XXS routed gate/up experts and Q2_K down experts. Q4 uses Q4_K for
all routed experts. Both retain Q8 attention projections, shared experts and
output, with F16/F32 tensors elsewhere and unchanged native Engram data.
They use the same 8,192-token activation imatrix, not a requantization of one
GGUF into the other.

Source: `deepseek-ai/DeepSeek-V4.1-Flash` at
`df42c109f1defefcbfcedbe7d905718a12266e40`. The separate encoder comes from the
same checkpoint. See the included upstream MIT license.
