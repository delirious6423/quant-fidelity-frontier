---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- oqe
- quantized
- deepseek
- deepseek-v4.1
- mixture-of-experts
- mtp
- apple-silicon
---

# DeepSeek-V4.1-Flash-1017-oQ4e-mtp

An MLX checkpoint of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`), exported with [oQ](https://github.com/jundot/omlx) at level **oQ4e** for Apple Silicon. It contains the full model: language backbone, Engram tables, vision encoder and multimodal aligner, and the DSpark MTP weights used for speculative decoding (Lightning MTP in oMLX).

DeepSeek-V4.1-Flash-1017 is DeepSeek's October 17, 2026 update to [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), with improvements across reasoning, coding, agentic and visual benchmarks and shorter outputs. See the base model card for benchmarks.

| | |
|---|---|
| Total tensor payload | 402.4 GiB (432.0 GB) in 82 shards, 1838 tensors |
| Average bits per weight | 4.529 (complete checkpoint), 4.365 excluding Engram |
| Resident weights with Engram on SSD | about 288 GiB |
| Recommended hardware | Apple Silicon Mac with 512 GB unified memory (tested on an M3 Ultra) |
| Prompt format | DeepSeek's reference encoder in [`encoding/`](encoding/README.md) (no Jinja chat template) |

## What oQ4e does for this model

For DeepSeek-V4.1 the oQ4 levels keep the release's own precision for every linear layer: routed experts stay MXFP4 and the dense attention and shared-expert projections stay MXFP8, exactly as DeepSeek shipped them. Those weights already fit the oQ4 budget at their original precision (4.365 bits per weight against a 4.6 target), so oQ does not requantize any of them. The quantization report records `allocation: original_precision_fits_independent_remaining_budget` and `calibration: not_needed_no_linear_requantization`, and the importance matrix of the *e* variant was not applied to any module because no linear layer is rounded.

The only lossy step is the **Engram tables** (the n-gram hash embeddings, about a quarter of all parameters), stored as 4-bit affine with group size 32 under their own budget, instead of the source's BF16. The original BF16 and FP32 tensors (norms, routers, hyper-connection parameters, attention sinks, `wo_a`, the vision encoder) are kept as they are.

### Bit distribution

Logical parameter counts exclude quantization scales and biases; storage and bits per weight include them.

| Storage format | Logical weights | Share | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| MXFP4, group 32 | 557.171B | 73.004% | 275.669 GiB | 4.25 |
| Engram affine 4-bit, group 32 | 196.614B | 25.762% | 114.444 GiB | 5.00 |
| BF16 | 3.419B | 0.448% | 6.369 GiB | 16.00 |
| MXFP8, group 32 | 5.959B | 0.781% | 5.723 GiB | 8.25 |
| FP32 | 0.042B | 0.006% | 0.158 GiB | 32.00 |
| **Total** | **763.205B** | **100%** | **402.363 GiB** | **4.529** |

### Storage by component

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone (excluding Engram) | 551.881B | 300.132 GB / 279.520 GiB |
| Engram tables | 196.614B | 122.884 GB / 114.444 GiB |
| DSpark MTP | 14.225B | 8.047 GB / 7.495 GiB |
| Vision encoder, aligner, image tokens | 0.485B | 0.971 GB / 0.904 GiB |

The Engram tables are listed in the same `model.safetensors.index.json` as everything else; no separate directory is needed.

## Memory

oMLX can keep the Engram tables on SSD (memory-mapped, reading only the rows each request needs), so they do not count against unified memory. Everything else stays resident and packed (MXFP4 and MXFP8 are not expanded to BF16).

Measured on an M3 Ultra (512 GB) with Engram SSD offload, one stream, 4,096-token prompt:

| Build | Prefill (4,096 tokens) | Decode (4k context) | MLX active after load | MLX peak |
|---|---|---|---|---|
| oMLX upstream `main` (what you get today) | 180 tok/s | 19.9 tok/s | 287.9 GiB | 295.4 GiB |
| oMLX development build with the 1017 numerics and fused decode kernels (not yet released) | 368 tok/s | 31.3 tok/s | 287.9 GiB | 296.7 GiB |

Decode is greedy and single-stream without MTP. Prefill runs in 2,048-token chunks. Both builds read the same checkpoint; the development build only changes how oMLX computes, and is listed to show where the model is headed on this hardware.

## Usage

Run it with [oMLX](https://github.com/jundot/omlx) (a DeepSeek-V4.1-capable version). Download it into your oMLX model directory, for example

```bash
hf download bearzi/DeepSeek-V4.1-Flash-1017-oQ4e-mtp --local-dir ~/models/DeepSeek-V4.1-Flash-1017-oQ4e-mtp
```

and load it like any other model. oMLX handles the prompt format, thinking mode, tool calls, images and Engram offload. The DSpark MTP weights are included, so Lightning MTP can be turned on in the model settings.

The model uses DeepSeek's own prompt format. For your own tooling, use the reference encoder in [`encoding/`](encoding/README.md) (copied unchanged from the base repository) or DeepSeek's [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe).

### Note on 1017 numerics

DeepSeek's 1017 reference adds one step to attention: the attention output is quantized to FP8 and dequantized (`act_quant(o)`) before the grouped `wo_a` projection. oMLX versions that predate 1017 (including upstream `main` at the time of this upload) skip that step. The model still runs and answers correctly with them (see the stock-oMLX results below), but its outputs then differ slightly from DeepSeek's reference.

## Verification

Before upload, the checkpoint was loaded and compared with the original DeepSeek-V4.1-Flash-1017 checkpoint (MXFP4 / FP8 weights with BF16 Engram tables), both run by oMLX with the 1017 reference numerics on the same M3 Ultra:

- **Teacher-forced agreement** over 1,024 tokens of mixed text: top-1 prediction identical at 95.90% of positions, top-5 overlap 88.48%, mean absolute change of the source's top-1 log-probability 0.0536 nats. Engram is the only quantized component, so this is the effect of the 4-bit Engram tables. Greedy generations from the two checkpoints are both correct but drift apart as soon as one near-tied token differs.
- **Greedy generations** with DeepSeek's prompt encoder (chat mode, non-thinking), using the build with the 1017 reference numerics:

**Prompt:** In exactly three sentences, explain why the sky is blue and why sunsets are red.

````text
Rayleigh scattering causes air molecules to scatter short blue wavelengths of sunlight much more strongly than longer red wavelengths, so the sky appears blue when you look away from the sun. At sunset, sunlight travels through a much longer path of atmosphere, scattering away most of the blue light and leaving the longer red and orange wavelengths to reach your eyes. This is why the sky shifts from blue overhead to red near the horizon as the sun sets.
````
_87 tokens, 32.0 tok/s; greedy output differs from the original checkpoint's from the first token_

**Prompt:** Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals, with a short docstring and two doctest examples.

````text
```python
def merge_intervals(intervals):
    """
    Merge overlapping closed intervals.

    Given a list of [start, end] intervals, return a new list of
    non-overlapping intervals sorted by start, where any intervals
    that overlap or touch are merged.

    >>> merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])
    [[1, 6], [8, 10], [15, 18]]
    >>> merge_intervals([[1, 4], [4, 5]])
    [[1, 5]]
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] =
````
_200 tokens (cut at 200), 31.9 tok/s; greedy output identical to the original checkpoint's for the first 57 tokens_

**Prompt:** 用两句话解释什么是光合作用。

````text
光合作用是绿色植物利用阳光将二氧化碳和水转化为有机物（如葡萄糖）并释放氧气的过程。这一过程不仅为植物自身提供养分，也为地球上几乎所有生命提供了能量来源和呼吸所需的氧气。
````
_42 tokens, 31.8 tok/s; greedy output identical to the original checkpoint's for the first 14 tokens_

**Prompt:** A train leaves at 9:40 and arrives at 13:05 after one 15-minute stop. How long was it actually moving? Answer briefly and show the arithmetic.

````text
From 9:40 to 13:05 is 3 hours 25 minutes = 205 minutes.  
Subtract the 15-minute stop: 205 − 15 = 190 minutes.  

**Answer: 190 minutes (3 hours 10 minutes).**
````
_57 tokens, 32.5 tok/s; greedy output identical to the original checkpoint's for the first 57 tokens_


- **Stock oMLX** (upstream `main`, without the 1017 `act_quant(o)` step) loads and runs the same checkpoint: teacher-forced top-1 agreement with the original checkpoint under the 1017 numerics 94.53% (against 95.90% with them), and its greedy answers:

**Prompt:** In exactly three sentences, explain why the sky is blue and why sunsets are red.

````text
The sky is blue because air molecules scatter short-wavelength blue light much more strongly than long-wavelength red light, a process called Rayleigh scattering. At sunset, sunlight travels through a much longer path of atmosphere, so most of the blue light is scattered away before it reaches your eyes, leaving the transmitted light dominated by reds and oranges. This same scattering explains both effects: blue dominates the scattered light you see overhead, while red dominates the light that survives the long journey to the horizon.
````
_99 tokens, 20.4 tok/s; greedy output identical to the original checkpoint's for the first 13 tokens_

**Prompt:** Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals, with a short docstring and two doctest examples.

````text
```python
def merge_intervals(intervals):
    """
    Merge overlapping closed intervals.

    Given a list of [start, end] intervals, return a new list of
    non-overlapping intervals sorted by start, where any intervals
    that overlap or touch are merged into one.

    >>> merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])
    [[1, 6], [8, 10], [15, 18]]
    >>> merge_intervals([[1, 4], [4, 5]])
    [[1, 5]]
    """
    if not intervals:
        return []

    # Sort by start, then by end for deterministic behavior
    sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(sorted_intervals[0])]

    for start, end in sorted_intervals[
````
_200 tokens (cut at 200), 20.1 tok/s; greedy output identical to the original checkpoint's for the first 147 tokens_

**Prompt:** 用两句话解释什么是光合作用。

````text
光合作用是绿色植物利用阳光将二氧化碳和水转化为有机物（如葡萄糖），并释放氧气的过程。这一过程不仅为植物自身提供养分，也是地球上绝大多数生命能量和氧气的根本来源。
````
_41 tokens, 20.2 tok/s; greedy output identical to the original checkpoint's for the first 41 tokens_

**Prompt:** A train leaves at 9:40 and arrives at 13:05 after one 15-minute stop. How long was it actually moving? Answer briefly and show the arithmetic.

````text
From 9:40 to 13:05 is 3 hours 25 minutes = 205 minutes.  
Subtract the 15-minute stop: 205 − 15 = 190 minutes.  

**Answer: 190 minutes (3 hours 10 minutes).**
````
_57 tokens, 20.8 tok/s; greedy output identical to the original checkpoint's for the first 57 tokens_


## License

The model weights are released by DeepSeek under the [MIT License](LICENSE); this quantized checkpoint keeps that license. The `LICENSE` file and the `encoding/` reference encoder are copied unchanged from the base repository.

## Citation

```bibtex
@misc{deepseekai2026deepseekv41flash,
      title={DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression},
      author={DeepSeek-AI},
      year={2026},
}
```

## Acknowledgements

Model by [DeepSeek-AI](https://huggingface.co/deepseek-ai). Quantized with [oQ](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md) from oMLX.
