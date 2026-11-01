---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - dspark
  - speculative-decoding
  - draft-model
  - deepseek
  - fle
---

# DeepSeek-V4.1-Flash-1017 DSpark drafter (GGUF, for FLE)

The DSpark speculative-decoding drafter that DeepSeek ships inside
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(the `mtp.0..2` tensors: three stages, 128 routed experts top-3, block of five tokens, Markov and confidence head),
converted to GGUF. It is not a standalone model: it drafts tokens that the target verifies, and every verified token
is identical to plain decoding.

| file | size | sha256 |
|---|---:|---|
| `FLE-DSpark-V4.1-1017-Q2K-Q8.gguf` | 5.2 GB | `c223adafb9aa0e125557907165f54c555f3fa576918f9561979136bca8822482` |

Routed experts Q2_K, dense projections Q8_0, hyper-connection weights F32, Markov head F16, confidence head F32.
Quantizing the hyper-connection (`hc_*`) or Markov tensors to K-quants breaks the draft (illegal memory access / zero
acceptance), so they are kept in float.

## How it was made

1. Download only shards 44–46 of the source checkpoint (they hold all 2,401 `mtp.*` tensors, 7.9 GB).
2. `convert_hf_to_gguf.py --dspark --outtype q8_0` (DSpark converter of the FLE engine / llama.cpp).
3. `llama-quantize --allow-requantize --tensor-type "ffn_(gate|up|down)_exps=q2_K" --tensor-type "hc_.*=f32"
   --tensor-type "markov_w1=f16" --tensor-type "markov_w2=f16" --tensor-type "conf_proj=f32" … Q8_0`

## Run (FLE engine)

Needs an engine that loads the Markov head as `markov_w1` / `markov_w2`
([FastLocalExperts](https://github.com/sirfyyn/FastLocalExperts) main from c80bda923 on).

```bash
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf \
  -md FLE-DSpark-V4.1-1017-Q2K-Q8.gguf -ngld 99 --spec-type draft-dspark --spec-draft-p-min 0.5 \
  -ngl 99 -cmoe -fit on
```

`--spec-draft-p-min` uses the confidence head: positions below the threshold are not sent for verification.

## Measured

Target: ISTA-DASLab GSQ-RCO IQ2_XXS (232 GiB with Engram), one RTX PRO 4500 (32 GB), 120 GB RAM, NVMe; experts on the CPU
with a GPU expert cache. FLE engine main, two runs each, 1,000-token warm-up then 1,100 measured tokens:

| setup | decode t/s | warm-up (counting) t/s | draft acceptance |
|---|---:|---:|---:|
| no drafter, expert cache 18 GB | 23.2 | 38 | – |
| this drafter, `p_min 0.5`, expert cache 15 GB | **25.0** | **55** | 74–83 % |

Without `p_min` the acceptance on free prose drops to 30–48 % and the drafter costs throughput on this
memory-bound setup.

## Credits

Weights: DeepSeek (MIT). Converter and engine: llama.cpp contributors and the FLE engine.
