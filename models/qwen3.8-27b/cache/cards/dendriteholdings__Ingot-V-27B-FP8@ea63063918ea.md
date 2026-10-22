---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- fp8
- qwen3.8
- vllm
- ingot
- bittensor
---

# Ingot-V-27B-FP8

**An FP8 build of Qwen3.8-27B that stays 35.6% closer to the full-precision model than Qwen's own FP8.**

Same format, same 30.9 GB and same serving commands as [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8). To switch, change the model name.

Ingot-V is the final champion of the first [Ingot](https://theingot.ai) round. Over six days, from 16 to 22 October 2026, 64 FP8 quantizations of Qwen3.8-27B from 30 miners were scored against the original BF16 model. Five were crowned in turn, and this is the last of them.

## At a glance

| | |
| --- | --- |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), BF16, 55.6 GB |
| This model | FP8 E4M3 weights in 128×128 blocks with dynamic activation scales, 30.9 GB |
| Layout | Identical to Qwen's official FP8: the same tensors, and the same 882 modules left in higher precision |
| Context | 262,144 tokens |
| Input | Text, images and video |
| Speculative decoding | Ships Qwen's multi-token-prediction head, unchanged |
| License | Apache 2.0, inherited from Qwen |

## How close it is to the original

Quantizing a model shrinks its weights and nudges its predictions. Ingot measures that nudge directly.

Both models read the BF16 model's own answers to 1,080 prompts drawn at random from the [Ingot corpus](https://huggingface.co/datasets/dendriteholdings/ingot_corpus). At every answer token, Ingot compares their next-token probabilities, using KL divergence. It also compares their internal activations at every 8th layer, using hidden-state distance. Lower is closer for both.

| Compared with Qwen's official FP8 | KL divergence from BF16 | Hidden-state distance from BF16 |
| --- | --- | --- |
| Ingot-V-27B-FP8 | 35.6% lower | 23.4% lower |

Each duel draws its own prompts. To get these figures, each king's improvement over the king it dethroned, measured on the same prompts, is multiplied along the line from Qwen's FP8 to Ingot-V.

## Benchmarks

| Benchmark | Tasks | Qwen3.8-27B BF16 | Qwen FP8 | Ingot-V FP8 |
| --- | --- | --- | --- | --- |
| SWE-bench Multilingual | 300 | 73.7 | 73.0 | 69.0 |
| Terminal-Bench 2.1 | 89 | 57.3 | 52.8 | 60.7 |
| AutomationBench | 600 | 25.0 | 25.3 | 25.3 |
| GPQA Diamond | 198 | 81.3 | 79.8 | 80.8 |
| MATH-500 | 500 | 97.4 | 97.2 | 96.0 |
| τ³-bench Banking | 97 | 46.4 | 45.4 | 46.4 |
| MMMU | 300 | 82.0 | 78.0 | 80.3 |

Ingot-V matches or beats Qwen's FP8 on five of the seven suites. On six it is within two points of BF16 or above it. The exception is SWE-bench Multilingual, 4.7 points under BF16.

All three models ran under the same protocol: temperature 0.6, top-p 0.95, up to 16,384 output tokens per turn and up to 150 agent steps. They were served with vLLM 0.31.0 at the full 262k context, with multi-token-prediction speculative decoding. Each suite ran once, and a single run can move by a few points between repeats, so treat small gaps as ties.

## Quickstart

This model runs anywhere Qwen3.8-27B-FP8 runs. Its 31 GB of weights fit on a single 80 GB GPU.

The command our benchmarks ran with, on vLLM 0.31.0 or newer:

```bash
vllm serve dendriteholdings/Ingot-V-27B-FP8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 2}'
```

For SGLang or TokenSpeed, use Qwen's [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) or [TokenSpeed recipe](https://lightseek.org/tokenspeed/recipes/models#qwen3-8) with this model's name.

Then call it through any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
reply = client.chat.completions.create(
    model="dendriteholdings/Ingot-V-27B-FP8",
    messages=[{"role": "user", "content": "Write a Python function to merge two sorted linked lists."}],
    temperature=1.0,
    top_p=0.95,
    extra_body={"top_k": 20},
)
print(reply.choices[0].message.content)
```

Qwen's recommended sampling settings apply unchanged:

- **Thinking mode, the default:** temperature 1.0, top-p 0.95, top-k 20.
- **Non-thinking mode:** temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5.

Qwen's [model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) covers thinking control, image and video input, and long contexts beyond 262k.

## How it was made

[Ingot](https://theingot.ai) is a competition on the Bittensor network, subnet 97, where miners quantize one fixed model. The validator scores every upload against the BF16 original.

A challenger takes the crown only by beating the reigning king on the same prompts, by a set margin on both measures. Each new king is published here unchanged.

| King | Crowned, UTC | Closer than Qwen's FP8, by KL | Miner uid |
| --- | --- | --- | --- |
| [Ingot-I-27B-FP8](https://huggingface.co/dendriteholdings/Ingot-I-27B-FP8) | 16 Oct, 18:21 | 13.6% | 195 |
| [Ingot-II-27B-FP8](https://huggingface.co/dendriteholdings/Ingot-II-27B-FP8) | 16 Oct, 19:37 | 22.2% | 228 |
| [Ingot-III-27B-FP8](https://huggingface.co/dendriteholdings/Ingot-III-27B-FP8) | 17 Oct, 00:34 | 29.0% | 224 |
| [Ingot-IV-27B-FP8](https://huggingface.co/dendriteholdings/Ingot-IV-27B-FP8) | 17 Oct, 19:57 | 34.3% | 180 |
| **Ingot-V-27B-FP8** | 19 Oct, 22:40 | **35.6%** | 15 |

Ingot-V held the crown from 19 October until the round closed on 22 October.

The weights are the winning miner's upload, byte for byte, and the credit is theirs. The miner is hotkey `5HMZMcoQ7HW2RmxQhVE4dcWHQiFYp2j51zAob5CHGW1J7rZC`, uid 15, with submission `039bfd9f-da1f-4888-b1b8-5454c658dca4`.

## Limitations

- **The closeness score covers text only.** It measures the language model. Image understanding is covered by the MMMU result above.
- **Everything else is Qwen's.** Capabilities, behaviour and safety are those of Qwen3.8-27B. Read Qwen's card before deploying.

## Next

Ingot's second round compresses the same model to 2 bits. Follow it at [theingot.ai](https://theingot.ai).

## License and citation

Apache 2.0, inherited from Qwen3.8-27B. If you use this model, please cite Qwen:

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```
