---
base_model: google/gemma-4-26B-A4B-it
tags:
- awq
- quantized
- fastserve
---

# gemma-4-26B-A4B-it — AWQ 4-bit

Auto-quantized from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
by **[fastserve](https://github.com/jireh-father/fastserve)**'s self-quantization pipeline
(`publish/`) — so it can be trusted the way a random community AWQ requant
sometimes can't. Two community quants fastserve auto-detected during its own
benchmark run looped garbage tokens instead of answering; every checkpoint
published here passed an accuracy gate against its bf16 baseline first.

## Quantization

- Method: AWQ (W8A8), group size 128, via [llm-compressor](https://github.com/vllm-project/llm-compressor)
- Calibration: 256 samples from `wikitext`

## Validation (accuracy gate — this is why you can trust it)

| | bf16 baseline | this AWQ checkpoint |
|---|---|---|
| GSM8K accuracy (n=15) | 0.5333 | 0.6 |

Within 0.1 (absolute) of the bf16 baseline, <30% degenerate
(repeated-token loops). **A checkpoint that failed this gate would not have
been uploaded.**

## Serve it with fastserve

[fastserve](https://github.com/jireh-father/fastserve) auto-detects this checkpoint — point it at the
**original** model id and it finds this AWQ + wires up speculative decoding:

```bash
git clone https://github.com/jireh-father/fastserve && cd fastserve && ./install.sh

# serve an OpenAI-compatible API (auto-picks this AWQ checkpoint)
./fastserve serve google/gemma-4-26B-A4B-it

# or benchmark the speedup vs the naive baseline
./fastserve bench google/gemma-4-26B-A4B-it --compare-baseline
```

Then query it like any OpenAI endpoint:

```bash
curl localhost:8000/v1/completions \
  -d '{"model": "glenic/gemma-4-26B-A4B-it-W8A8-INT8", "prompt": "Q: What is 17*4?\nA:", "max_tokens": 64}'
```

## Or serve directly with vLLM

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model glenic/gemma-4-26B-A4B-it-W8A8-INT8
```

## License

Inherits the base model's license — see
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) for terms.
