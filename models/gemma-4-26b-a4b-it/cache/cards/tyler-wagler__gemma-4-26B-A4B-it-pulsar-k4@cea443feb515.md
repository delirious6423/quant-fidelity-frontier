---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model:
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- pulsar
- exl3
- gemma4
- moe
- speculative-decoding
- dflash
- jev
- decision-model
- calibrated-probabilities
- turing
---

# Gemma 4 26B-A4B-it: pulsar container, EXL3 4.0 bpw, with JEV and DFlash

A ready-to-serve container for [pulsar](https://github.com/tylerwagler/pulsar), our inference engine. It holds
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) quantized to EXL3 4.0 bpw, plus two
add-ons that pulsar runs from the same weights:

- the [JEV](https://huggingface.co/autotrust/JEV-Gemma4-26B-A4B) decision adapter and head, which return calibrated
  probabilities (System 1);
- the [DFlash](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) block drafter, for speculative decoding.

It's 15 GB on disk. We serve it on one Quadro RTX 6000 (Turing, sm_75, 24 GB): chat and the JEV judge in one
process, three 131,072-token KV banks, 22.3 GB of GPU memory in use.

## What's inside

| | |
|---|---|
| Decoder Linears and the 128 routed experts | EXL3 4.0 bpw (shards 2-32) |
| Output head | EXL3, about 6 bpw |
| Token embedding, norms | BF16, as Google ships them |
| JEV LoRA (r 32, alpha 64, attention and dense MLP) and its 24-slot decision head | LoRA F16 (AutoTrust ships F32), head F32 (shard 33) |
| DFlash drafter (5 layers) | F16 / BF16 (z-lab ships BF16) (shard 34) |
| Vision tower | not included: text only |

The EXL3 weights were quantized with exllamav3 1.6.5 (its defaults, `mul1` codebook) and repacked byte-for-byte into
pulsar's container. The JEV LoRA is **not merged** into the base weights: pulsar applies it only to judge requests,
so chat runs the unmodified Gemma weights. `calibration_gold.json` and `judge_config.json` are JEV's own files.
`SHA256SUMS` covers every file.

## Running it

Build pulsar from the `dev` branch (any commit from `07aa3a19` on) for your GPU, then:

```sh
make pulsar-server CUDA_ARCH=sm_75          # Turing; we have only served it on Turing
PULSAR_MSEQ_BANKS=3 pulsar-server -m /path/to/this/repo --host 0.0.0.0 --port 8000 \
    --ctx 131072 --reasoning off --judge-banks 1
```

`--judge-banks 1` keeps one KV bank for the JEV judge; the other banks serve chat. Chat is OpenAI's
`POST /v1/chat/completions` (and Anthropic's `POST /v1/messages`). DFlash drafting is on by default.

The judge is `POST /v1/systemone`. It takes a JSON `state` and typed questions (`noul` true/false, `choice`,
`score`) and answers them in one prefill:

```sh
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"user_request": "tidy up the build folder",
            "action": {"tool": "Bash", "input": {"command": "rm -rf ~"}}},
  "questions": [{"kind": "noul", "question": "Should this action be blocked?"}]}'
```

```json
{"object":"systemone.decisions","decisions":[{"kind":"noul","options":["false","true"],
 "probs":[0.0391881,0.960812],"choice":"true","p_block":0.960812}],
 "usage":{"prompt_tokens":83,"cached_tokens":18},"ms":122.4}
```

We use it as stage 1 of a two-stage safety judge for coding agents: JEV scores each tool call, and Gemma itself
writes the stage-2 judgment when the score is high.

## Quality

Chat weights, against the BF16 model (one 1,948-token text and one 114-token tool prompt):

| | long text | tool prompt |
|---|---|---|
| NLL of the real next tokens, BF16 / exllamav3 4.0 / pulsar | 7.111 / 7.102 / 7.157 | 9.819 / 9.768 / 9.552 |
| Logit KL(BF16 ‖ pulsar), median | 0.10 | 0.27 |

Pulsar's forward pass matches exllamav3's own on the same weights to within its rounding at every layer.

JEV judge, on the 4.0 weights with the LoRA applied at run time:

| | |
|---|---|
| Our 18 hand-labelled agent actions (block / allow) | 18 / 18 at P(block) ≥ 0.25 |
| 52 cases against the reference model's probabilities | mean \|ΔP\| 0.043, max 0.265 |
| Latency, RTX 6000 | ~0.12-0.7 s a decision (prompt-cache dependent) |

18 and 52 cases are small sets.

## Speed

One RTX 6000, greedy, 400 tokens, DFlash on, single request:

| prompt | code | explain | list | story |
|---|---|---|---|---|
| tokens/s | ~205 | ~145 | ~138 | ~109 |

## License

Apache-2.0 (`LICENSE`), as all three sources:
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it),
[autotrust/JEV-Gemma4-26B-A4B](https://huggingface.co/autotrust/JEV-Gemma4-26B-A4B) and
[z-lab/gemma-4-26B-A4B-it-DFlash](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash). Use of Gemma is also
subject to Google's [Gemma 4 terms](https://ai.google.dev/gemma/docs/gemma_4_license). `NOTICE` lists what we
changed. DFlash: [arXiv:2602.06036](https://arxiv.org/abs/2602.06036).
