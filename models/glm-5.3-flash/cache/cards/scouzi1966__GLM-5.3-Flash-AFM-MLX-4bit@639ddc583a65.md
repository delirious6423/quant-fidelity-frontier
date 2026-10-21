---
language:
- en
- zh
license: mit
library_name: mlx
pipeline_tag: text-generation
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- afm
- afmkit
- mlx
- 4-bit
- mtp
- apple-silicon
---

> [!WARNING]
> **Experimental AFM artifact.** This checkpoint targets AFM and
> [AFMKit](https://github.com/scouzi1966/AFMKit) `v0.1.18-rc.3` or newer. It
> may not work with other MLX runtimes or older AFM releases. Behavior, quality,
> performance, memory use, compatibility, and file layout may change.

# GLM-5.3-Flash AFM MLX 4-bit MTP

This is an AFM format-v3 conversion of the official
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/04c4e9e95c5da8862dced7e5056455116f83a7e0)
checkpoint at revision `04c4e9e95c5da8862dced7e5056455116f83a7e0`. It is
converted directly from the official FP8 SafeTensors distribution; it is not a
re-quantization of a third-party 4-bit checkpoint.

The conversion contains 201 weight files totaling 181,694,111,440 bytes
(approximately 169.2 GiB), excluding runtime memory and KV cache. Loading the
checkpoint measured approximately 169.4 GB wired memory on a 512 GB Apple
Silicon development machine. A smaller minimum memory configuration has not
been established.

## Conversion

- AFM conversion format: `3`
- Profile: `mlx-affine-4`
- Quantization: affine 4-bit, group size 64
- Source MTP layer: preserved
- `num_nextn_predict_layers`: `1`
- Conversion units completed: `201/201`
- Mapped tensors: `3,059`
- Source tensors omitted from the NextN layer: `0`
- Routed experts are reconstructed in numerical expert order.
- Router weights remain FP32.
- Attention projection layout transformations are applied for the AFM GLM runtime.
- Vision weights, tokenizer, processor assets, and the upstream chat template
  are preserved.
- Detailed source, shard, size, and SHA-256 provenance is recorded in
  `.afm-mlx-conversion.json`.

End-to-end vision quality is not qualified. Weight preservation alone is not an
image-quality parity claim.

## MTP status

The structural layer 45 NextN tensors are present and converted. This is
different from a general quality or speedup claim.

Qualification with AFMKit `v0.1.18-rc.3` succeeded:

- The server loaded `GLM embedded NextN layer` at depth 1.
- An explicit greedy streaming request returned exactly `PASS` with
  `finish_reason=stop`.
- The diagnostic profile recorded `draftedTokens=4`, `acceptedTokens=4`,
  `headForwards=9`, `targetForwards=5`, and zero rejection replays.
- All 201 tensor files matched their recorded sizes and SHA-256 digests before
  upload.
- Hugging Face verified all 212 published files after upload.

This is runtime and integrity qualification, not a standard benchmark result.
Throughput and correctness under other prompts, batch sizes, cache states, tool
calls, and vision inputs remain unmeasured.

## Run with AFM

Use AFM built against AFMKit `v0.1.18-rc.3` or newer:

```bash
afm mlx -m scouzi1966/GLM-5.3-Flash-AFM-MLX-4bit \
  --mtp --reasoning-effort low --port 9999
```

The GLM-5.3 template does not expose a direct thinking off switch. Low reasoning
effort reduces but does not eliminate reasoning output.

```bash
curl http://127.0.0.1:9999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "scouzi1966/GLM-5.3-Flash-AFM-MLX-4bit",
    "messages": [{"role": "user", "content": "Return exactly PASS and nothing else."}],
    "temperature": 0,
    "max_tokens": 128,
    "stream": true
  }'
```

## Attribution and license

Original model by Z.AI. The original MIT license is included in `LICENSE`. The
upstream model card is preserved in `UPSTREAM_README.md`; upstream model claims
are not measurements of this conversion. AFM conversion and packaging are
provided by the AFM / AFMKit project.
