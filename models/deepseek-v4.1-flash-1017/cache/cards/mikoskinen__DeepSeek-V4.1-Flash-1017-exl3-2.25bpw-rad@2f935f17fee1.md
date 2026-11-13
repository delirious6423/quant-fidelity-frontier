---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- radiance
- exl3
- rocm
- rdna4
- uncensored
---

# DeepSeek V4.1 Flash 1017, EXL3 2.25 bpw for Radiance

This is a Radiance container (`.rad`) of
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) for four AMD
Radeon AI PRO R9700 cards. It's the file behind the DeepSeek results in
[r9700-exact-serving](https://github.com/mikoskinen/r9700-exact-serving), and that repository's README
has the commands that start it with the same engine image and settings.

| | |
|---|---|
| Container | `deepseek-v4.1-flash-1017-exl3-2.25bpw.rad`, 272,254,373,888 bytes (253.6 GiB) |
| Files here | the container in six parts, `.part-00` to `.part-04` at 45 GiB each and `.part-05` at 28.6 GiB, because the Hub accepts files up to 50 GB |
| Routed experts | EXL3 from the `2.25bpw` branch of [turboderp/DeepSeek-V4.1-Flash-1017-exl3](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3) (commit 9183eaae), 2 bits in layers 6 to 34 and 2.5 bits in layers 0 to 5 and 35 to 39 |
| Every other weight | byte-copied from [bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad](https://huggingface.co/bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad): the attention and shared experts in q6_K, q5_K and q8_0, the Engram table in iq4_nl and the DSpark drafter. That container is an uncensored build, with the refusal direction removed from the attention output projections of layers 11 to 28. |
| `dspark-dlw-q8_0.rad` | a weight sidecar with 43 of the DSpark drafter's dense tensors in q8_0, made with `rad-sidecar` from the container |
| `dspark-conf.f32` | the release's `mtp.2.confidence_head.proj.weight` as raw float32, the drafter's confidence head |
| Engine | a Radiance fork with the radiance-exl3 plugin, as the image `ghcr.io/mikoskinen/r9700-exact-serving:deepseek-e231d919` |

## Download and join

The join needs about 320 GB of free disk:

```bash
hf download mikoskinen/DeepSeek-V4.1-Flash-1017-exl3-2.25bpw-rad --local-dir ~/models/ds41-exl3
cd ~/models/ds41-exl3
for p in deepseek-v4.1-flash-1017-exl3-2.25bpw.rad.part-0[0-5]; do
  cat "$p" >> deepseek-v4.1-flash-1017-exl3-2.25bpw.rad && rm "$p"
done
sha256sum -c SHA256SUMS
```

## License

MIT, the license of DeepSeek-V4.1-Flash-1017 and of both source repositories. The full text is in
`LICENSE`.
