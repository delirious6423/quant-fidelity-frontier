---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: mlx
tags:
- mlx
- speculative-decoding
- mtp
---

# MTP drafter for Qwen/Qwen3.8-27B

The native multi-token-prediction head of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), extracted as a
standalone MLX drafter and quantized to 4-bit affine (group 64).
It carries no embedding or LM head of its own — it binds to the target model's at load,
so it pairs with any quantization of the same base model.

Speculative decoding verifies every proposed token against the target model, so this
drafter changes throughput only, never output: greedy decoding is byte-identical with
and without it.

## Reproducing

Built by [`scripts/build-mtp-drafter.mjs`](https://github.com/bendyline/gezel/blob/main/scripts/build-mtp-drafter.mjs)
from [Gezel](https://github.com/bendyline/gezel) — a local-first desktop app for assembling a team of AI agents that
run on your own machine:

```bash
git clone https://github.com/bendyline/gezel.git
node scripts/build-mtp-drafter.mjs --source Qwen/Qwen3.8-27B --out <dir>
```

The script fetches only the checkpoint shard(s) carrying the `mtp.*` tensors (1 of 18 for this
model), splits the head into a standalone drafter, and quantizes it to 4-bit.

## License

Apache-2.0, inherited from Qwen/Qwen3.8-27B. Weights are a derivative of that checkpoint.
