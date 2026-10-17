---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
tags:
  - mlx
  - quantized
  - qwen4-exp
---

# Qwen3.8-Flash-Next MLX Activation-Weighted 3-bit — Native PLE Pack

This is a reproducible MLX conversion of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at immutable
revision `f5d08274bafd880402bd16f5e3e6c514136ec06c`.

This complete mere.run pack preserves every source checkpoint file and tensor
byte-for-byte, then adds `MERERUN_PLE_STORE.json`. A compatible mere.run build
uses that manifest to mirror only the 33 PLE-bearing safetensors files to its
internal application cache when the installed model lives on an external
volume. The one-time copy is size- and SHA-256-verified; subsequent launches
reuse it. No local conversion or `mere.run model optimize` step is required.

The 48 base routed-expert banks use fresh MLX affine Q3/group-64 codes generated directly from the original BF16 checkpoint. Their scales and biases are refit against frozen image-and-text expert-input second moments. Remaining eligible core, MTP, and vision matrices stay Q4; the 160-wide n-gram table stays Q4/group-32.

- Artifact payload: 83.49 GiB
- Quantized Q2 modules: 0
- Quantized Q3/group-64 modules: 144
- Quantized Q4/group-32 modules: 128
- Quantized Q4/group-64 modules: 783
- Source: 180B parameters including 125B main, 51B n-gram embedding, and 4B MTP
- Native context: 262,144 tokens

## Run locally with mere.run

This profile requires the managed model update after mere.run v0.45.0. With a
build that contains that update, review the license and pull the checkpoint:

```bash
mere.run model pull vision-chat-q38-flash-next-3bit-native-ple \
    --accept-license-terms
```

The repository is public and ungated. The acceptance option records your
acknowledgment of the bundled Qwen Community License 1.0.

To generate text with the bundled multi-token prediction (MTP) head, run:

```bash
mere.run text chat \
    --model vision-chat-q38-flash-next-3bit-native-ple \
    --context-size 32768 \
    --max-tokens 256 \
    --temperature 0 \
    --no-thinking \
    --stream \
    --stats \
    --prompt "Explain sparse attention in three short sentences."
```

For an external SSD, add `--cache-dir /Volumes/Models/huggingface-cache` to
the pull command and replace the sample path with your mounted volume. Keep the
volume mounted when you use the model. mere.run automatically keeps the
32,431,337,095-byte PLE-bearing subset in its internal cache when capacity
allows; if the internal reserve check fails, inference continues directly from
the installed model without changing its weights.

## Native qualification

The fresh Q3 checkpoint passed a no-regression comparison against the pinned
published Q4 checkpoint in mere.run on a 128 GiB Apple Silicon Mac. MTP was
disabled during checkpoint selection.

- The candidate passed 58 of 61 cases by exact expected output.
- The three absolute misses matched the published Q4 output exactly.
- The sealed holdout produced identical candidate and Q4 outputs in all 16
  cases, including exact output on all eight image and OCR cases.
- The holdout peak memory footprint was 62,065,713,472 bytes for this profile
  and 77,180,432,496 bytes for Q4, a 19.58% reduction. Neither run increased
  swap usage.

The bounded suites don't establish general model quality or BF16 parity. For
the test identities, hashes, memory measurements, and limits, see
`MERERUN_QUALIFICATION.json`.

## Native PLE placement qualification

The packaged artifact was checked against source revision
`c699bd611366cbc441377275bd1b7a6d2e18e1df`: all 147 source files are present,
the index still maps all 3,817 tensors, and all 33 PLE-bearing files pass their
published SHA-256 digests. The runtime installed-table check reported the full
32,000,153,600 logical PLE bytes with only 15,360 bytes of additional active
MLX memory.

On a 128 GiB Apple Silicon Mac, an adjacent seven-request, 128-token series
under active Spotlight/UI contention measured 7.52 aggregate decode tok/s from
the packaged internal placement versus 6.22 tok/s from the external source
(+20.8%), with one versus three pathological tail runs. Absolute healthy-run
speed drifted during indexing, so this is evidence for the placement policy,
not a universal throughput guarantee. The PLE tensor values and model outputs
are unchanged by construction.

## Runtime status

The tensor inventory, source hashes, MLX packing, fused-expert split, convolution
layout, and zero-centered RMSNorm conversion are validated by the bundled
`MERERUN_CONVERSION.json`. Use a Qwen4Exp-aware runtime such as mere.run or
mlx-vlm.

## License

This redistribution retains the upstream **Qwen Community License 1.0** in
`LICENSE`. Review it before use. In particular, it contains attribution/display
requirements for very large commercial products and separate-license conditions
for certain commercial Model-as-a-Service and AI Work Assistant uses. The model
is not gated; downloading or using it does not remove those terms.

The upstream model card is preserved as `README.upstream.md`.
