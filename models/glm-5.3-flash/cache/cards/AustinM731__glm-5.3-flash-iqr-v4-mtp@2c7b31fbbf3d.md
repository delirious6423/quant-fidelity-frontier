---
license: mit
license_link: LICENSE
base_model:
- zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- radiance
- rocm
- amd
- rdna4
- moe
- iqr
- mtp
- 4-bit
- int8
---

# GLM-5.3-Flash, IQR-V4 + Native MTP: radiance container

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (rev `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`) converted into a single `.rad` container for the [radiance](https://codeberg.org/StillDeadcode/radiance) inference engine (AMD RDNA4 / ROCm), with **native serial MTP drafting (checkpoint layer 45)** included.

Layers 0–44 and the vision tower are preserved byte-for-byte (`rad-convert --reuse --in-place`) from [animusnull/glm-5.3-flash-iqr-v4](https://huggingface.co/animusnull/glm-5.3-flash-iqr-v4) (rev `a2d63477`), and checkpoint layer 45 (`599` `mtp.*` weights: 576 IQR-V4 routed experts, 6 IQ8R linears, 17 BF16/F32 tensors) was appended using the identical `glm53-iqr-v4.recipe`.

**Engine requirement.** Serving with `num-speculative-tokens > 0` requires [StillDeadcode/radiance#661](https://codeberg.org/StillDeadcode/radiance/pulls/661). Until that PR merges into an official release, use the drop-in overlay image `docker.io/aml731/radiance:2.3.0-glm53-mtp` (`sha256:e6fdee6ef424302c67ac8c5e72947e3e1013da3f33f486a87004de0b0a640571`), which replaces only `/opt/radiance/share/radiance/architectures/glm5next.so` on top of official `stilldeadcode/radiance:2.3.0`. Unmodified `stilldeadcode/radiance:2.3.0` can still serve this container with `num-speculative-tokens: 0`.

| | |
|---|---|
| File | `glm53-flash-iqr-v4-mtp.rad`, 162.60 GiB (`174,594,022,936` bytes) |
| SHA-256 | `e7dee9cd1a18808d281c0e2f9e3bd499a16e5ec50ad5b7c46dd8c83cf481153a` |
| Base container (layers 0–44) | `animusnull/glm-5.3-flash-iqr-v4` rev `a2d63477` (`1e20c4bc…2687`, 25,544 weights retained in place) |
| MTP layer (layer 45) | 599 `mtp.*` weights from `zai-org/GLM-5.3-Flash` rev `46a050be` (`rad-convert --max-spec 3 --kv-cache-dtype i8 --reuse ... --in-place`) |
| Routed experts (layers 3–45) | 4-bit computed-trellis codes (IQR-V4, `tcq=16`, `tcq_code=mad`, `fwht128`), one f32 scale a row |
| Trunk & MTP linears | Dense FFN, shared experts, and MLA `q_a`/`kv_a`/`q_b`/`output` as rotated int8 (IQ8R, `group=256`, `fwht128`) |
| As released (BF16/F32) | KDA projections & state, MLA `kv_b`, DSA indexers, routers, norms, mHC connections, MTP `eh_proj`/norms, embedding, `lm_head`, vision tower |
| Speculator | Native serial MTP (`num-speculative-tokens: 2` recommended on 4× R9700 `expert_tiered`; supports `1..3+`) |

## Throughput (4× Radeon AI PRO R9700, TP4 `expert_tiered`, `i8` latent KV)

Measured on official Radiance 2.3.0 (`max-model-len: 524288`, `host-pool-mib: 65536`, `prefix-cache-mib: 8192`) with 36-cell closed-loop workload benchmark (60 s/cell):

| Cell | `k=0` (baseline) | `MTP=2` | Gain |
|---|---:|---:|---:|
| `code c1 512tok` dec p50 | 69.4 tok/s | **106.5 tok/s** | +53.5% (2.26 tok/blk) |
| `code c1 d8K-session` dec p50 | 67.0 tok/s | **111.0 tok/s** | +65.7% (2.54 tok/blk) |
| `code c1 d32K-session` dec p50 | 65.9 tok/s | **103.7 tok/s** | +57.4% (2.35 tok/blk) |
| `prose c1 512tok` dec p50 | 70.2 tok/s | **97.2 tok/s** | +38.5% (2.01 tok/blk) |
| `json c1 512tok` dec p50 | 70.4 tok/s | **101.0 tok/s** | +43.5% (2.11 tok/blk) |
| `tool c1 512tok` dec p50 | 69.8 tok/s | **124.9 tok/s** | +78.9% (2.60 tok/blk) |
| `code c16 512tok` agg | 175.4 tok/s | **228.1 tok/s** | +30.0% |
| `prose c16 512tok` agg | 182.8 tok/s | **223.1 tok/s** | +22.0% |
| `json c8 512tok` agg | 186.2 tok/s | **240.7 tok/s** | +29.3% |
| `tool c8 512tok` agg | 159.1 tok/s | **237.1 tok/s** | +49.0% |

## Serve

```yaml
server:
  host: 0.0.0.0
  port: 7999
  kernels: glm,libr4d,libavx,libref
models:
  GLM-5.3-Flash:
    model: /models/glm53-flash-iqr-v4-mtp.rad
    tp: 4
    max-num-seqs: 4
    max-model-len: 524288
    max-num-batched-tokens: 2048
    prefill-interleave: 4
    kv-cache-dtype: i8
    placement: expert_tiered
    host-pool-mib: 65536
    prefix-cache-mib: 8192
    num-speculative-tokens: 2
    temp: 1.0
    top-p: 0.95
```
