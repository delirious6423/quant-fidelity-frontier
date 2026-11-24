---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - radiance
  - rocm
  - amd
  - rdna4
  - int8
---

# Qwen3.8-27B, KL-IQR V2 containers

The V2 Nano, Tiny and Small entries for the [radiance](https://codeberg.org/StillDeadcode/radiance) quant challenge,
with the engine image they were scored on. The entry is pull request
[#364](https://codeberg.org/StillDeadcode/radiance/pulls/364); its `docs/QC-QWEN38-KL.md` says what is in the
containers.

| Class | Label | File | Bits a weight | Bytes | sha256 |
|---|---|---|---|---|---|
| Nano | KL-IQR-V2-3.00 | `qwen3.8-27b-kl-v2-nano.rad` | 2.9989 | 11014828032 | `b9cf85db21346e4927c6ae01ddd43bf84b9dbe8b45edb2113e615e4f3a31fd0a` |
| Tiny | KL-IQR-V2-4.00 | `qwen3.8-27b-kl-v2-tiny.rad` | 3.9991 | 14377656320 | `cc0d548d33acd4bdc1319c116402b57627733dafeb021d8e82a893540f51256c` |
| Small | KL-IQR-V2-5.50 | `qwen3.8-27b-kl-v2-small.rad` | 5.4988 | 19420073984 | `0bd5a06adf125fb41a8af24971aa33b7000fe99fac4d93c060fc901ab3a9fcc0` |
| Engine image | | `radiance-1.15.1-q38-v2entry.tar.zst` | | 248863919 | `637b60342617c11b8515d53838c32790e50c00ccab1d554c5604574d1df4309a` |

All three use int8 activations in every GEMM and read their weights in the stored form.

The Nano and Tiny containers hold row scales and norm weights trained against the bf16 model on the challenge's
calibration split, with every code frozen: about 4.6 million numbers at Nano and 1.2 million at Tiny. Small holds no
trained value. The pull request's doc lists exactly what was trained.

The V1 entries stay in their own repositories:
[nano](https://huggingface.co/Ashlyn4202025/qwen3.8-27b-kl-iqr-nano),
[tiny](https://huggingface.co/Ashlyn4202025/qwen3.8-27b-kl-iqr-tiny),
[small](https://huggingface.co/Ashlyn4202025/qwen3.8-27b-kl-iqr-small).

## Running them

They need the pull request's branch, or the image in this repository, which was built from that branch's engine
source. Stock radiance releases do not read them. AMD RDNA4 (gfx1201), two cards.

    zstd -dc radiance-1.15.1-q38-v2entry.tar.zst | docker load      # perire2019/radiance:1.15.1-q38-v2entry

    docker run --rm --network host --init --device /dev/kfd --device /dev/dri \
      --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
      -v "$PWD:/m:ro" --entrypoint radiance perire2019/radiance:1.15.1-q38-v2entry \
      --model /m/qwen3.8-27b-kl-v2-nano.rad --tp 2 --kv-cache-dtype bf16 --tp-wire exact \
      --max-model-len 262144 --max-num-seqs 16

For a KL run add `--kld-ref DIR --kld-out report.json`, as the challenge does.
