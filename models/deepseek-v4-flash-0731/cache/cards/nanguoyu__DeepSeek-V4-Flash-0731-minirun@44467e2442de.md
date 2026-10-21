---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
  - deepseek
  - minirun
  - mlx
  - apple-silicon
  - macos
  - ios
---

# DeepSeek-V4-Flash-0731-minirun

The weights of [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) at
revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, repacked byte for byte into the container format
[Minirun](https://minirun.dev) reads. Minirun streams this model from an external SSD on
Mac and iPhone through a memory budget you set, rather than loading it into
memory, so the model does not have to fit in the machine. The reader is open
source: [`Sources/StorageCore/Container/`](https://github.com/nanguoyu/minirun-app/tree/main/Sources/StorageCore/Container) in
[`nanguoyu/minirun-app`](https://github.com/nanguoyu/minirun-app).

## What this is

A **byte-preserving repack**. No requantization, no retraining, no numerical
change of any kind:

- the routed expert weights are the upstream FP4 packed values and their
  `ue8m0` scales, copied verbatim and only reordered;
- the remaining quantized matrices are the upstream FP8 `e4m3` values and their
  `ue8m0` block scales, copied verbatim;
- all other tensors are the upstream BF16/F32/I64 bytes, copied verbatim.

Every byte of weight data in this repository is a byte of the source
checkpoint at the pinned revision above, in a different order. The only bytes
that are not are container headers, and zero padding where a container's
scale region is shorter than its alignment unit.

The model configuration and tokenizer are verbatim files from the same pinned
source revision. Their byte counts and SHA-256 identities are recorded in
`index.json`.

Contents: 47 directories, 577 data files,
166.9 GB total, plus a per-directory manifest giving
each file's shape, element width and offsets.

## Run it with Minirun

**Get the app.** On a Mac, download [`Minirun.dmg`](https://downloads.minirun.dev/Minirun.dmg). On an iPhone,
install [TestFlight](https://apps.apple.com/app/testflight/id899247664) and open
[testflight.apple.com/join/QuYTybjV](https://testflight.apple.com/join/QuYTybjV)
on the phone; the source is at [`nanguoyu/minirun-app`](https://github.com/nanguoyu/minirun-app) if you
would rather build it.

**Point it at this repository.** In **Settings → Storage**, use *Add a folder…*
to register a folder on an external NVMe drive. In **Settings → Models**, open
*Find Models*, select this repository and press *Download* -- or point Minirun
at a copy you already have. Run *Verify all files*, then start a chat.

**What to expect.**

| | |
| --- | --- |
| On disk | 167 GB |
| MacBook Pro (M1 Pro, 32 GB, USB4 enclosure) | ≈ 5 s / token at a 10.6 GB budget, and it still runs at 2 GB |
| iPhone 16 Pro (USB 3 port, powered dock) | ≈ 17 s / token under 2 GB of memory |

Speeds change with the drive, the cable and the budget; the enclosure and cable
matter more than the SSD — USB4 against a 10 Gb/s hub is about 3× on the same
bytes — and the iPhone needs a powered dock.

**Requirements.** An Apple-silicon Mac on macOS 15 or later; an iPhone 15 Pro or
later on iOS 18 or later; an external NVMe drive with room for the model.

[minirun.dev](https://minirun.dev) · [Docs](https://minirun.dev/docs) · [This model](https://minirun.dev/models/deepseek-v4-flash-0731) ·
[GitHub](https://github.com/nanguoyu/minirun-app)

## Provenance

| | |
| --- | --- |
| Source model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Source revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Relationship | byte-preserving repack (no requantization) |

## License

This repository redistributes model weights owned by DeepSeek under the **MIT
License**, reproduced verbatim in [`LICENSE`](LICENSE) and copied unmodified
from the source repository at the pinned revision above.

Copyright (c) 2023 DeepSeek.

The MIT License permits use, copying, modification and redistribution,
including commercially, provided the copyright notice and the permission
notice are included in all copies or substantial portions of the Software.
The weights are provided "as is", without warranty of any kind. Refer to
[`LICENSE`](LICENSE) for the governing text; the summary above is not a
substitute for it.
