---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- exl3
- glm
- mirror
---

# GLM-5.3-Flash EXL3 4.05 bpw (turboderp), mirror

Mirror of [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)@[`2a30229e67012798ba9f0cd832bb78abf4c363d5`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2a30229e67012798ba9f0cd832bb78abf4c363d5) for reproducibility of [github.com/hyukudan/strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab).

The files are exact copies of that upstream revision: same paths, same bytes. They were not re-quantized or modified. All credit belongs to the original authors below; please use and cite the upstream repositories. This mirror only pins the revision our engines were built and measured against.

## Original authors

- [turboderp](https://huggingface.co/turboderp): EXL3 4.05 bpw quantization (exllamav3) and the exllamav3 project
- [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash): original model GLM-5.3-Flash (MIT)
- Base model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)@`46a050bec13002ad29ee795c16e61cf4fa3c4a9c`

## License

MIT. The upstream `LICENSE` file is included verbatim.

## How our engines use it

- The engine runs these weights on two Strix Halo nodes with tensor parallelism over USB4 (TP2), default R1 decode identity (exact), plus `mtp.safetensors` for multi-token prediction experiments.
- See the [engine README](https://github.com/hyukudan/strix-halo-engine-lab#glm-53-flash).

## Files

30 files, 165,151,541,665 bytes (165.15 GB). `README.md` is this card; the upstream model card is kept verbatim as `README.upstream.md`. Large-file sha256 values equal the upstream LFS hashes and were recomputed from our local copies before upload (any mismatch would have stopped the upload). `SHA256SUMS` lists every file.

| File | Bytes | sha256 |
|---|---:|---|
| `LICENSE` | 1,070 | `30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1` |
| `README.upstream.md` (upstream `README.md`) | 7,238 | `4f46db37e6a7cbe3a494d6d7f616a96f4fffe6d1de68be361199a715c8d928fc` |
| `chat_template.jinja` | 8,617 | `41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3` |
| `config.json` | 86,509 | `df80c17a68120aeae4c1eca8a9aa67866603484f0487cd5458c08dfd3c45156d` |
| `generation_config.json` | 194 | `230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f` |
| `model-00001-of-00019.safetensors` | 8,587,818,716 | `1c7a87b0fb108521d0cf3e495ca63c37732a06744131e8933ca12d2df0d8806e` |
| `model-00002-of-00019.safetensors` | 8,587,936,824 | `a67cd8e559ac013fc0d1dc58b66bd105e54bee0a31147b295089b2537646d4dc` |
| `model-00003-of-00019.safetensors` | 8,590,375,904 | `d069cea25c6082a70b2dbc601725687f8d164a5dd047894acb58476b4175c1af` |
| `model-00004-of-00019.safetensors` | 8,590,340,084 | `d222d3137bec069a165219542d49d02defef2df69763464e93463a176e2ba935` |
| `model-00005-of-00019.safetensors` | 8,590,383,216 | `6fdd4557f29e1b77248b6bbd6b35c691ef4b1dc63b7305c9f5cb0080e1c82836` |
| `model-00006-of-00019.safetensors` | 8,587,986,708 | `3e9c1f39a631dedd5cce7614947d1f3eab71c93dbb4c79bd2eec0ab413236055` |
| `model-00007-of-00019.safetensors` | 8,590,342,372 | `62eca4decce10159107948a320083d8e9dcf60f25e058877a1662b80f1412220` |
| `model-00008-of-00019.safetensors` | 8,590,383,800 | `e8f8762fa1c6a698fc74d50760b82ec72d6043fb9ef6e05dac4032999f57d7c5` |
| `model-00009-of-00019.safetensors` | 8,587,944,528 | `f9d2e2df7fc7a65c33514a584e5a228d5eb4bec72fc516d9c649c94b06e3d00e` |
| `model-00010-of-00019.safetensors` | 8,590,383,896 | `5eb19bda8461e44323898172492f9f5c948f5c8ad00816b0e004feaf8b11ccfa` |
| `model-00011-of-00019.safetensors` | 8,590,342,356 | `f2e687a3d410af6bd48f90d5770fac1054ad5c337d3cfa2bd56a5600387ba36a` |
| `model-00012-of-00019.safetensors` | 8,590,383,392 | `caee2cdd11e781195a3bbb70b53da0c08c4add3485e200a4a28119d238201523` |
| `model-00013-of-00019.safetensors` | 8,587,986,540 | `094d5b9341505d25852993e0eef6d137ba75c4179bec2204bd067af264da5dd1` |
| `model-00014-of-00019.safetensors` | 8,590,342,372 | `765f21794314da81bccdf6a1c9b3fff3f86711d34d83f7551667455a9775dc1a` |
| `model-00015-of-00019.safetensors` | 8,590,383,888 | `f08d208c8ad27b3bb9c75c42c90e33a3e8acd722ccf5f4f74df1638084d46067` |
| `model-00016-of-00019.safetensors` | 8,590,341,500 | `e4c0bbf93639fd96eddb9a38505c4de99d650f82d376c0e303a7c3fd38544c3d` |
| `model-00017-of-00019.safetensors` | 8,587,986,956 | `8e954f3e1660889a50974f46e66ad75a4e3bcadb0398a91646d58a09d9c2f54c` |
| `model-00018-of-00019.safetensors` | 8,590,342,372 | `bad6f1e51e7bd25a4da4fe55efe4f91512bf4e82085db2baa5cce1adea027c58` |
| `model-00019-of-00019.safetensors` | 6,661,824,340 | `de2031315b49f89e3d091e92bead8b239ea46e304236f34b5734a68b0b07d547` |
| `model.safetensors.index.json` | 15,963,650 | `ee1f2dbea800dea0b4225c38193f7ef41180ed42dd51f80a0cee58daf44cf606` |
| `mtp.safetensors` | 3,793,519,833 | `1bd09bd4eff4adc568e4ac3e92efb436c2d3aa5f0dceab4c72005d796066822d` |
| `processor_config.json` | 909 | `aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed` |
| `quantization_config.json` | 47,905,678 | `d18c35bf6184a67a3b612b15c216bdfdf9e81505c0e02b646152925b13fafef3` |
| `tokenizer.json` | 20,217,442 | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| `tokenizer_config.json` | 761 | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |

All files of the upstream revision are mirrored.
