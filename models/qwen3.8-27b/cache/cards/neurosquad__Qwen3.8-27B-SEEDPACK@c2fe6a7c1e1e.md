---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- en
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
tags:
- seedpack
- gguf
- lossless-compression
- qwen3.8
- qwen
- unsloth
---

<p align="center">
  <a href="https://github.com/glmn-ai/seedpack"><img src="https://huggingface.co/neurosquad/Qwen3.8-27B-SEEDPACK/resolve/main/assets/seedpack-banner.png" alt="SeedPack" width="860"></a>
</p>
<p align="center">
  <img src="https://huggingface.co/neurosquad/Qwen3.8-27B-SEEDPACK/resolve/main/assets/seedpack-stats.png" alt="Qwen3.8-27B: 33 files, 472.07 GB, in one 45.41 GB archive, 10.4× smaller" width="860">
</p>

# Qwen3.8-27B-SEEDPACK

**The complete [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) repository in one 45.41 GB archive instead of 472.07 GB.** That covers the BF16 model in its 2 shards, all 24 quants from UD-IQ1_S to UD-Q8_K_XL, the multi-token-prediction layer (`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`), the vision projector in BF16 and F16, the imatrix and the README. The archive is 9.62% of the original size, smaller than the BF16 model alone (54.66 GB).

Every file is restored **byte for byte**. The SHA-256 of each file matches unsloth/Qwen3.8-27B-GGUF; the full list is [below](#contents-of-the-archive).

## What is SeedPack?

[SeedPack](https://github.com/glmn-ai/seedpack) is a lossless GPU archiver for GGUF model repositories, made by [NeuroSquad](https://huggingface.co/neurosquad).

A GGUF repository ships the same weights 20–30 times: the BF16 model plus every quantization of it. SeedPack uses the fact that every quant is a deterministic function of the same weights:

- **The seed is stored once.** The BF16 model is coded with an entropy coder built for BF16 weights: 54.66 GB → 35.78 GB (65.5%). Its shards stay separate files, each restored as it is, but the quants are grown from them as from one model.
- **Quants are grown, not stored.** Each quant is computed again from the seed on the GPU with a bit-exact CUDA port of llama.cpp's quantizers (`ggml-quants.c`) and the repository's importance matrix.
- **Only the difference is kept.** Where a re-derived quant differs from the published file, SeedPack stores a residual coded with the quantizer's own probabilities:
  - Q8_0, the MTP layer and the F16 vision projector come out exactly: **965.3 KB** together.
  - Q4_0 and Q4_1 were made with an earlier importance matrix (45 chunks, as their metadata records) than the one in the repository (1251 chunks). SeedPack fits that matrix back from the quants themselves, by least squares on their block scales, and keeps it in the archive (6.3 MB): Q4_0 (54.6 MB of 16.06 GB) and Q4_1 (43.5 MB of 17.54 GB).
  - The 21 Unsloth Dynamic (UD) quants come from Unsloth's own pipeline, whose codes drift from plain rounding along each row, so they cannot be re-derived exactly from BF16 and the imatrix. Their residuals cost **8.91 GB**, 2.5% of their 351.53 GB: the coder knows where in the row each weight is, and a tensor that several quants share (the same Q5_K `ffn_up` in UD-Q5_K_M and UD-Q5_K_XL, say) is stored once, the later files keeping a reference to it (4837 of them here).
- **Big models are streamed.** The BF16 model is larger than a 12 GB GPU and than the test machine's 32 GB of RAM, so packing and unpacking go through it in windows of about 1 GB of tensors. It is never held whole, in RAM or on disk, unless you extract it.
- **Everything is verified.** Unpacking checks the BLAKE3 hash of every file and the CRC32 of every chunk.

## How to use

### 1. Install SeedPack

```bash
pip install "seedpack>=0.3.0"
```

Version 0.3.0 or newer is needed: it is the first to handle a model split into shards, to code the IQ grid types and the drift of the Dynamic quants, and to fit an importance matrix back. Requirements:
- An NVIDIA GPU, Turing (RTX 20xx) or newer, with a CUDA 13 driver (R580 or newer). The model does not have to fit in GPU memory.
- Python 3.11+.
- Windows 10/11 or Linux. Tested on an RTX 5070 (12 GB) under Windows 10. No CUDA Toolkit is needed.
- Disk space for the archive (45.41 GB) plus the files you extract.

The first run compiles the GPU quantizers for your card (about 15 s) and caches them.

### 2. Extract quants or the whole repository

SeedPack reads the archive straight from this repository. `list` fetches only the index; `x` downloads the archive once into the Hugging Face cache.

```bash
seedpack list hf://neurosquad/Qwen3.8-27B-SEEDPACK                                           # what is inside
seedpack x hf://neurosquad/Qwen3.8-27B-SEEDPACK Qwen3.8-27B-UD-Q4_K_XL.gguf -o models               # one quant
seedpack x hf://neurosquad/Qwen3.8-27B-SEEDPACK "*UD-Q4_K_XL*" mmproj-F16.gguf -o models      # several, by name or mask
seedpack x hf://neurosquad/Qwen3.8-27B-SEEDPACK -o models                                    # the whole repository
seedpack test hf://neurosquad/Qwen3.8-27B-SEEDPACK                                           # decode and verify everything, write nothing
```

A downloaded copy works the same way:

```bash
hf download neurosquad/Qwen3.8-27B-SEEDPACK Qwen3.8-27B.seedpack --local-dir .
seedpack x Qwen3.8-27B.seedpack Qwen3.8-27B-UD-Q4_K_XL.gguf -o models
```

### 3. Run the model

The extracted files are the original Unsloth GGUFs, so any GGUF runtime works:

```bash
llama-cli -m models/Qwen3.8-27B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf --jinja
```

Ollama, LM Studio, Jan and KoboldCpp open the extracted `.gguf` files directly. Recommended settings are in the original card below.

## Speed

Measured on an NVIDIA RTX 5070 (12 GB), an Intel Core i7-8700K (6 cores / 12 threads), 32 GB RAM and Windows 10, with the source files on two SATA SSDs in a stripe (about 1 GB/s) and the archive written to another SATA SSD. Times include process start and file writes.

| Operation | Time | Speed |
|---|---:|---:|
| Pack the whole repository (472.07 GB) | 1373 s | 344 MB/s |

Peak GPU memory is 8.8 GB when packing. Extracting one quant reads the whole coded BF16 model (35.78 GB), so it takes at least as long as the disk needs for that; IQ quants take longer, their quantizers are slow even on the GPU.

The archive was made with SeedPack 0.3.0.

## Contents of the archive

<details>
<summary>33 files, 472.07 GB. Click to expand.</summary>

| File | Size | In the archive | SHA-256 |
|---|---:|---:|---|
| `BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf` | 49.99 GB | 32.73 GB | `b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f` |
| `BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf` | 4.67 GB | 3.06 GB | `92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02` |
| `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | 1.37 GB | 43.2 KB | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |
| `Qwen3.8-27B-Q4_0.gguf` | 16.06 GB | 54.6 MB | `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d` |
| `Qwen3.8-27B-Q4_1.gguf` | 17.54 GB | 43.5 MB | `3e020514545c310dfc511dc8d3ddc23482b645189cb9287816d84bed6eddd4ac` |
| `Qwen3.8-27B-Q8_0.gguf` | 29.05 GB | 893.6 KB | `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348` |
| `Qwen3.8-27B-UD-IQ1_M.gguf` | 6.73 GB | 1.09 GB | `1b5165a7149ea51e683c8eaf23372188ad9fc9d1a795386f7a1b558acf847dc6` |
| `Qwen3.8-27B-UD-IQ1_S.gguf` | 6.19 GB | 454.1 MB | `3895b6eaa91e705c06ad1938d16c22e86f073c6a67df86260a1da79be3d1f887` |
| `Qwen3.8-27B-UD-IQ2_S.gguf` | 8.37 GB | 967.3 MB | `7897d2c5a5cee46aef50895141b2c8a0803c1185f3d03c4fda4cd137a7ad77fe` |
| `Qwen3.8-27B-UD-IQ2_XXS.gguf` | 7.27 GB | 313.9 MB | `e792d8fb3142fe6d9171876d6da0f71f05a71028718debc72dbec93ff645e67d` |
| `Qwen3.8-27B-UD-IQ3_S.gguf` | 12.04 GB | 1.08 GB | `d847e2c1e4aa276e4b7b8e9ad7628050e61e165d49ab995407bc36677a6f3864` |
| `Qwen3.8-27B-UD-IQ3_XXS.gguf` | 10.93 GB | 485.3 MB | `c0b7c3038681ed2e3040456c1dd45f9858b6c2290bed172c70388a94874f3eee` |
| `Qwen3.8-27B-UD-IQ4_XS.gguf` | 14.25 GB | 754.3 MB | `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199` |
| `Qwen3.8-27B-UD-Q2_K_XL.gguf` | 9.83 GB | 444.6 MB | `fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0` |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | 13.15 GB | 123.8 MB | `8c2a45ff85e7674ca185ec8eb6cdeab0e617ed9d8018caed0b64380eb2a67a5e` |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | 16.46 GB | 660.0 MB | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| `Qwen3.8-27B-UD-Q4_K_S.gguf` | 15.36 GB | 98.6 MB | `75bc9c8adba2842e72f0ab5201aaa07133c5010b566305c09187fcbdcd364017` |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17.56 GB | 337.8 MB | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |
| `Qwen3.8-27B-UD-Q5_K_M.gguf` | 19.77 GB | 582.1 MB | `2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd` |
| `Qwen3.8-27B-UD-Q5_K_S.gguf` | 18.67 GB | 51.0 MB | `d8d62ffcf84d42658dd6ccf9782b4d0404700af78b26d750507510c7597b5bfe` |
| `Qwen3.8-27B-UD-Q5_K_XL.gguf` | 20.88 GB | 130.5 MB | `8601193d3d5760c37fb8ce1b43afebc69df5fb24e1fbc5a547c32e2200305276` |
| `Qwen3.8-27B-UD-Q6_K.gguf` | 21.98 GB | 285.4 MB | `c9c206812fbe4ac7b76a729e25928b63f2ae89d37f69da7a71c20aec763cd436` |
| `Qwen3.8-27B-UD-Q6_K_L.gguf` | 24.19 GB | 435.8 MB | `121355b4c7422771da25adc74090e3c90138f77ce5c92d348687d47824ec80f4` |
| `Qwen3.8-27B-UD-Q6_K_M.gguf` | 23.09 GB | 13.4 MB | `493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb` |
| `Qwen3.8-27B-UD-Q6_K_XL.gguf` | 25.30 GB | 189.2 MB | `701d8fa9ed214ab21bfc130cd2a7df19ca89bbef7713e2dfb19f3c63696aa917` |
| `Qwen3.8-27B-UD-Q8_K_L.gguf` | 28.05 GB | 409.5 MB | `2a13bba36d2efa213f9275abc430c40e4d914b145bfde976f30f1bb5b7e23ac2` |
| `Qwen3.8-27B-UD-Q8_K_XL.gguf` | 31.46 GB | 967.2 KB | `af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377` |
| `imatrix_unsloth.gguf` | 13.6 MB | (bundle) | `0ee5b10bd0c2fa2127c6f4b43dbfe1efd71e383b63217af9dade1de36599f1c1` |
| `mmproj-BF16.gguf` | 931.1 MB | 604.9 MB | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` |
| `mmproj-F16.gguf` | 927.6 MB | 28.5 KB | `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e` |
| `.gitattributes` | 4.2 KB | (bundle) | `git blob ba7d80f9eb55035cd728cd81a851f32b44147bfa` |
| `README.md` | 7.5 KB | (bundle) | `git blob 3fddda1850c5fd0c93e9ac3f7e20d8691b2163de` |
| `config.json` | 3.8 KB | (bundle) | `git blob c2bb5cf2a82d965d5b11aa078ba9571fe949a4d5` |

`README.md`, `config.json` and `.gitattributes` are in the small-files bundle (7.2 MB with the imatrix).

</details>

## Original model card

> The original card of [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) follows, unchanged. The model is by the [Qwen team](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0), and the GGUF quantizations are by [Unsloth](https://unsloth.ai).

---


# Read our How to [Run Qwen3.8-27B Guide!](https://unsloth.ai/docs/models/qwen3.8)
<div>
  <p style="margin: 0 0 0px 0; margin-top: 0px;">
    <em><a href="https://unsloth.ai/docs/basics/dynamic-3.0-ggufs">Unsloth Dynamic 3.0</a> achieves superior accuracy & outperforms other leading quants.</em>
  </p>
  <div style="display: flex; gap: 5px; align-items: center; margin-bottom: 0px;">
    <a href="https://github.com/unslothai/unsloth/">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="133">
    </a>
    <a href="https://discord.gg/unsloth">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/Discord%20button.png" width="173">
    </a>
    <a href="https://unsloth.ai/docs/models/qwen3.8">
      <img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/documentation%20green%20button.png" width="143">
    </a>
  </div>
    <ul style="margin: 0;">
    <li>Introducing <a href="https://unsloth.ai/docs/basics/dynamic-3.0-ggufs">Dynamic V3.0</a> GGUFs for SOTA accuracy and quantization performance</li>
    <li>Run and fine-tune Qwen3.8 in <a href="https://unsloth.ai/docs/new/desktop">Unsloth Desktop</a> with <strong>Thinking toggles</strong>. <a href="https://unsloth.ai">Download</a> for Mac, Windows and Linux.</a> <a href="github.com/unslothai/unsloth">GitHub repo</a></li>
    <li>Developer Role Support so Qwen3.8 can work in agentic tools like Codex and more!</li>
    <li>Tool calling improvements: Makes parsing nested objects to make tool calling succeed more.</li>
   <li>See below for 4-bit Qwen3.8-27B run inside of Unsloth Desktop:</li>
</div>

<img width="600" alt="qwen3.8 unsloth desktop" src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FSqxs6NjShWrLfRKhDy1m%2Fvolcano%202.gif?alt=media&token=395274a0-b437-403a-8a01-8e8502f9d225" />

Analysis of best Qwen3.8 GGUF providers. Unsloth Dynamic v3.0 delivers >10% top-1% better accuracy at the same size compared to every other provider. [Read more](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)

<img width="600" alt="qwen3.8 unsloth desktop" src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fl9piThTmmUsePst5w3F2%252Fimage.png%3Falt%3Dmedia%26token%3D821e88e2-ad4e-40bf-adf2-e13837228e82&width=768&dpr=3&quality=100&sign=dd5f7cb5&sv=2" />


---

# Qwen3.8-27B

Following the widespread community adoption of the Qwen3.5 and Qwen3.6 series, we are pleased to introduce Qwen3.8, the most capable generation in the Qwen open-model family to date.

Built on the architectural foundation of Qwen3.5, Qwen3.8 delivers substantial gains across coding, professional work, research, and long-horizon agentic tasks. Qwen3.8-27B brings these advances to a compact, deployment-friendly dense model: a native vision-language model that understands images and videos, with flexible thinking control, designed to carry complex, multi-step tasks through to completion with greater reliability.

## Qwen3.8 Highlights

Qwen3.8-27B features the following enhancements:
- **Core Capabilities**: Comprehensive improvements across coding, professional work, research, and long-horizon agentic tasks.
- **Agent Execution**: Stronger autonomous planning and better handling of environment feedback, leading to more reliable end-to-end task completion.
- **Downstream Compatibility**: Broader support for popular harnesses and development tools, making it easier to integrate into your existing stack.
- **Flexible Thinking Control**: Thinking mode is on by default and can be disabled per request; reasoning depth can be tuned with `reasoning_effort`, and reasoning context from historical messages is retained via `preserve_thinking`.
- **Vision-Language Understanding**: Native support for image and video understanding, from STEM diagrams and documents to hour-scale videos.


## Model Overview

- Type: Causal Language Model with Vision Encoder
- Training Stage: Pre-training & Post-training
- Language Model
    - Number of Parameters: 27B
    - Hidden Dimension: 5120
    - Token Embedding: 248,320 (Padded)
    - Number of Layers: 64
    - Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
    - Gated DeltaNet:
        - Number of Linear Attention Heads: 48 for V and 16 for QK
        - Head Dimension: 128
    - Gated Attention:
        - Number of Attention Heads: 24 for Q and 4 for KV
        - Head Dimension: 256
        - Rotary Position Embedding Dimension: 64
    - Feed Forward Network:
        - Intermediate Dimension: 17,408
    - LM Output: 248,320 (Padded)
    - MTP (Multi-Token Prediction): trained with multiple steps
- Context Length: 262,144 natively and extensible up to 1,000,000 tokens.

## Best Practices

To achieve optimal performance, we recommend the following settings:

1. **Sampling Parameters**: We suggest using the following sets of sampling parameters:  
    
    - Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
    - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
    
    For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetition. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.

2. **Adequate Output Length**: To optimize performance on agentic tasks, we recommend allocating sufficient output length to allow the model to generate detailed and comprehensive responses. For frameworks that support separate token limits for internal reasoning and final outputs, we suggest the following configuration within the 1M context length:
    
    - Reasoning Content: Set the maximum output length to 262,144 tokens.
    - Final Response: Set the maximum output length to 131,072 tokens.

    These settings provide the necessary capacity for complex reasoning while ensuring ample space for high-quality final deliverables.

3. **Processing Ultra-Long Texts**: Qwen3.8-27B natively supports context lengths of up to 262,144 tokens. For long-horizon tasks where the total length (including both input and output) exceeds this limit, we recommend using RoPE scaling techniques to handle long texts effectively, e.g., YaRN.

4. **Long Video Understanding**: To optimize inference efficiency for plain text and images, the `size` parameter in the released `video_preprocessor_config.json` is conservatively configured. It is recommended to set the `longest_edge` parameter in the video_preprocessor_config file to 469,762,048 (corresponding to 224k video tokens) to enable higher frame-rate sampling for hour-scale videos and thereby achieve superior performance. For example,
    ```json
    {"longest_edge": 469762048, "shortest_edge": 4096}
    ```

## Citation

If you find our work helpful, feel free to give us a cite.


```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```