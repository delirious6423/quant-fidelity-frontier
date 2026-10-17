---
quantized_by: ubergarm
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
tags:
- imatrix
- conversational
- qwen3_5
- ik_llama.cpp
---

## `ik_llama.cpp` imatrix Quantizations of Qwen/Qwen3.8-27B
*NOTE* `ik_llama.cpp` can also run your existing GGUFs from bartowski, unsloth, mradermacher, etc if you want to try it out before downloading my quants. Only a couple quants in this collection are compatible with mainline llamma.cpp/LMStudio/KoboldCPP/etc as mentioned in the specific description, all others require ik_llama.cpp.

Some of ik's new quants are supported with [Nexesenex/croco.cpp](https://github.com/Nexesenex/croco.cpp) fork of KoboldCPP with Windows builds. Also check for [ik_llama.cpp windows builds by Thireus here.](https://github.com/Thireus/ik_llama.cpp/releases).

These quants provide best in class perplexity for the given memory footprint.

## Big Thanks
Shout out to Wendell and the **Level1Techs** crew, the community [Forums](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826), [YouTube Channel](https://www.youtube.com/@Level1Techs)!  **BIG thanks** for providing **BIG hardware** expertise and access to run these experiments and make these great quants available to the community!!!

Also thanks to all the folks in the quantizing and inferencing community on [BeaverAI Club Discord](https://huggingface.co/BeaverAI) and on [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) for tips and tricks helping each other run, test, and benchmark all the fun new models! Thanks to huggingface for hosting all these big quants!

Finally, I *really* appreciate the support from [aifoundry.org](https://aifoundry.org) so check out their open source RISC-V based solutions!

## Quant Collection
Perplexity computed against *wiki.test.raw*. (lower is "better")

These two are just test quants for baseline perplexity comparison and not available for download here:
* `BF16` 50.894 GiB (16.002 BPW)
  - PPL over 580 chunks for n_ctx=512 = 6.9540 +/- 0.04500
* `Q8_0` 27.042 GiB (8.502 BPW)
  - PPL over 580 chunks for n_ctx=512 = 6.9554 +/- 0.04500

## MTP IQ4_KS 15.749 GiB (4.731 BPW)
PPL over 580 chunks for n_ctx=512 = 6.9938 +/- 0.04527

New improved recipe! Size includes extra `iq4_ks` mtp head so no need for `-mtprot iq4_ks` now.

<details>

<summary>👈 Secret Recipe</summary>

```bash
#!/usr/bin/env bash

custom="
# 64 Repeating Layers [0-63] + blk.64 MTP/nextn tensors

## MTP/nextn tensors
blk\.64\..*\.weight=iq4_ks

## Gated Attention/Delta Net [Blended 0-63]
blk\..*\.attn_gate\.weight=iq4_ks
blk\..*\.attn_qkv\.weight=iq4_ks
blk\..*\.attn_output\.weight=iq4_ks
blk\..*\.attn_q\.weight=iq4_ks
blk\..*\.attn_k\.weight=iq4_ks
blk\..*\.attn_v\.weight=iq4_ks
blk\..*\.ssm_alpha\.weight=q8_0
blk\..*\.ssm_beta\.weight=q8_0
blk\..*\.ssm_out\.weight=q6_0

# Dense Layers [0-63]
blk\..*\.ffn_down\.weight=iq4_ks
blk\..*\.ffn_(gate|up)\.weight=iq4_ks

# Non-Repeating Layers
token_embd\.weight=q6_0
output\.weight=q8_0
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

    #--dry-run \
./build/bin/llama-quantize \
    --custom-q "$custom" \
    --extra-output-tensor iq4_ks \
    --imatrix /mnt/data/models/ubergarm/Qwen3.8-27B-GGUF/imatrix-Qwen3.8-27B-BF16.dat \
    /mnt/data/models/ubergarm/Qwen3.8-27B-GGUF-mtp/Qwen3.8-27B-BF16-00001-of-00002.gguf \
    /mnt/data/models/ubergarm/Qwen3.8-27B-GGUF-mtp/Qwen3.8-27B-MTP-IQ4_KS.gguf \
    IQ4_KS \
    16
```

</details>

## Quick Start

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp.git
cd ik_llama.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_F16=ON
cmake --build build --config Release -j $(nproc)

# wget https://huggingface.co/ubergarm/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-MTP-IQ4_KS.gguf
model=/mnt/ai/models/ubergarm/Qwen3.8-27B-GGUF/Qwen3.8-27B-MTP-IQ4_KS.gguf
# wget wget https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/mmproj-Qwen3.8-27B-Q8_0.gguf
mmproj=/mnt/ai/models/ubergarm/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-Q8_0.gguf

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-server \
  --model "$model" \
  --alias "Qwen3.8-27B" \
  -c 131072 \
  -ctk q8_0 -ctv q8_0 \
  -ctkd q8_0 -ctvd q8_0 \
  --merge-qkv \
  -muge \
  -ngl 99 \
  -t 1 \
  -tb 1 \
  -tm 16 \
  --host 127.0.0.1 \
  --port 8080 \
  --parallel 1 \
  --jinja \
  --ctx-checkpoints 32 \
  -cram 32768 \
  --spec-type mtp:n_max=4,p_min=0.0 \
  --no-mmproj-offload \
  --mmproj "$mmproj"
```

This command keeps the mmproj on CPU/RAM and is set to 16 physical cores currently, reduce `-tm 8` etc if you have less cores.

If you have multiple CUDA GPUs add `-sm graph`.

Optional custom chat templates with `--chat-template-file ./chat_template.jinja \` which may help performance especially with some harness combinations. Pick one of these or bring your own custom chat template as desired.
```bash
# wget https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/raw/main/chat_template.jinja
# wget https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja
```

## References
* [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
* [ubergarm on quantizing LLMs and tuning GPUs with aifoundry.org](https://blog.aifoundry.org/p/adventures-in-model-quantization)
* [ubergarm-imatrix-calibration-corpus-v02.txt](https://gist.github.com/ubergarm/edfeb3ff9c6ec8b49e88cdf627b0711a?permalink_comment_id=5682584#gistcomment-5682584)
* [Getting Started Guide (out of date)](https://github.com/ikawrakow/ik_llama.cpp/discussions/258)
* [Quant Cookers Guide (out of date)](https://github.com/ikawrakow/ik_llama.cpp/discussions/434)
* [high quality imatrix MoE optimized mainline llama.cpp quants AesSedai](https://huggingface.co/AesSedai/)
