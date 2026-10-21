---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - qwen
  - qwen3
  - qwen3_5
---

# Qwen3.8 27B IQ4_XS + FFN IQ3_S

This repository stores Qwen3.8 27B quantized to IQ4_XS with `ffn_gate` and `ffn_up` quantized to IQ3_S. This makes the model lighter (12877MiB on VRAM vs. Unsloth IQ4_XS's 14285MiB), allowing for a bigger context size.

## Quantization details

- Importance matrix: [cHunter789's mix of mradermacher and ubergarm imatrix files](https://huggingface.co/cHunter789/Qwen3.8-27B-i1-IQ4_KS_KT-GGUF/blob/1603fbfca97298390daf1ae03339fa279aa94be2/Qwen3.8-27B.imatrix.mradermacher-ubergarm.dat).

- Quantization command:

  ```bash
  llama-quantize \
    --imatrix Qwen3.8-27B.imatrix.mradermacher-ubergarm.dat \
    --tensor-type token_embd.weight=q8_0 \
    --tensor-type "blk.64.ffn_(.*?).weight=q4_k" \
    --tensor-type "blk.64.attn_(k|q|output).weight=q4_k" \
    --tensor-type "blk.64.attn_v.weight=q5_k" \
    --tensor-type "blk.64.nextn.eh_proj.weight=q8_0" \
    --tensor-type ffn_down=iq4_xs \
    --tensor-type ffn_up=iq3_s \
    --tensor-type ffn_gate=iq3_s \
    Qwen3.8-27B-BF16-00001-of-00002.gguf Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf IQ4_XS
  ```

- Serve command:

  With the following command, I am able to run the model with MTP and a context window of 100k tokens (q8_0 / q8_0) on 5070 (12GB VRAM) + 4060 (8GB VRAM). Token generation starts at around 60 t/s and eventually falls to 35-40 t/s.

  ```bash
  GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 llama-server \
    -m Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf -mm Qwen3.8-27B-GGUF-mmproj-BF16.gguf \
    -ag -sm tensor -ts 1.66,1 -ngl 999 -np 4 -kvu -t 8 -b 2048 -ub 512 \
    -c 100096 -ctk q8_0 -ctv q8_0 -fa on \
    --jinja \
    --spec-draft-n-max 2 \
    --spec-type draft-mtp,ngram-mod \
    --no-mmproj-offload \
    --image-min-tokens 1024 \
    --cors-origins localhost \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --repeat-penalty 1.0 \
    --reasoning-preserve \
    --alias Qwen/Qwen3.8-27B-IQ4_XS
  ```