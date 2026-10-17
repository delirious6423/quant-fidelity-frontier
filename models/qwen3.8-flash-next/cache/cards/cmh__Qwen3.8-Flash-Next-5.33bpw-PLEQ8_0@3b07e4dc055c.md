---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next — 5.33 BPW PLEQ8_0

[Download](https://huggingface.co/cmh/Qwen3.8-Flash-Next-5.33bpw-PLEQ8_0/resolve/main/Qwen3.8-Flash-Next.gguf)


## Model Details

| Property        | Value       |
| --------------- | ----------- |
| File size | 109.80 GiB |
| Total BPW   | 5.330    |
| Embedding       | Q8_0        |
| BPW without Embedding | 3.9936     |
| MoE avg BPW  | 3.8403     |

## Tensor Distribution *


![block_bpw](https://cdn-uploads.huggingface.co/production/uploads/631f1e015ba8c02634093d59/xLg_DcOH-1ACKJ9WMLju7.png)

<small>* excluding Non-MoE and n-gram embedding</small>

## Recommended llama.cpp configurations

Tested configurations for 8 logical cores, 64 GB of RAM and 12 GB of VRAM:

Headless, full context:

```
./llama-server \
  -fit off \
  -t 8 \
  -np 1 \
  -lzm on \
  -ctk q8_0 \
  -ctv q8_0 \
  -ngl 99 \
  -cmoe \
  -c 0 \
  --reasoning-preserve \
  --reasoning-effort xhigh \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0.1 \
  --presence-penalty 0.0 \
  -m Qwen3.8-Flash-Next.gguf
```

Desktop, with vision but half context to save some ram for the desktop environment:

```
./llama-server \
  -fit off \
  -t 8 \
  -np 1 \
  -lzm on \
  -b 512 \
  -ub 512 \
  -kvu \
  -ctk q8_0 \
  -ctv q8_0 \
  -ngl 99 \
  -ncmoe 46 \
  -cram 0 \
  --ctx-checkpoints 8 \
  --checkpoint-min-step 1024 \
  -c 131072 \
  --reasoning-preserve \
  --reasoning-effort xhigh \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0.1 \
  --presence-penalty 0.0 \
  --image-min-tokens = 1024 \
  --no-mmproj-offload \
  --mmproj mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  -m Qwen3.8-Flash-Next.gguf

```

## Imatrix

[imatrix_unsloth.gguf_file](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/imatrix_unsloth.gguf_file)

## Quantization Recipe

```
^per_layer_token_embd\.weight$=q8_0
^blk\.\d+\.attn_k_norm\.weight$=f32
^blk\.\d+\.attn_q_norm\.weight$=f32
^blk\.\d+\.ffn_gate_inp\.weight$=f32
^blk\.\d+\.ffn_gate_inp_shexp\.weight$=f32
^blk\.\d+\.ssm_a$=f32
^blk\.\d+\.ssm_conv1d\.weight$=f32
^blk\.\d+\.ssm_dt\.bias$=f32
^blk\.\d+\.ssm_norm\.weight$=f32
^blk\.\d+\.ssm_alpha\.weight$=f32
^blk\.\d+\.ssm_beta\.weight$=f32
^blk\.\d+\.hc_attn_norm\.weight$=f32
^blk\.\d+\.hc_ffn_norm\.weight$=f32
^blk\.\d+\.hc_attn_inject\.weight$=f32
^blk\.\d+\.hc_ffn_inject\.weight$=f32
^blk\.\d+\.indexer\.q_norm\.weight$=f32
^blk\.\d+\.indexer\.k_norm\.weight$=f32
^blk\.\d+\.ple_norm_conv\.weight$=f32
^blk\.\d+\.ple_norm_key\.weight$=f32
^blk\.\d+\.ple_norm_query\.weight$=f32
^blk\.\d+\.ple_conv1d\.weight$=f32
^output_hc_norm\.weight$=f32
^blk\.\d+\.ffn_down_exps\.weight$=iq4_nl
^blk\.47\.ffn_up_exps\.weight$=q4_k
^blk\.(1|2|3|4|5|43|44|45|46)\.ffn_up_exps\.weight$=iq4_xs
^blk\.(0|31|32|33|34|35|36|37|38|39|40|41|42)\.ffn_up_exps\.weight$=iq3_xxs
^blk\.(6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30)\.ffn_up_exps\.weight$=iq3_s
^blk\.47\.ffn_gate_exps\.weight$=q4_k
^blk\.(1|2|3|4|5|43|44|45|46)\.ffn_gate_exps\.weight$=iq4_xs
^blk\.(0|31|32|33|34|35|36|37|38|39|40|41|42)\.ffn_gate_exps\.weight$=iq3_xxs
^blk\.(6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30)\.ffn_gate_exps\.weight$=iq3_s
```

----

Credits to: 
- [Thireus](https://gguf0.thireus.com/quant_assign.html) since it's taking inspiration from Qwen3.5-122B-A10B's recipe.
- [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF) for the mmproj and [llama.cpp ](https://github.com/ggml-org/llama.cpp).
- [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) for the BF16 and the imatrix. 
