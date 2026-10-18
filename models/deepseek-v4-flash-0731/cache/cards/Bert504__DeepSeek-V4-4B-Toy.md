---
license: mit
language:
- zh
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- Toy
- 4B
- GGUF
- Deepseekv4
---

# Deepseek v4 4B Toy Model (Proof of Concept)

This is a toy model implementing the Deepseek v4 architecture, with approximately 4 billion parameters. This model serves as a demonstration and proof of concept that you can create and train your own model at a chosen size, and ultimately run it with `llama.cpp`.

I built this using the Transformers library and `llama.cpp`, and encountered a few obstacles along the way. Below, I share my full recipe and the workarounds I developed.

**Notice: The model uploaded is not fully trained and does not generate usable outputs. It should only be considered as a proof-of-concept example.**

---

| Settings | Toy 4B | Flash 284B |
| :--- | :--- | :--- |
| layer (block) count | 33 | 43 |
| context_length | 1048576 |1048576 |
| embedding_length | 1024 | 4096 |
| attention.head_count | 32 | 64 |
| expert per token | 4 | 6 |
| attention.q_lora_rank | 256 | 1024 |
| expert_feed_forward_length | 512 | 2048 |
| expert_count | 64 | 256 |
| hash MoE layer count | 0 | 2 |
| sparse_attention ratio | 4 | 4 |
| heavily_compressed_attention | 128 | 128 |
| tokenizer | DeepSeek V3 | DeepSeek V3 |

## Recipe for a customized model

### Step 1: Configure and save your model

Configure your model parameters using the Transformers library (I'm using version 5.15.0), then save the model.

```python
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, AutoTokenizer

config = DeepseekV4Config(
    vocab_size=129280,
    hidden_size=1024,
    moe_intermediate_size=512,
    num_hidden_layers=33,
    num_attention_heads=32,
    num_key_value_heads=1,
    head_dim=512,
    q_lora_rank=256,
    n_routed_experts=64,
    n_shared_experts=1,
    num_experts_per_tok=4,
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    routed_scaling_factor=1.5,
    max_position_embeddings=1048576,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1.0e-6,
    swiglu_limit=10.0,
    sliding_window=128,
    o_groups=8,
    o_lora_rank=256,
    index_n_heads=32,
    index_head_dim=64,
    index_topk=256,
    num_nextn_predict_layers=1,
    hidden_act="silu",
    rms_norm_eps=1.0e-6,
    use_cache=True,
    bos_token_id=0,
    eos_token_id=1,
    tie_word_embeddings=False,
    partial_rotary_factor=None,
    attention_bias=False,
    mlp_bias=False,
    attention_dropout=0.0,
    mlp_layer_types=["moe"] * 33
)

model = DeepseekV4ForCausalLM(config)
```

### Step 2: Fix model key names with `fix_format.py`

There are mismatches between the key name format saved by Transformers and the format expected by `llama.cpp`. They must be fixed before conversion.

Run the script as follows:

```bash
fix_format.py model_folder_name fixed_model_folder_name
```

### Step 3: Add lines to support non-MXFP4 experts (and another mismatch fix)

In the Deepseek v4 conversion script included in `llama.cpp`, MoE experts are expected to already be quantized to MXFP4. However, for a newly created model, the experts are still in FP32 or FP16 format. I had to add a few lines to make the script work with non-quantized experts.

See the changes here:  
[https://github.com/ggml-org/llama.cpp/commit/2539badcbb0fbca280baed1fc44f20afcf18c1ca](https://github.com/ggml-org/llama.cpp/commit/2539badcbb0fbca280baed1fc44f20afcf18c1ca)

(This commit also fixes another key-name mismatch in the conversion script.)

### Step 4: Convert and quantize the model

Navigate to your `llama.cpp` folder and run:

```bash
python convert_hf_to_gguf.py fixed_model_folder_name --outfile toymodel.gguf
llama-quantize.exe toymodel.gguf dsv4-4B-toy_Q8_MXFP4.gguf 38
```

Now your model is saved with MXFP4 experts.

---

## Training of This Model

To verify that the conversion recipe is correct, I trained the model on approximately 7 million tokens for 7 epochs, using a collection of Chinese corpora that I happened to have on hand. 

![image](https://cdn-uploads.huggingface.co/production/uploads/646f2f1d075e11ca78dab2bc/rFf72yzVwwUVNi4ZaJJSe.png)

Training loss has not stopped dropping, and the model output is still somewhat gibberish, but the output highly resembles what has appeared in the training corps, both before and after the conversion and quantization. So I think the recipe is prooved to be successful (that it doesn't significantly harm or change the model's outputs).

The model is only at pretraining stage, not SFT, not RLHF, etc.