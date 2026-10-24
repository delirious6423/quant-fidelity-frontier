---
library_name: transformers
base_model:
- zai-org/GLM-5.3-Flash
---

This tiny model is intended for debugging. It is randomly initialized using the configuration adapted from [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

Structural comparison with `zai-org/GLM-5.3-Flash`:

| Feature | Source | Tiny |
|---|---:|---:|
| Checkpoint tensor elements | 321.32B | 112.09M, including 16.70K FP32 scale elements |
| Safetensors weight files | 328.34 GB across shards | 207.31 MB (`model.safetensors`) |
| Architecture / modalities | GLM5-Next; text, image, video | Same |
| Vocabulary rows / tied embeddings | 154,880 / false | Same |
| Text layers / attention cadence | 45; three KDA then DSA | 8; two complete three-KDA + DSA cycles |
| Text hidden size | 4,096 | 128 |
| KDA heads / head dimension | 64 / 128 | 64 / 128 |
| DSA heads / NoPE / value dimensions | 64 / 256 / 256 | 64 / 32 / 256 |
| Indexer heads / dimension / top-k | 32 / 128 / 2,048 | 32 / 128 / 128 |
| Routed experts / selected experts / shared experts | 288 / 8 / 1 | 32 / 8 / 1 |
| Hyper-Connections | Four manifold-constrained streams | Same |
| MTP | One DSA/MoE layer with shared embedding/head | Same |
| Vision depth / hidden / output size | 24 / 1,024 / 4,096 | 2 / 128 / 128 |
| Checkpoint precision | BF16 plus E4M3 block-FP8 and FP32 scales | Same; 32x32 blocks for reduced axes |

The KDA and DSA indexer heads remain 128-wide because vLLM's GLM5-Next KDA and Hadamard-128 indexer kernels require that ABI. The source E4M3 format is retained with reduced 32x32 blocks; the TP=4 example selects vLLM's Triton block-FP8 GEMM because H20's automatic Humming kernel requires an output scale group of at least 64. With the unchanged 32-wide FP8 K block, vLLM TP=4 partitions residual-width linear inputs by four. The 32-wide candidate reached GLM5-Next model loading but was rejected with `input_size_per_partition = 8 is not divisible by weight quantization block_k = 32`; a 64-wide residual would likewise partition to 16. The tested 128-wide checkpoint is therefore the smallest executable residual width for this required checkpoint/backend/TP combination.
Tokenizer, processor, chat template and generation assets are copied unchanged from the pinned source revision.

| File path | Size |
|------|------|
| model.safetensors | 207.3MB |


### Example usage:

- vLLM (✅ verified)

```bash
vllm serve tiny-random/glm-5.3-flash \
  --tensor-parallel-size 4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --max-model-len 4096 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.3 \
  --kernel-config '{"moe_backend":"triton","linear_backend":"triton"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'
```

- Transformers (✅ verified)

```python
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.integrations.hub_kernels import allow_all_hub_kernels
from transformers.video_utils import VideoMetadata

model_id = "tiny-random/glm-5.3-flash"
# Transformers resolves the official fine-grained FP8 kernel from the Hub.
with allow_all_hub_kernels():
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_id)
    image = Image.new("RGB", (28, 28), "red")
    video = np.stack([
        np.asarray(Image.new("RGB", (28, 28), color))
        for color in ("red", "green", "blue", "yellow")
    ])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "video", "video": video},
        {"type": "text", "text": "Describe the image and the video."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        video_metadata=[VideoMetadata(total_num_frames=4, fps=2, frames_indices=[0, 1, 2, 3])],
    ).to(model.device)
    output = model.generate(**inputs, do_sample=False, max_new_tokens=10)
print(processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Codes to create this repo (✅ verified):

<details><summary>Click to expand</summary>

```python
"""Create the GLM5-Next multimodal checkpoint in its public FP8 schema."""
import json
from pathlib import Path
from shutil import copyfile

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForImageTextToText, set_seed

source_model_id = "zai-org/GLM-5.3-Flash"
save_folder = "/tmp/tiny-random/glm-53-flash"
source_revision = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
set_seed(42)
folder = Path(save_folder)
folder.mkdir(parents=True, exist_ok=True)
raw = json.loads(Path(hf_hub_download(
    source_model_id, "config.json", revision=source_revision)).read_text())
raw["quantization_config"]["weight_block_size"] = [32, 32]
text = raw["text_config"]
text.update(
    hidden_size=128,
    num_hidden_layers=8,
    # Retain the source's TP=4-friendly global head topology. The KDA
    # and DSA runtime kernels operate on 16 heads per TP rank.
    num_attention_heads=64,
    num_key_value_heads=64,
    intermediate_size=128,
    moe_intermediate_size=128,
    n_routed_experts=32,
    num_experts_per_tok=8,
    q_lora_rank=128,
    # FA3 sparse-MQA uses this latent as q_v and requires >=256.
    kv_lora_rank=256,
    qk_nope_head_dim=32,
    qk_rope_head_dim=0,
    # Retain source value-head width required by sparse-MQA FA3.
    v_head_dim=256,
    index_head_dim=128,
    # The Hadamard-128 indexer fused kernel accepts 32 or 64; source is 32.
    index_n_heads=32,
    index_topk=128,
    index_kpool=4,
    linear_head_dim=128,
    first_k_dense_replace=3,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "indexed_attention"] * 2,
    mlp_layer_types=["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"],
    indexer_types=["full"] * 8,
)
text["linear_attn_config"].update(
    head_dim=128,
    num_heads=64,
    kda_layers=[0, 1, 2, 4, 5, 6],
    full_attn_layers=[3, 7],
)
# TP=4 shards vision attention too: retain four visual heads and a
# 32-wide visual head dimension for an executable multimodal tower.
raw["vision_config"].update(
    depth=2, hidden_size=128, intermediate_size=256, num_heads=4,
    out_hidden_size=128, projection_intermediate_size=64, image_size=28, patch_size=14,
)

config_for_build = dict(raw)
config_for_build.pop("quantization_config")
(folder / "config.json").write_text(json.dumps(config_for_build, indent=2) + "\n")
config = AutoConfig.from_pretrained(folder)
model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)

with torch.no_grad():
    for name, value in model.named_parameters():
        if name.endswith("A_log"):
            value.zero_()
        elif name.endswith("dt_bias"):
            value.fill_(1)
        elif name.endswith("e_score_correction_bias"):
            value.zero_()
        elif name.endswith("weight") and "norm" in name:
            value.fill_(1)
        else:
            value.normal_(mean=0.0, std=0.02)

state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}

# Transformers builds the published GLM5-Next graph with convenient
# module names (``attn_hc`` / ``forget_gate`` / fused ``conv1d``), while
# both the pinned source checkpoint and vLLM consume the public checkpoint
# schema below.  Convert it before quantization: this is a lossless key
# rewrite except for splitting the concatenated Q|K|V convolution weight.
for name in list(state):
    replacement = (
        name.replace(".attn_hc.base", ".hc_attn_base")
        .replace(".attn_hc.fn", ".hc_attn_fn")
        .replace(".attn_hc.scale", ".hc_attn_scale")
        .replace(".ffn_hc.base", ".hc_ffn_base")
        .replace(".ffn_hc.fn", ".hc_ffn_fn")
        .replace(".ffn_hc.scale", ".hc_ffn_scale")
        .replace(".self_attn.forget_gate.", ".self_attn.")
    )
    if replacement != name:
        state[replacement] = state.pop(name)
projection_size = text["linear_attn_config"]["num_heads"] * text["linear_attn_config"]["head_dim"]
for name in list(state):
    if not name.endswith(".self_attn.conv1d.weight"):
        continue
    prefix = name.removesuffix(".conv1d.weight")
    q, k, v = state.pop(name).split(projection_size, dim=0)
    state[prefix + ".q_conv1d.weight"] = q.contiguous()
    state[prefix + ".k_conv1d.weight"] = k.contiguous()
    state[prefix + ".v_conv1d.weight"] = v.contiguous()
block_out, block_in = raw["quantization_config"]["weight_block_size"]

def pack(name: str, weight: torch.Tensor) -> None:
    rows, columns = weight.shape
    assert rows % block_out == 0 and columns % block_in == 0, (name, weight.shape)
    blocks = weight.float().reshape(rows // block_out, block_out, columns // block_in, block_in)
    scales = (blocks.abs().amax(dim=(1, 3)) / 448).clamp_min(1e-12).to(torch.float32)
    state[name] = (blocks / scales[:, None, :, None]).clamp(-448, 448).reshape(weight.shape).to(
        torch.float8_e4m3fn
    ).contiguous()
    state[name.removesuffix("weight") + "weight_scale_inv"] = scales.contiguous()

def keep_bf16(name: str) -> bool:
    # GLM5-Next keeps every KDA/linear-attention tensor in BF16.  They are
    # exposed below ``self_attn`` in the Transformers graph rather than
    # under a ``linear_attention`` component name.
    kda_prefixes = tuple(
        f"model.language_model.layers.{index}.self_attn."
        for index in (0, 1, 2, 4, 5, 6)
    )
    return (
        name in {"model.language_model.embed_tokens.weight", "lm_head.weight"}
        or name.startswith(kda_prefixes)
        # GLM5-Next's DSA value expansion deliberately remains BF16.
        or name.endswith(".self_attn.kv_b_proj.weight")
        or ".norm." in name
        or ".indexer." in name
        or ".linear_attention." in name
        or name.endswith(".mlp.gate.weight")
        or name.endswith(".mlp.gate.e_score_correction_bias")
        or name.startswith("model.visual.")
    )

# The upstream skip list contains absolute source-layer paths.  Retain
# only the reduced graph's BF16 components so the Transformers FP8 loader
# constructs KDA, DSA value, router, indexer and vision modules correctly.
exclusions = {"model.language_model.embed_tokens", "lm_head", "model.visual"}
for module_name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and keep_bf16(module_name + ".weight"):
        exclusions.add(module_name)
raw["quantization_config"]["modules_to_not_convert"] = sorted(exclusions)

for name in list(state):
    if not name.endswith(".mlp.experts.gate_up_proj"):
        continue
    prefix = name.removesuffix(".gate_up_proj")
    gate_up = state.pop(name)
    down = state.pop(prefix + ".down_proj")
    for expert in range(gate_up.shape[0]):
        gate, up = gate_up[expert].chunk(2, dim=0)
        pack(f"{prefix}.{expert}.gate_proj.weight", gate)
        pack(f"{prefix}.{expert}.up_proj.weight", up)
        pack(f"{prefix}.{expert}.down_proj.weight", down[expert])

for name in list(state):
    value = state[name]
    if not (name.endswith(".weight") and value.ndim == 2) or keep_bf16(name):
        continue
    pack(name, value)

# vLLM consumes the next-token predictor from layer index num_hidden_layers.
# It is a DSA/MoE layer, not a KDA/mHC layer, and shares the global token
# embedding and LM head exactly like the source checkpoint.
base_prefix = "model.language_model.layers.7."
mtp_prefix = "model.language_model.layers.8."
for name, value in list(state.items()):
    # GLM5-Next's embedded draft DSA/MoE layer has no Hyper-Connections;
    # source layer 45 likewise omits every hc_* tensor.
    if name.startswith(base_prefix) and ".hc_" not in name:
        state[mtp_prefix + name.removeprefix(base_prefix)] = value.clone()
state[mtp_prefix + "eh_proj.weight"] = torch.empty(128, 256, dtype=torch.bfloat16).normal_(0, 0.02)
for name in ("enorm.weight", "hnorm.weight", "shared_head.norm.weight"):
    state[mtp_prefix + name] = torch.ones(128, dtype=torch.bfloat16)

save_file(state, folder / "model.safetensors", metadata={"format": "pt"})
(folder / "config.json").write_text(json.dumps(raw, indent=2) + "\n")

assets = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "chat_template.json", "processor_config.json", "preprocessor_config.json",
    "video_preprocessor_config.json", "generation_config.json", "LICENSE",
}
for name in sorted(set(list_repo_files(source_model_id, revision=source_revision)) & assets):
    copyfile(hf_hub_download(source_model_id, name, revision=source_revision), folder / name)
```

</details>

### Printing the model:

<details><summary>Click to expand</summary>

```text
Glm5NextForConditionalGeneration(
  (model): Glm5NextModel(
    (visual): Glm5NextVisionModel(
      (patch_embed): Glm5NextVisionPatchEmbed(
        (proj): Conv3d(3, 128, kernel_size=(2, 14, 14), stride=(2, 14, 14))
      )
      (rotary_pos_emb): Glm5NextVisionRotaryEmbedding()
      (blocks): ModuleList(
        (0-1): 2 x Glm5NextVisionBlock(
          (norm1): Glm5NextRMSNorm((128,), eps=1e-05)
          (norm2): Glm5NextRMSNorm((128,), eps=1e-05)
          (attn): Glm5NextVisionAttention(
            (qkv): Linear(in_features=128, out_features=384, bias=True)
            (proj): Linear(in_features=128, out_features=128, bias=True)
            (q_norm): Glm5NextRMSNorm((32,), eps=1e-05)
            (k_norm): Glm5NextRMSNorm((32,), eps=1e-05)
          )
          (mlp): Glm5NextVisionMLP(
            (gate_proj): Linear(in_features=128, out_features=256, bias=True)
            (up_proj): Linear(in_features=128, out_features=256, bias=True)
            (down_proj): Linear(in_features=256, out_features=128, bias=True)
            (act_fn): SiLUActivation()
          )
        )
      )
      (merger): Glm5NextVisionPatchMerger(
        (proj): Linear(in_features=128, out_features=128, bias=False)
        (post_projection_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
        (gate_proj): Linear(in_features=128, out_features=64, bias=False)
        (up_proj): Linear(in_features=128, out_features=64, bias=False)
        (down_proj): Linear(in_features=64, out_features=128, bias=False)
        (act1): GELU(approximate='none')
        (act_fn): SiLUActivation()
      )
      (downsample): Conv2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
      (post_layernorm): Glm5NextRMSNorm((128,), eps=1e-05)
    )
    (language_model): Glm5NextTextModel(
      (embed_tokens): Embedding(154880, 128, padding_idx=154820)
      (layers): ModuleList(
        (0-2): 3 x Glm5NextTextDecoderLayer(
          (self_attn): Glm5NextTextLinearAttention(
            (q_proj): Linear(in_features=128, out_features=8192, bias=False)
            (k_proj): Linear(in_features=128, out_features=8192, bias=False)
            (v_proj): Linear(in_features=128, out_features=8192, bias=False)
            (conv1d): Conv1d(24576, 24576, kernel_size=(4,), stride=(1,), padding=(3,), groups=24576, bias=False)
            (forget_gate): Glm5NextTextForgetGate(
              (f_a_proj): Linear(in_features=128, out_features=128, bias=False)
              (f_b_proj): Linear(in_features=128, out_features=8192, bias=False)
            )
            (b_proj): Linear(in_features=128, out_features=64, bias=False)
            (g_a_proj): Linear(in_features=128, out_features=128, bias=False)
            (g_b_proj): Linear(in_features=128, out_features=8192, bias=False)
            (o_norm): Glm5NextTextRMSNormGated()
            (o_proj): Linear(in_features=8192, out_features=128, bias=False)
          )
          (mlp): Glm5NextTextMLP(
            (gate_proj): Linear(in_features=128, out_features=128, bias=False)
            (up_proj): Linear(in_features=128, out_features=128, bias=False)
            (down_proj): Linear(in_features=128, out_features=128, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): Glm5NextTextRMSNorm((128,), eps=1e-05)
          (post_attention_layernorm): Glm5NextTextRMSNorm((128,), eps=1e-05)
          (attn_hc): Glm5NextTextHyperConnection(
            (input_norm): Glm5NextTextUnweightedRMSNorm()
          )
          (ffn_hc): Glm5NextTextHyperConnection(
            (input_norm): Glm5NextTextUnweightedRMSNorm()
          )
        )
        (3): Glm5NextTextDecoderLayer(
          (self_attn): Glm5NextTextAttention(
            (q_a_proj): Linear(in_features=128, out_features=128, bias=False)
            (q_a_layernorm): Glm5NextTextRMSNorm((128,), eps=1e-05)
            (q_b_proj): Linear(in_features=128, out_features=2048, bias=False)
            (kv_a_proj_with_mqa): Linear(in_features=128, out_features=256, bias=False)
            (kv_a_layernorm): Glm5NextTextRMSNorm((256,), eps=1e-05)
            (kv_b_proj): Linear(in_features=256, out_features=18432, bias=False)
            (o_proj): Linear(in_features=16384, out_features=128, bias=False)
            (indexer): Glm5NextTextIndexer(
              (wq_b): Linear(in_features=128, out_features=4096, bias=False)
              (wk): Linear(in_features=128, out_features=128, bias=False)
              (k_norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True, bias=True)
              (weights_proj): Linear(in_features=128, out_features=32, bias=False)
            )
          )
          (mlp): Glm5NextTextMoE(
            (experts): Glm5NextTextExperts()
            (gate): Glm5NextTextTopkRouter()
            (shared_experts): Glm5NextTextMLP(
              (gate_proj): Linear(in_features=128, out_features=128, bias=False)
              (up_proj): Linear(in_features=128, out_features=128, bias=False)
              (down_proj): Linear(in_features=128, out_features=128, bias=False)
              (act_fn): SiLUActivation()
            )
          )
          (input_layernorm): Glm5NextTextRMSNorm((128,), eps=1e-05)
          (post_attention_layernorm): Glm5NextTextRMSNorm((128,), eps=1e-05)
          (attn_hc): Glm5NextTextHyperConnection(
            (input_norm): Glm5NextTextUnweightedRMSNorm()
          )
          (ffn_hc): Glm5NextTextHyperConnection(
            (input_norm): Glm5NextTextUnweightedRMSNorm()
          )
        )
        (4-6): 3 x Glm5NextTextDecoderLayer(
          (self_attn): Glm5NextTextLinearAttention(
            (q_proj): Linear(in_features=128, out_features=8192, bias=False)
            (k_proj): Linear(in_features=128, out_features=8192, bias=False)
            (v_proj): Linear(in_features=128, out_features=8192, bias=False)
            (conv1d): Conv1d(24576, 24576, kernel_size=(4,), stride=(1,), padding=(3,), groups=24576, bias=False)
            (forget_gate): Glm5NextTextForgetGate(
              (f_a_proj): Linear(in_features=128, out_features=128, bias=False)
              (f_b_proj): Linear(in_features=128, out_features=8192, bias=False)
            )
            (b_proj): Linear(in_features=128, out_features=64, bias=False)
            (g_a_proj): Linear(in_features=128, out_features=128, bias=False)
            (g_b_proj): Linear(in_features=128, out_features=8192, bias=False)
            (o_norm): Glm5NextTextRMSNormGated()
            (o_proj): Linear(in_features=8192, out_features=128, bias=False)
          )
          (mlp): Glm5NextTextMoE(
            (experts): Glm5NextTextExperts()
            (gate): Glm5NextTextTopkRouter()
            (shared_experts): Glm5NextTextMLP(
              (gate_proj): Linear(in_features=128, out_features=128, bias=False)
              (up_proj): Linear(in_features=128, out_features=128, bias=False)
              (down_proj): Linear(in_features=128, out_features=128, bias=False)
              (act_fn): SiLUActivation()
            )
          )
          (input_layernorm): Glm5NextTextRMSNorm((128,), eps=1e-05)
          (post_attention_layernorm): Glm5NextTextRMSNorm((128,), eps=1e-05)
          (attn_hc): Glm5NextTextHyperConnection(
            (input_norm): Glm5NextTextUnweightedRMSNorm()
          )
          (ffn_hc): Glm5NextTextHyperConnection(
            (input_norm): Glm5NextTextUnweightedRMSNorm()
          )
        )
        (7): Glm5NextTextDecoderLayer(
          (self_attn): Glm5NextTextAttention(
            (q_a_proj): Linear(in_features=128, out_features=128, bias=False)
            (q_a_layernorm): Glm5NextTextRMSNorm((128,), eps=1e-05)
            (q_b_proj): Linear(in_features=128, out_features=2048, bias=False)
            (kv_a_proj_with_mqa): Linear(in_features=128, out_features=256, bias=False)
            (kv_a_layernorm): Glm5NextTextRMSNorm((256,), eps=1e-05)
            (kv_b_proj): Linear(in_features=256, out_features=18432, bias=False)
            (o_proj): Linear(in_features=16384, out_features=128, bias=False)
            (indexer): Glm5NextTextIndexer(
              (wq_b): Linear(in_features=128, out_features=4096, bias=False)
              (wk): Linear(in_features=128, out_features=128, bias=False)
              (k_norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True, bias=True)
              (weights_proj): Linear(in_features=128, out_features=32, bias=False)
            )
          )
          (mlp): Glm5NextTextMoE(
            (experts): Glm5NextTextExperts()
            (gate): Glm5NextTextTopkRouter()
            (shared_experts): Glm5NextTextMLP(
              (gate_proj): Linear(in_features=128, out_features=128, bias=False)
              (up_proj): Linear(in_features=128, out_features=128, bias=False)
              (down_proj): Linear(in_features=128, out_features=128, bias=False)
              (act_fn): SiLUActivation()
            )
          )
          (input_layernorm): Glm5NextTextRMSNorm((128,), eps=1e-05)
          (post_attention_layernorm): Glm5NextTextRMSNorm((128,), eps=1e-05)
          (attn_hc): Glm5NextTextHyperConnection(
            (input_norm): Glm5NextTextUnweightedRMSNorm()
          )
          (ffn_hc): Glm5NextTextHyperConnection(
            (input_norm): Glm5NextTextUnweightedRMSNorm()
          )
        )
      )
      (norm): Glm5NextTextRMSNorm((128,), eps=1e-05)
      (hc_head): Glm5NextTextHyperHead()
    )
  )
  (lm_head): Linear(in_features=128, out_features=154880, bias=False)
)
```

</details>

### Test environment:

- Pillow: 12.3.0
- accelerate: 1.13.0
- huggingface_hub: 2.3.0
- kernels: 0.17.2
- numpy: 2.5.2
- safetensors: 0.8.0
- tokenizers: 0.23.3
- torch: 2.13.0+cu126
- transformers: 5.20.2
- vllm: 0.1.dev1+g5a3eba034.d20260730.cu126

### Change log:

- 2026-10-24: Initial version.

This repo is generated by GPT-6.1-Sol