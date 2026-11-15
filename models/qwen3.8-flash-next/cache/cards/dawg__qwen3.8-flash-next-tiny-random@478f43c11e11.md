---
library_name: transformers
base_model:
- Qwen/Qwen3.8-Flash-Next
---

This tiny model is intended for debugging. It is randomly initialized using the configuration adapted from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

Structural comparison with `Qwen/Qwen3.8-Flash-Next`:

| Feature | Source | Tiny |
|---|---|---|
| Architecture / modalities | Qwen4Exp; text, image, video | Same |
| Vocabulary / tied embeddings | 248,320 / false | Same |
| Backbone layers / hidden size | 48 / 2,560 | 8 / 8 |
| Attention Q/KV heads; head dim | 24/2; 256 | 12/1; 32 |
| Attention cycle | 3 linear + 1 QSA | Same; two cycles |
| Linear key/value heads; head dim | 16/48; 128 | 4/12; 32 |
| Routed experts / top-k | 512 / 10 | 32 / 10 |
| Expert/shared intermediate size | 640 / 640 | 128 / 128 |
| Hyper-Connections / mixing rank | 4 / 320 | 4 / 4 |
| QSA heads / dimension / token budget / compression | 4 / 128 / 2,048 / 4 | Same |
| PLE layer / n-gram orders / heads per order | 2 / 2,3 / 8 | Same |
| PLE rows / embedding columns / dtype | 320,001,536 / 160 / BF16 | 9,216 / 1 / BF16 |
| PLE hash bucket base | 20,000,000 | 512; official prime layout |
| PLE convolution / serialized table shards | 4 / 128 | Same |
| MTP | 1 QSA layer; HC fusion; shared token embeddings | Same |
| Vision depth / hidden size / output size | 27 / 1,152 / 2,560 | 2 / 64 / 8 |
| Vision patch / temporal patch / merge | 16 / 2 / 2 | Same |

PLE retains the source n-gram orders, head count, hash/prime-layout algorithm,
projections and convolution with reduced auxiliary hash capacity. The main token
vocabulary and IDs are unchanged.
The Qwen4Exp Hyper-Connections/MTP TP=4 path requires an eight-wide hidden state:
the four-wide configuration creates non-contiguous per-rank MTP block outputs in
vLLM. The two untied BF16 token embedding/head matrices at width 8 occupy
7,946,240 bytes; the complete verified checkpoint is 10,886,632 bytes.
Tokenizer and processor assets are copied unchanged from the pinned source.
vLLM 0.32.0 was verified on one four-NVIDIA-H20 node with TP=4, CUDA Graph,
three-token MTP, and real text, image, and four-frame video requests.

| File path | Size |
|------|------|
| model.safetensors | 10.9MB |


### Example usage:

- vLLM (✅ verified)

```bash
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
vllm serve tiny-random/qwen3.8-flash-next \\
  --tensor-parallel-size 4 \\
  --disable-custom-all-reduce \\
  --kernel-config '{"moe_backend":"triton"}' \\
  --gdn-prefill-backend triton \\
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \\
  --max-model-len 4096 \\
  --max-num-batched-tokens 512 \\
  --max-num-seqs 4 \\
  --gpu-memory-utilization 0.3 \\
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16]}'
```

- SGLang (unverified)

  - Run on one four-GPU node. Enables TP=4, NEXTN/MTP and CUDA Graph.
  - Unverified on the tested NVIDIA H20 worker: its CUDA 12.2 driver cannot
    load the CUDA 13 SGLang kernel and TileLang artifacts required by this
    Qwen4Exp release. Use the Test environment versions with a CUDA 13-capable
    driver for TP=4 validation.

```bash
model_id=tiny-random/qwen3.8-flash-next
sglang serve \
  --model-path $model_id \
  --tp-size 4 \
  --fp8-gemm-backend triton \
  --moe-runner-backend triton \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --context-length 4096 \
  --max-total-tokens 4096 \
  --chunked-prefill-size 512 \
  --mem-fraction-static 0.4 \
  --cuda-graph-config '{"decode": {"bs": [1, 2, 4, 8]}}' \
  --disable-custom-all-reduce
```

- Transformers (✅ verified)

```python
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen4ExpForConditionalGeneration
from transformers.video_utils import VideoMetadata

model_id = "tiny-random/qwen3.8-flash-next"
model = Qwen4ExpForConditionalGeneration.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", experts_implementation="batched_mm",
)
processor = AutoProcessor.from_pretrained(model_id)
image = Image.new("RGB", (64, 64), (220, 30, 30))
# Four predecoded RGB frames, with timestamps supplied through video_metadata.
video = np.stack([np.asarray(Image.new("RGB", (64, 64), color)) for color in
                  [(220, 30, 30), (30, 220, 30), (30, 30, 220), (220, 220, 30)]])
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "video", "video": video},
    {"type": "text", "text": "Describe the image and the video."},
]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    do_sample_frames=False,
    video_metadata=[VideoMetadata(total_num_frames=4, fps=2, frames_indices=[0, 1, 2, 3])],
).to(model.device)
output = model.generate(**inputs, do_sample=False, max_new_tokens=10)
print(processor.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
```

### Codes to create this repo (✅ verified):

<details><summary>Click to expand</summary>

```python
import json
from copy import deepcopy
from pathlib import Path
from shutil import copyfile

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForImageTextToText, set_seed
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

source_model_id = "Qwen/Qwen3.8-Flash-Next"
save_folder = "/tmp/tiny-random/qwen38-flash-next"
source_revision = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
set_seed(42)
folder = Path(save_folder)
folder.mkdir(parents=True, exist_ok=True)
raw = json.loads(Path(hf_hub_download(source_model_id, 'config.json', revision=source_revision)).read_text())
raw['text_config'].update(
    hidden_size=8, ple_embed_dim=16, ngram_vocab_size_base=512, hc_lowrank=4, num_hidden_layers=8,
    layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2,
    num_attention_heads=12, num_key_value_heads=1, head_dim=32,
    linear_num_key_heads=4, linear_num_value_heads=12,
    linear_key_head_dim=32, linear_value_head_dim=32,
    num_experts=32, moe_intermediate_size=128, shared_expert_intermediate_size=128,
)
raw['text_config']['rope_parameters']['mrope_section'] = [2, 1, 1]
raw['vision_config'].update(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=8)
(folder / 'config.json').write_text(json.dumps(raw, indent=2))
config = AutoConfig.from_pretrained(folder)
model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
backbone = model.model.language_model
text = backbone.config
width = text.hidden_size
model.mtp = torch.nn.ModuleDict({
    "fc_embedding": torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16),
    "fc_hidden": torch.nn.Linear(width, width, bias=False, dtype=torch.bfloat16),
    "pre_fc_norm_embedding": Qwen4ExpTextRMSNorm(width, eps=text.rms_norm_eps).to(torch.bfloat16),
    "pre_fc_norm_hidden": Qwen4ExpTextRMSNorm(width * text.hc_count, eps=text.rms_norm_eps).to(torch.bfloat16),
    "layers": torch.nn.ModuleList([deepcopy(backbone.layers[-1])]),
    "hyper_connection_mixer": deepcopy(backbone.hyper_connection_mixer),
})
with torch.no_grad():
    for name, param in sorted(model.named_parameters()):
        if name.endswith("A_log"):
            param.zero_()
        elif name.endswith("dt_bias"):
            param.fill_(1)
        elif "norm" in name and name.endswith("weight"):
            param.fill_(1 if "visual" in name or ".linear_attn.norm." in name else 0)
        elif name.endswith("bias"):
            param.zero_()
        else:
            torch.nn.init.normal_(param, mean=0, std=0.02)
state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
# The public checkpoint stores PLE embeddings in shard_N tensors.
for name in list(state):
    if not name.endswith(".ple.ple_embedding.ngram_embedding.weight"):
        continue
    table = state.pop(name)
    parts = text.split_ngram_parts
    rows = (table.shape[0] + parts - 1) // parts
    prefix = name.removesuffix(".weight")
    for part in range(parts):
        state[f"{prefix}.shard_{part}.weight"] = table[part * rows:(part + 1) * rows].contiguous()
save_file(state, folder / "model.safetensors", metadata={"format": "pt"})
# Preserve tokenizer, processor and generation assets exactly as published.
assets = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
    "chat_template.json", "preprocessor_config.json", "processor_config.json",
    "video_preprocessor_config.json", "generation_config.json",
}
for name in sorted(set(list_repo_files(source_model_id, revision=source_revision)) & assets):
    copyfile(hf_hub_download(source_model_id, name, revision=source_revision), folder / name)
```

</details>

### Printing the model:

<details><summary>Click to expand</summary>

```text
Qwen4ExpForConditionalGeneration(
  (model): Qwen4ExpModel(
    (visual): Qwen4ExpVisionModel(
      (patch_embed): Qwen4ExpVisionPatchEmbed(
        (proj): Conv3d(3, 64, kernel_size=(2, 16, 16), stride=(2, 16, 16))
      )
      (pos_embed): Embedding(2304, 64)
      (rotary_pos_emb): Qwen4ExpVisionRotaryEmbedding()
      (blocks): ModuleList(
        (0-1): 2 x Qwen4ExpVisionBlock(
          (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True, bias=True)
          (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True, bias=True)
          (attn): Qwen4ExpVisionAttention(
            (qkv): Linear(in_features=64, out_features=192, bias=True)
            (proj): Linear(in_features=64, out_features=64, bias=True)
          )
          (mlp): Qwen4ExpVisionMLP(
            (linear_fc1): Linear(in_features=64, out_features=128, bias=True)
            (linear_fc2): Linear(in_features=128, out_features=64, bias=True)
            (act_fn): GELUTanh()
          )
        )
      )
      (merger): Qwen4ExpVisionPatchMerger(
        (norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True, bias=True)
        (linear_fc1): Linear(in_features=256, out_features=256, bias=True)
        (act_fn): GELU(approximate='none')
        (linear_fc2): Linear(in_features=256, out_features=8, bias=True)
      )
    )
    (language_model): Qwen4ExpTextModel(
      (embed_tokens): Embedding(248320, 8)
      (layers): ModuleList(
        (0): Qwen4ExpTextDecoderLayer(
          (linear_attn): Qwen4ExpTextGatedDeltaNet(
            (conv1d): Conv1d(640, 640, kernel_size=(4,), stride=(1,), padding=(3,), groups=640, bias=False)
            (norm): Qwen4ExpTextRMSNormGated()
            (out_proj): Linear(in_features=384, out_features=8, bias=False)
            (in_proj_qkv): Linear(in_features=8, out_features=640, bias=False)
            (in_proj_z): Linear(in_features=8, out_features=384, bias=False)
            (in_proj_b): Linear(in_features=8, out_features=12, bias=False)
            (in_proj_a): Linear(in_features=8, out_features=12, bias=False)
          )
          (mlp): Qwen4ExpTextSparseMoeBlock(
            (gate): Qwen4ExpTextTopKRouter()
            (experts): Qwen4ExpTextExperts(
              (act_fn): SiLUActivation()
            )
            (shared_expert): Qwen4ExpTextMLP(
              (gate_proj): Linear(in_features=8, out_features=128, bias=False)
              (up_proj): Linear(in_features=8, out_features=128, bias=False)
              (down_proj): Linear(in_features=128, out_features=8, bias=False)
              (act_fn): SiLUActivation()
            )
            (shared_expert_gate): Linear(in_features=8, out_features=1, bias=False)
          )
          (attn_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
          (mlp_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
        )
        (1): Qwen4ExpTextDecoderLayer(
          (linear_attn): Qwen4ExpTextGatedDeltaNet(
            (conv1d): Conv1d(640, 640, kernel_size=(4,), stride=(1,), padding=(3,), groups=640, bias=False)
            (norm): Qwen4ExpTextRMSNormGated()
            (out_proj): Linear(in_features=384, out_features=8, bias=False)
            (in_proj_qkv): Linear(in_features=8, out_features=640, bias=False)
            (in_proj_z): Linear(in_features=8, out_features=384, bias=False)
            (in_proj_b): Linear(in_features=8, out_features=12, bias=False)
            (in_proj_a): Linear(in_features=8, out_features=12, bias=False)
          )
          (mlp): Qwen4ExpTextSparseMoeBlock(
            (gate): Qwen4ExpTextTopKRouter()
            (experts): Qwen4ExpTextExperts(
              (act_fn): SiLUActivation()
            )
            (shared_expert): Qwen4ExpTextMLP(
              (gate_proj): Linear(in_features=8, out_features=128, bias=False)
              (up_proj): Linear(in_features=8, out_features=128, bias=False)
              (down_proj): Linear(in_features=128, out_features=8, bias=False)
              (act_fn): SiLUActivation()
            )
            (shared_expert_gate): Linear(in_features=8, out_features=1, bias=False)
          )
          (ple): Qwen4ExpTextPLELayer(
            (ple_embedding): Qwen4ExpTextNGramEmbedding(
              (ngram_embedding): Embedding(9216, 1)
            )
            (key_proj): Linear(in_features=16, out_features=32, bias=False)
            (value_proj): Linear(in_features=16, out_features=8, bias=False)
            (norm_key): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (norm_query): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (norm_conv): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (conv1d): Conv1d(32, 32, kernel_size=(4,), stride=(1,), dilation=(3,), groups=32, bias=False)
          )
          (attn_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
          (mlp_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
        )
        (2): Qwen4ExpTextDecoderLayer(
          (linear_attn): Qwen4ExpTextGatedDeltaNet(
            (conv1d): Conv1d(640, 640, kernel_size=(4,), stride=(1,), padding=(3,), groups=640, bias=False)
            (norm): Qwen4ExpTextRMSNormGated()
            (out_proj): Linear(in_features=384, out_features=8, bias=False)
            (in_proj_qkv): Linear(in_features=8, out_features=640, bias=False)
            (in_proj_z): Linear(in_features=8, out_features=384, bias=False)
            (in_proj_b): Linear(in_features=8, out_features=12, bias=False)
            (in_proj_a): Linear(in_features=8, out_features=12, bias=False)
          )
          (mlp): Qwen4ExpTextSparseMoeBlock(
            (gate): Qwen4ExpTextTopKRouter()
            (experts): Qwen4ExpTextExperts(
              (act_fn): SiLUActivation()
            )
            (shared_expert): Qwen4ExpTextMLP(
              (gate_proj): Linear(in_features=8, out_features=128, bias=False)
              (up_proj): Linear(in_features=8, out_features=128, bias=False)
              (down_proj): Linear(in_features=128, out_features=8, bias=False)
              (act_fn): SiLUActivation()
            )
            (shared_expert_gate): Linear(in_features=8, out_features=1, bias=False)
          )
          (attn_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
          (mlp_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
        )
        (3): Qwen4ExpTextDecoderLayer(
          (self_attn): Qwen4ExpTextAttention(
            (q_proj): Linear(in_features=8, out_features=768, bias=False)
            (k_proj): Linear(in_features=8, out_features=32, bias=False)
            (v_proj): Linear(in_features=8, out_features=32, bias=False)
            (o_proj): Linear(in_features=384, out_features=8, bias=False)
            (q_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (k_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (indexer): Qwen4ExpTextQSAIndexer(
              (index_qk_proj): Linear(in_features=8, out_features=640, bias=False)
              (q_layernorm): Qwen4ExpTextRMSNorm((128,), eps=1e-06)
              (k_layernorm): Qwen4ExpTextRMSNorm((128,), eps=1e-06)
            )
          )
          (mlp): Qwen4ExpTextSparseMoeBlock(
            (gate): Qwen4ExpTextTopKRouter()
            (experts): Qwen4ExpTextExperts(
              (act_fn): SiLUActivation()
            )
            (shared_expert): Qwen4ExpTextMLP(
              (gate_proj): Linear(in_features=8, out_features=128, bias=False)
              (up_proj): Linear(in_features=8, out_features=128, bias=False)
              (down_proj): Linear(in_features=128, out_features=8, bias=False)
              (act_fn): SiLUActivation()
            )
            (shared_expert_gate): Linear(in_features=8, out_features=1, bias=False)
          )
          (attn_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
          (mlp_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
        )
        (4-6): 3 x Qwen4ExpTextDecoderLayer(
          (linear_attn): Qwen4ExpTextGatedDeltaNet(
            (conv1d): Conv1d(640, 640, kernel_size=(4,), stride=(1,), padding=(3,), groups=640, bias=False)
            (norm): Qwen4ExpTextRMSNormGated()
            (out_proj): Linear(in_features=384, out_features=8, bias=False)
            (in_proj_qkv): Linear(in_features=8, out_features=640, bias=False)
            (in_proj_z): Linear(in_features=8, out_features=384, bias=False)
            (in_proj_b): Linear(in_features=8, out_features=12, bias=False)
            (in_proj_a): Linear(in_features=8, out_features=12, bias=False)
          )
          (mlp): Qwen4ExpTextSparseMoeBlock(
            (gate): Qwen4ExpTextTopKRouter()
            (experts): Qwen4ExpTextExperts(
              (act_fn): SiLUActivation()
            )
            (shared_expert): Qwen4ExpTextMLP(
              (gate_proj): Linear(in_features=8, out_features=128, bias=False)
              (up_proj): Linear(in_features=8, out_features=128, bias=False)
              (down_proj): Linear(in_features=128, out_features=8, bias=False)
              (act_fn): SiLUActivation()
            )
            (shared_expert_gate): Linear(in_features=8, out_features=1, bias=False)
          )
          (attn_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
          (mlp_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
        )
        (7): Qwen4ExpTextDecoderLayer(
          (self_attn): Qwen4ExpTextAttention(
            (q_proj): Linear(in_features=8, out_features=768, bias=False)
            (k_proj): Linear(in_features=8, out_features=32, bias=False)
            (v_proj): Linear(in_features=8, out_features=32, bias=False)
            (o_proj): Linear(in_features=384, out_features=8, bias=False)
            (q_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (k_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (indexer): Qwen4ExpTextQSAIndexer(
              (index_qk_proj): Linear(in_features=8, out_features=640, bias=False)
              (q_layernorm): Qwen4ExpTextRMSNorm((128,), eps=1e-06)
              (k_layernorm): Qwen4ExpTextRMSNorm((128,), eps=1e-06)
            )
          )
          (mlp): Qwen4ExpTextSparseMoeBlock(
            (gate): Qwen4ExpTextTopKRouter()
            (experts): Qwen4ExpTextExperts(
              (act_fn): SiLUActivation()
            )
            (shared_expert): Qwen4ExpTextMLP(
              (gate_proj): Linear(in_features=8, out_features=128, bias=False)
              (up_proj): Linear(in_features=8, out_features=128, bias=False)
              (down_proj): Linear(in_features=128, out_features=8, bias=False)
              (act_fn): SiLUActivation()
            )
            (shared_expert_gate): Linear(in_features=8, out_features=1, bias=False)
          )
          (attn_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
          (mlp_hyper_connection): Qwen4ExpTextGatedResidual(
            (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
            (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
            (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
            (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
          )
        )
      )
      (rotary_emb): Qwen4ExpTextRotaryEmbedding()
      (hyper_connection_mixer): Qwen4ExpTextGatedResidual(
        (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
        (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
        (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
      )
    )
  )
  (lm_head): Linear(in_features=8, out_features=248320, bias=False)
  (mtp): ModuleDict(
    (fc_embedding): Linear(in_features=8, out_features=8, bias=False)
    (fc_hidden): Linear(in_features=8, out_features=8, bias=False)
    (pre_fc_norm_embedding): Qwen4ExpTextRMSNorm((8,), eps=1e-06)
    (pre_fc_norm_hidden): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
    (layers): ModuleList(
      (0): Qwen4ExpTextDecoderLayer(
        (self_attn): Qwen4ExpTextAttention(
          (q_proj): Linear(in_features=8, out_features=768, bias=False)
          (k_proj): Linear(in_features=8, out_features=32, bias=False)
          (v_proj): Linear(in_features=8, out_features=32, bias=False)
          (o_proj): Linear(in_features=384, out_features=8, bias=False)
          (q_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
          (k_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
          (indexer): Qwen4ExpTextQSAIndexer(
            (index_qk_proj): Linear(in_features=8, out_features=640, bias=False)
            (q_layernorm): Qwen4ExpTextRMSNorm((128,), eps=1e-06)
            (k_layernorm): Qwen4ExpTextRMSNorm((128,), eps=1e-06)
          )
        )
        (mlp): Qwen4ExpTextSparseMoeBlock(
          (gate): Qwen4ExpTextTopKRouter()
          (experts): Qwen4ExpTextExperts(
            (act_fn): SiLUActivation()
          )
          (shared_expert): Qwen4ExpTextMLP(
            (gate_proj): Linear(in_features=8, out_features=128, bias=False)
            (up_proj): Linear(in_features=8, out_features=128, bias=False)
            (down_proj): Linear(in_features=128, out_features=8, bias=False)
            (act_fn): SiLUActivation()
          )
          (shared_expert_gate): Linear(in_features=8, out_features=1, bias=False)
        )
        (attn_hyper_connection): Qwen4ExpTextGatedResidual(
          (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
          (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
          (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
          (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
        )
        (mlp_hyper_connection): Qwen4ExpTextGatedResidual(
          (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
          (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
          (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
          (block_inject_weight): Linear(in_features=32, out_features=4, bias=False)
        )
      )
    )
    (hyper_connection_mixer): Qwen4ExpTextGatedResidual(
      (hc_norm): Qwen4ExpTextRMSNorm((32,), eps=1e-06)
      (input_mix_weight_down): Linear(in_features=32, out_features=4, bias=False)
      (input_mix_weight_up): Linear(in_features=4, out_features=32, bias=False)
    )
  )
)
```

</details>

### Test environment:

- Pillow: 13.0.0
- flashinfer-python: 0.7.0.post1
- huggingface_hub: 2.3.0
- numpy: 2.3.5
- safetensors: 0.8.0
- tokenizers: 0.23.3
- torch: 2.13.0
- torchcodec: 0.17.0
- transformers: 5.19.0
- python (vLLM): 3.12.13
- CUDA (vLLM): 13.0
- torch (vLLM): 2.13.0+cu130
- transformers (vLLM): 5.19.0
- vllm: 0.32.0
- flashinfer-python (vLLM): 0.7.0.post1

### Change log:

- 2026-10-23: Initial version.
- 2026-10-23: Creation examples use only public dependencies; source assets are copied unchanged.
- 2026-10-24: Verified vLLM TP=4 with CUDA Graph, MTP, text, image, and video requests.

This repo is generated by GPT-6.1-Sol