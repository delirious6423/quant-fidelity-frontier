---
library_name: transformers
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
---

This tiny model is intended for debugging. It is randomly initialized using the configuration adapted from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

Note:
  - Structural comparison with the original `deepseek-ai/DeepSeek-V4.1-Flash-1017`:

| Structural feature | Original V4.1 Flash 1017 | This tiny model |
|---|---|---|
| Modalities | Vision + text | Same |
| Vocabulary / special-token IDs | 129,280; original BOS/EOS/PAD/image IDs | Same |
| Text backbone layers | 40 | 6 |
| Hidden size | 5,120 | 128 |
| Attention heads / head dim | 64 / 512 | 4 / 512 (SGLang DSV4 ABI; `head_dim >= 32`) |
| RoPE head dim | 64 | Same |
| Q/O low-rank projections | q=1,280; o=1,024; 8 output groups | q=128; o=128; 2 groups |
| Routed/shared experts | 384 / 1, top-6 | 32 / 1, top-6 |
| MoE intermediate size | 2,304 | 256 (128 per TP rank) |
| Sparse attention ratios | Backbone: 2 sliding-only, 18 ratio-2, 20 ratio-1; MTP: 3 sliding-only | Backbone: 2 sliding-only, 2 ratio-2, 2 ratio-1; MTP: 3 sliding-only |
| KV / index source layers | KV: 2,8,14,20; index: 2,8,14,20,24,28,32,36 | KV/index: 1,3; index head dim 128 |
| Candidate source layer | 20 | 3 |
| Hyper-Connections | 4 residual streams | Same |
| Engram | layers 1,14; 8 heads x 256; ~384M rows/table | layers 1,2; 4 heads x 64; 6,748 / 7,592 rows |
| DSpark/MTP | 3 stages; targets 37,38,39; Markov rank 256; 128 experts, top-3 | 3 stages; targets 3,4,5; rank 16; 8 experts, top-3 |
| Vision tower | 32 layers; width 1,024; 16 heads | 2 layers; width 64; 4 heads |
| Vision patch/downsample | patch 14; downsample 3 | Same |
| Quantization | Routed experts FP4; most other linears FP8; UE8M0 scales; 32x32 blocks | Same |
| Public checkpoint | 96,085 tensors; ~510.3 GB | 1,629 tensors; 96.7 MB |

  - Topology (sparse attention + indexer, Hyper-Connections, Engram, DSpark/MTP,
    vision, FP4/FP8 quantization) is preserved with reduced widths and counts.
    Derived from the source inference implementation on the meta device.

| File path | Size |
|------|------|
| model.safetensors | 96.7MB |


### Example usage:

- SGLang

- TP=2 inference with DSpark/MTP and CUDA Graph was validated for both
  text-only and image-plus-text requests on two NVIDIA H20 GPUs, using the
  versions listed under Test environment.
- The attention latent stays 512-wide and the indexer 128-wide per SGLang
  DSV4 ABI; all sharded widths, heads, experts, and vocab rows divide cleanly by TP=2.
- The online example enables the embedded three-stage DSpark/MTP draft explicitly.

```bash
# Requires an SGLang preview/main build with DeepSeek-V4.1 support.
# Runs the target model and its embedded three-stage DSpark/MTP draft with TP=2.
model_id=yujiepan/deepseek-v4.1-flash-1017-tiny-random
sglang serve \
  --model-path $model_id \
  --trust-remote-code \
  --tp-size 2 \
  --ep-size 1 \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --context-length 4096 \
  --max-total-tokens 4096 \
  --chunked-prefill-size 512 \
  --mem-fraction-static 0.5 \
  --disable-custom-all-reduce
```

### Codes to create this repo:

<details><summary>Click to expand</summary>

```python
import json
import hashlib
import sys
from pathlib import Path
from typing import Any, TypedDict

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import AutoTokenizer

source_model_id = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
save_folder = "/tmp/yujiepan/deepseek-v41-flash-1017-tiny-random"
source_revision = "1bae0e1088993b9b82de08f59de5be5a34e42170"
# DeepSeek-V4.1-Flash-1017 is a multimodal (vision + text) MoE model. The released checkpoint uses the
# inference-stack tensor naming (see the repo's inference/model.py), not HF-transformers naming, so
# this generator mirrors that layout. The config is nested: text_config + vision_config, with a
# shared quantization_config (fp8 weights, fp4 routed experts, ue8m0 scales, 32x32 blocks).
config = {
    "architectures": [
        "DeepseekV41ForCausalLM"
    ],
    "model_type": "deepseek_v41",
    "dtype": "bfloat16",
    "transformers_version": "5.6.0",
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "image_token_id": 129264,
    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [
            32,
            32
        ],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4"
    },
    "text_config": {
        "model_type": "deepseek_v41_text",
        "vocab_size": 129280,
        "hidden_size": 128,
        # TP=2 leaves 128 columns per rank, the minimum accepted by the
        # FlashInfer MXFP4 MoE kernel.
        "moe_intermediate_size": 256,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        # SGLang's DSV4 cache/backend ABI fixes this at 448 non-RoPE + 64 RoPE.
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "o_groups": 2,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-20,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "use_cache": True,
        "tie_word_embeddings": False,
        "max_position_embeddings": 1048576,
        "rope_theta": 10000,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536
        },
        "n_routed_experts": 32,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        # one entry per layer (backbone + MTP): 0 = sliding-window only, r = KV compressed r-to-1
        "compress_ratios": [0, 2, 2, 1, 1, 0, 0, 0, 0],
        "compress_rope_theta": 160000,
        # layers that produce (own) the shared compressed KV / index keys
        "kv_source_layer_ids": [1, 3],
        "index_source_layer_ids": [1, 3],
        "index_n_heads": 4,
        # The SGLang FP4 indexer kernels use a fixed 128-wide key/query layout.
        "index_head_dim": 128,
        "index_topk": 512,
        "candidate_source_layer_id": 3,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-06,
        # n-gram engram hash tables added into the residual at a few layers
        # The final KV source layer must remain Engram-free in SGLang; the
        # source model likewise carries Engram on an earlier KV source only.
        "engram_layer_ids": [1, 2],
        # row counts of the two hash tables; derived from engram_vocab_size via the prime layout
        # in the repo's engram.py (find_next_prime chain), computed for the tiny vocab below
        "engram_num_embeddings": [6748, 7592],
        "engram_max_ngram_size": 4,
        "engram_vocab_size": 512,
        "engram_n_heads": 4,
        "engram_head_dim": 64,
        "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 99092,
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [3, 4, 5],
        # Keep the source low-rank Markov head topology while trimming the two
        # vocabulary-sized BF16 matrices. SGLang only requires a positive rank.
        "dspark_markov_rank": 16,
        "dspark_n_routed_experts": 8,
        "dspark_num_experts_per_tok": 3
    },
    "vision_config": {
        "model_type": "deepseek_v41_vision",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "intermediate_size": 128,
        "patch_size": 14,
        "rope_theta": 10000,
        "downsample_ratio": 3,
        "max_image_tokens": 1024,
        "min_pixels": 295936,
        "max_wh_ratio": None
    }
}

def main():
    torch.manual_seed(42)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        source_model_id,
        revision=source_revision,
        trust_remote_code=True,
    )
    state_dict = generate(config, tokenizer)
    save_file(state_dict, Path(save_folder) / "model.safetensors")
    with open(Path(save_folder) / "model.safetensors", "rb") as f:
        state_dict = f.read()
        print("Hash: ", hashlib.sha256(state_dict).hexdigest())

    with open(Path(save_folder) / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    tokenizer.save_pretrained(save_folder)

    try:
        from transformers import GenerationConfig
        generation_config = GenerationConfig.from_pretrained(
            source_model_id,
            revision=source_revision,
            trust_remote_code=True,
        )
        generation_config.save_pretrained(save_folder)
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping generation config: {exc}")

BF16 = "torch.bfloat16"
F32 = "torch.float32"
FP8 = "torch.float8_e4m3fn"
SCALE = "torch.float8_e8m0fnu"
I8 = "torch.int8"

class TensorSpec(TypedDict):
    shape: list[int]
    dtype: str

Config = dict[str, Any]
State = dict[str, TensorSpec]
TensorDict = dict[str, torch.Tensor]

def initialize(state: State, config: Config, init_bound: float) -> TensorDict:
    tensors: TensorDict = {}
    scale_dtype = torch.float8_e8m0fnu
    fp4_boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

    def scale_for(
        amax: torch.Tensor, qmax: float, dtype: torch.dtype
    ) -> torch.Tensor:
        scale = amax.float().clamp_min(torch.finfo(torch.float32).tiny) / qmax
        if dtype == scale_dtype:
            scale = torch.pow(2.0, torch.round(torch.log2(scale)))
        return scale.to(dtype)

    def fp8_tensor(
        shape: list[int], scale_shape: list[int], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # block sizes are inferred from the scale shape, so this also handles the engram embedding
        # table, which is quantized per-row over 32-column blocks (block_out == 1)
        assert shape[0] % scale_shape[0] == 0 and shape[1] % scale_shape[1] == 0
        block_out = shape[0] // scale_shape[0]
        block_in = shape[1] // scale_shape[1]
        value = torch.empty(shape, dtype=torch.bfloat16).uniform_(
            -init_bound, init_bound
        )
        blocks = value.view(
            scale_shape[0], block_out, scale_shape[1], block_in
        ).transpose(1, 2)
        scales = scale_for(blocks.abs().amax(dim=(-1, -2)), 448, dtype)
        weight = (
            (blocks.float() / scales.float()[..., None, None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .transpose(1, 2)
            .reshape(shape)
            .contiguous()
        )
        return weight, scales

    def fp4_tensors(
        count: int,
        shape: list[int],
        scale_shape: list[int],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_dim, packed_in_dim = shape
        block_out, block_in = config["quantization_config"]["weight_block_size"]
        assert out_dim % block_out == 0
        assert packed_in_dim * 2 % block_in == 0
        assert packed_in_dim == scale_shape[1] * 16
        value = torch.empty(
            count, out_dim, packed_in_dim * 2, dtype=torch.bfloat16
        ).uniform_(-init_bound, init_bound)
        blocks = value.view(count, out_dim, scale_shape[1], 32)
        scales = scale_for(blocks.abs().amax(dim=-1), 6, dtype)
        normalized = (blocks.float() / scales.float()[..., None]).clamp(-6, 6)
        code = torch.bucketize(normalized.abs(), fp4_boundaries)
        code += normalized.signbit() * 8
        code = code.view(count, out_dim, packed_in_dim * 2)
        packed = (code[..., ::2] | (code[..., 1::2] << 4)).to(torch.uint8)
        weight = packed.view(torch.int8).contiguous()
        return weight, scales

    dtype_map: dict[str, torch.dtype] = {
        BF16: torch.bfloat16,
        F32: torch.float32,
        FP8: torch.float8_e4m3fn,
        SCALE: scale_dtype,
        I8: torch.int8,
    }
    fp4_groups: dict[
        tuple[tuple[int, ...], tuple[int, ...], torch.dtype], list[str]
    ] = {}
    for name, spec in state.items():
        if spec["dtype"] != I8:
            continue
        scale_spec = state[name.replace(".weight", ".scale")]
        key = (
            tuple(spec["shape"]),
            tuple(scale_spec["shape"]),
            dtype_map[scale_spec["dtype"]],
        )
        fp4_groups.setdefault(key, []).append(name)

    fp4_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    max_batch_elements = 4 * 1024 * 1024
    for (shape_tuple, scale_shape_tuple, dtype), names in fp4_groups.items():
        shape = list(shape_tuple)
        scale_shape = list(scale_shape_tuple)
        logical_elements = shape[0] * shape[1] * 2
        batch_size = max(1, max_batch_elements // logical_elements)
        for start in range(0, len(names), batch_size):
            batch_names = names[start: start + batch_size]
            weights, scales = fp4_tensors(
                len(batch_names), shape, scale_shape, dtype
            )
            for index, name in enumerate(batch_names):
                fp4_cache[name] = (
                    weights[index].clone(),
                    scales[index].clone(),
                )

    for name, spec in state.items():
        if name in tensors:
            continue
        shape, dtype = spec["shape"], dtype_map[spec["dtype"]]
        scale_name = name.replace(".weight", ".scale")
        if spec["dtype"] == FP8:
            scale_spec = state[scale_name]
            scale_type = dtype_map[scale_spec["dtype"]]
            tensors[name], tensors[scale_name] = fp8_tensor(
                shape, scale_spec["shape"], scale_type
            )
        elif spec["dtype"] == I8:
            tensors[name], tensors[scale_name] = fp4_cache.pop(name)
        elif not name.endswith(".scale"):
            tensors[name] = torch.empty(shape, dtype=dtype).uniform_(
                -init_bound, init_bound
            )
        print(f"{name}: {shape} {dtype}", flush=True)
    file_size_by_name = {name: tensor.numel() * tensor.element_size() for name, tensor in tensors.items()}
    total_file_size = sum(file_size_by_name.values())
    k = 20
    topk = sorted(file_size_by_name.items(), key=lambda x: x[1], reverse=True)[:k]
    print(f"File size: {total_file_size / 1024 / 1024:.2f} MB")
    print(f"Top {k} largest tensors:")
    for name, size in topk:
        print(f"  {name}: {size / 1024 / 1024:.2f} MB")
    return tensors

def generate(config: Config, tokenizer: Any, init_bound: float = 0.2) -> TensorDict:
    assert init_bound > 0
    text = config["text_config"]
    vision = config["vision_config"]
    quant = config["quantization_config"]
    rope = text["rope_scaling"]

    # Transformers has no deepseek_v41 registration or auto_map. The source repository instead
    # ships a complete readable runtime under inference/. Instantiate that implementation on the
    # meta device so the official modules, rather than a handwritten tensor list, define the tiny
    # checkpoint structure. Pin the inspected source revision for reproducibility.
    runtime_files = (
        "model.py",
        "engram.py",
        "vision.py",
        "kernel.py",
        "image_processor.py",
    )
    runtime_dir = None
    for filename in runtime_files:
        file_path = hf_hub_download(
            source_model_id,
            filename=f"inference/{filename}",
            revision=source_revision,
        )
        runtime_dir = str(Path(file_path).parent)
    assert runtime_dir is not None
    sys.path.insert(0, runtime_dir)
    try:
        from model import ModelArgs, Transformer
    finally:
        sys.path.remove(runtime_dir)

    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=min(text["max_position_embeddings"], 128),
        dtype="fp8" if quant["quant_method"] == "fp8" else "bf16",
        expert_dtype=quant.get("expert_dtype"),
        vocab_size=text["vocab_size"],
        dim=text["hidden_size"],
        moe_inter_dim=text["moe_intermediate_size"],
        n_layers=text["num_hidden_layers"],
        n_mtp_layers=text["num_nextn_predict_layers"],
        n_heads=text["num_attention_heads"],
        n_routed_experts=text["n_routed_experts"],
        n_shared_experts=text["n_shared_experts"],
        n_activated_experts=text["num_experts_per_tok"],
        score_func=text["scoring_func"],
        norm_topk_prob=text["norm_topk_prob"],
        route_scale=text["routed_scaling_factor"],
        swiglu_limit=text["swiglu_limit"],
        q_lora_rank=text["q_lora_rank"],
        head_dim=text["head_dim"],
        rope_head_dim=text["qk_rope_head_dim"],
        norm_eps=text["rms_norm_eps"],
        o_groups=text["o_groups"],
        o_lora_rank=text["o_lora_rank"],
        window_size=text["sliding_window"],
        compress_ratios=tuple(text["compress_ratios"]),
        kv_source_layers=tuple(text["kv_source_layer_ids"]),
        index_source_layers=tuple(text["index_source_layer_ids"]),
        compress_rope_theta=text["compress_rope_theta"],
        original_seq_len=rope["original_max_position_embeddings"],
        rope_theta=text["rope_theta"],
        rope_factor=rope["factor"],
        beta_fast=rope["beta_fast"],
        beta_slow=rope["beta_slow"],
        index_n_heads=text["index_n_heads"],
        index_head_dim=text["index_head_dim"],
        index_topk=text["index_topk"],
        candidate_source_layer=text["candidate_source_layer_id"],
        candidate_topk_blocks=text["candidate_topk_blocks"],
        candidate_block_size=text["candidate_block_size"],
        hc_mult=text["hc_mult"],
        hc_sinkhorn_iters=text["hc_sinkhorn_iters"],
        hc_eps=text["hc_eps"],
        engram_layer_ids=tuple(text["engram_layer_ids"]),
        engram_num_embeddings=tuple(text["engram_num_embeddings"]),
        engram_max_ngram_size=text["engram_max_ngram_size"],
        engram_vocab_size=text["engram_vocab_size"],
        engram_n_heads=text["engram_n_heads"],
        engram_head_dim=text["engram_head_dim"],
        engram_pad_id=text["engram_pad_token_id"],
        engram_compressed_vocab_size=text["engram_compressed_vocab_size"],
        vision_n_layers=vision["num_hidden_layers"],
        vision_dim=vision["hidden_size"],
        vision_n_heads=vision["num_attention_heads"],
        vision_inter_dim=vision["intermediate_size"],
        vision_patch_size=vision["patch_size"],
        vision_rope_theta=vision["rope_theta"],
        vision_downsample_ratio=vision["downsample_ratio"],
        vision_max_n_token=vision["max_image_tokens"],
        vision_min_pixels=vision["min_pixels"],
        vision_max_wh_ratio=vision["max_wh_ratio"],
        image_token_id=config["image_token_id"],
        dspark_block_size=text["dspark_block_size"],
        dspark_noise_token_id=text["dspark_noise_token_id"],
        dspark_target_layer_ids=tuple(text["dspark_target_layer_ids"]),
        dspark_markov_rank=text["dspark_markov_rank"],
        dspark_n_routed_experts=text["dspark_n_routed_experts"],
        dspark_n_activated_experts=text["dspark_num_experts_per_tok"],
    )

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            runtime_model = Transformer(args, tokenizer)
    finally:
        torch.set_default_dtype(previous_dtype)

    dtype_map = {
        torch.bfloat16: BF16,
        torch.float32: F32,
        torch.float8_e4m3fn: FP8,
        torch.float8_e8m0fnu: SCALE,
        torch.float4_e2m1fn_x2: I8,
    }
    state: State = {}
    block_out, block_in = quant["weight_block_size"]
    for name, tensor in runtime_model.state_dict().items():
        # Runtime modules tie these to the backbone. The public HF checkpoint stores only the
        # backbone copies; safetensors cannot duplicate shared storage either.
        if name.startswith("mtp.") and name.split(".", 2)[-1] in (
            "embed.weight",
            "head.weight",
        ):
            continue

        shape = list(tensor.shape)
        dtype = dtype_map[tensor.dtype]

        # inference/convert.py dequantizes wo_a to bf16 for its grouped einsum. Undo that runtime
        # representation here: the public HF checkpoint stores wo_a as block-quantized fp8.
        if name.endswith(".attn.wo_a.weight"):
            assert shape[0] % block_out == 0 and shape[1] % block_in == 0
            dtype = FP8
            state[name] = {"shape": shape, "dtype": dtype}
            state[name.replace(".weight", ".scale")] = {
                "shape": [shape[0] // block_out, shape[1] // block_in],
                "dtype": SCALE,
            }
            continue

        # The readable runtime promotes these parameters for computation. The published HF
        # checkpoint keeps them in bf16, as confirmed from all 48 source shard headers.
        if (
            name == "head.weight"
            or name.endswith(".markov_head.head.weight")
            or name.endswith(".confidence_head.proj.weight")
            or ".attn.compressor.w" in name
            or (
                name.startswith("vision.")
                and (
                    name == "vision.norm.weight"
                    or name.endswith(".norm1.weight")
                    or name.endswith(".norm2.weight")
                )
            )
        ):
            dtype = BF16

        state[name] = {"shape": shape, "dtype": dtype}

    del runtime_model
    return initialize(state, config, init_bound)

main()
```

</details>

### Test environment:

- Pillow: 12.2.0
- flashinfer-python: 0.7.2
- huggingface_hub: 1.33.0
- numpy: 2.3.5
- safetensors: 0.8.0
- sglang: 0.5.23.dev384+g9ab74b851
- sympy: 1.14.0
- tilelang: 0.1.12
- torch: 2.14.1
- transformers: 5.19.0

### Change log:

- 2026-10-20: Initial version with TP=2 and DSpark/MTP support.

This repo is generated by GPT-5.6-Sol.