---
base_model:
- Qwen/Qwen3.8-27B
---
## Model Details

This example demonstrates how to use AutoScheme in AutoRound to automatically generate a mixed-bit recipe.

This is a 2.8-bits-per-weight (averaged over quantized layers) mixed-bit weight-only quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), generated with [AutoRound](https://github.com/intel/auto-round). 
The bits overhead of scale and zero points are included.

Please note that the 2-bit and 3-bit kernels are currently not as efficient as the 4-bit and 8-bit kernels.

## vLLM Inference

This model requires [vLLM PR #52890](https://github.com/vllm-project/vllm/pull/52890).

Until the PR is available in your installed vLLM release, install vLLM from the PR branch:

```bash
pip install -v "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/52729/head"

```

Serve the model:

```bash
vllm serve Intel/Qwen3.8-27B-bpw2.8-AutoRound 
```

Example request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Intel/Qwen3.8-27B-bpw2.8-AutoRound ",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize mixed-precision MoE quantization in one sentence."}
    ],
    "temperature": 0,
    "max_tokens": 256
  }'
```

"We need answer user: \"Summarize mixed-precision MoE quantization in one sentence.\" Need concise one sentence. Need think. Mixed-precision MoE quantization likely refers to quantizing mixture-of-experts models with different precision for different parts, e.g., weights, activations, experts, or different precision per layer/expert to balance accuracy and efficiency. Need one sentence. Could say: \"Mixed-precision MoE quantization assigns different bit precisions to different parts of a mixture-of-experts model (e.g., weights, activations, or individual experts) to reduce memory and compute while preserving accuracy.\" That's one sentence. Need final only. Ensure no extra.\n</think>\n\nMixed-precision MoE quantization assigns different bit precisions to different parts of a mixture-of-experts model—such as weights, activations, or individual experts—to reduce memory and compute while preserving 


## Accuracy

| Eval Backend | models                   |  MMLU  | GSM8K  | MMLU-Pro |
| :----------- | :----------------------- | :----: | :----: | :------: |
| lmeval hf    | BF16                     | 0.8349 | 0.7043 |  0.6278  |
| lmeval hf    | SignRoundv1(67565e1)     | 0.8158 | 0.3829 |  0.5656  |
|              | SignRoundv2(main branch) | 0.8148 | 0.6603 |  0.5738  |



| Eval Backend   | models                   |  MMLU  | GSM8K  | MMLU-Pro |
| :------------- | :----------------------- | :----: | :----: | :------: |
| evalscope vllm | BF16                     | 0.8749 | 0.9757 |  0.8042  |
| evalscope vllm | SignRoundv1(67565e1)     | 0.7963 | 0.9174 |          |
| evalscope vllm | SignRoundv2(main branch) | 0.8365 | 0.9454 | 0.7121          |

~~~bash
evalscope eval --model qwen38 --api-url http://127.0.0.1:8000/v1   --api-key EMPTY   --datasets gsm8k  --eval-batch-size 16     --generation-config '{"max_tokens":8192}'
~~~



## Generate the Quantized Model

AutoRound>0.14.2 is required

You could remove AR_AUTO_SCHEME_NSAMPLES=128 AR_AUTO_SCHEME_SEQLEN=512 if it's too slow. The default values are 16/256


v2 version
```bash
AR_AUTO_SCHEME_NSAMPLES=128 AR_AUTO_SCHEME_SEQLEN=512 python3 -m auto_round --model Qwen/Qwen3.8-27B --enable_alg_ext --options "W2A16G64,W3A16" --iters 1000 --nsamples 512 --lr 2e-3 --shared_layers "q_proj,k_proj,v_proj" "gate_proj,up_proj" "in_proj_qkv,in_proj_z" --format auto_round --avg_bits 2.8 --output_dir ./quantized --tasks mmlu,gsm8k,leaderboard_mmlu_pro --enable_torch_compile
```

v1 version
```bash
AR_AUTO_SCHEME_NSAMPLES=128 AR_AUTO_SCHEME_SEQLEN=512 python3 -m auto_round --model Qwen/Qwen3.8-27B --options "W2A16G64,W3A16" --iters 1000 --nsamples 512 --lr 2e-3 --shared_layers "q_proj,k_proj,v_proj" "gate_proj,up_proj" "in_proj_qkv,in_proj_z" --format auto_round --avg_bits 2.8 --output_dir ./quantized --tasks mmlu,gsm8k,leaderboard_mmlu_pro --enable_torch_compile
```



## Limitations

Quantization can change model behavior and may reduce accuracy on some tasks. This checkpoint is optimized for experiments with mixed INT2/INT4 MoE inference and requires a vLLM build with compressed-tensors WNA16 MoE Humming support. Users should run task-specific quality and safety evaluations before deployment.

## Ethical Considerations

The model may produce factually incorrect, biased, unsafe, or otherwise undesirable outputs. Downstream users are responsible for validating the model for their use case and for following the base model license and applicable policies.

## Citation

```bibtex
@article{cheng2025signroundv2,
  title={SignRoundV2: Toward Closing the Performance Gap in Extremely Low-Bit Post-Training Quantization for LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Guo, Heng and Shen, Haihao and Ma, Zaner},
  journal={arXiv preprint arXiv:2512.04746},
  year={2025}
}
```