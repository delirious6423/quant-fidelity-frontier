---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---
```bash
llama-quantize \
  --output-tensor-type Q6_K \
  --token-embedding-type Q8_0 \
  --tensor-type ffn_down=Q6_K \
  --tensor-type ssm_out=Q6_K \
  --imatrix /models/bartowski/Qwen3.8-27B-GGUF/Qwen3.8-27B-imatrix.gguf \
  /models/my/Qwen3.8-27B-BF16.gguf \
  /models/my/Qwen3.8-27B-Q4_K_M.gguf \
  Q4_K
```