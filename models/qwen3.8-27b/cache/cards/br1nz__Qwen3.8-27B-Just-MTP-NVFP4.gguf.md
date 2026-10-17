---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---
Just the MTP heads, quantized to NVFP4. 

I use this with llama.cpp to run the MTP heads on a Blackwell card while I run the main model on something else.

use -md to specify the model and -devd to specify the CUDA card.

e.g.

```
llama-server \
  -m "Qwen3.8-27B-UD-Q8_K_XL.gguf" \
  -ngl -1 --no-context-shift --reasoning-format deepseek \
  --chat-template-kwargs '{"preserve_thinking": "True"}' -rea on  \
  --kv-unified \
  -fa on \
  --load-mode mlock \
  --alias Qwen3.8 \
  --jinja \
  --reasoning-budget -1 \
  --reasoning-effort low \
  -b 4096 \
  -cram -1  \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --cache-reuse 256 --threads-http 6 --cache-prompt --reasoning-preserve \
  -ctk f16 -ctv f16 \
  -np 1 \
  -c 0 \
  -dev ROCm1,ROCm2,CUDA0 \
  -ngld -1 -ctkd f16 -ctvd f16 \
  --spec-type draft-mtp \
  --spec-draft-n-max 10 \
  --spec-draft-p-min 0.8 \
  -md Qwen3.8-27B-Just-MTP-NVFP4.gguf \
  -devd CUDA0 \
  -ts 12,12,2 \

```
