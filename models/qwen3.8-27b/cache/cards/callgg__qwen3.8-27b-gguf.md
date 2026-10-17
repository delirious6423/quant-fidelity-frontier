---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---
## qwen3.8-27b-gguf
- execute tools and/or agentic workflow
- fit in 12GB vram/ram or less

run it with ggk
```bash
ggk server engine -- -m qwen3.8-27b-q2_k_s.gguf --mmproj mmproj-qwen3.8-27b-q4_0.gguf --jinja --spec-type draft-mtp -md mtp-qwen3.8-27b-q4_0.gguf --spec-draft-n-max 2 --parallel 1 -ts 1,0
```

or run it with llama.cpp
```
./llama-server -m qwen3.8-27b-q2_k_s.gguf --mmproj mmproj-qwen3.8-27b-q4_0.gguf --jinja --spec-type draft-mtp -md mtp-qwen3.8-27b-q4_0.gguf --spec-draft-n-max 2 --parallel 1 --host 127.0.0.1 --port 8888
```

or opt lmstudio, etc.