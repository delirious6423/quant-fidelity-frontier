---
license: other
license_name: swift-open-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- gufo
- abliterated
- qwen3.8
- flash-next
---

# Swift 1.5 Flash-Next Abliterated for Gufo

UkisAI's Swift 1.5 Qwen3.8 Flash-Next with the refusal direction projected out of its 149 residual-writer tensors, in the layout Gufo runs on AMD Strix Halo (`gfx1151`): Q8_0 dense projections, Q4_1 routed-expert down projections, a Q4 multi-token-prediction head and the BF16 vision projector. NOTICE lists every source revision and change.

| File | Size | Role |
|---|---|---|
| `Swift-1.5-Flash-Next-Abliterated-Q8dense-Q41down.gguf` | 106.8 GB | model |
| `Swift-1.5-Flash-Next-Abliterated-MTP-Q4exps.gguf` | 1.6 GB | MTP draft head |
| `mmproj-Swift-1.5-Flash-Next-BF16.gguf` | 0.9 GB | vision projector |

The Q4_1 routed-expert down projections need a Gufo build that loads them (gufo-org/gufo issue 906).

```sh
gufo serve --host 127.0.0.1 --port 8080 llm \
  --model Swift-1.5-Flash-Next-Abliterated-Q8dense-Q41down.gguf \
  --mmproj mmproj-Swift-1.5-Flash-Next-BF16.gguf \
  --speculative mtp --mtp-model Swift-1.5-Flash-Next-Abliterated-MTP-Q4exps.gguf \
  --sessions 1 --enable-yarn --context 1048576 \
  --reasoning-effort xhigh --preserve-thinking on
```
