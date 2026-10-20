---
library_name: node-llama-cpp
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
quantized_by: giladgd
tags:
- node-llama-cpp
- llama.cpp
- conversational
---

# gemma-4-26B-A4B-it-GGUF

> [!NOTE]
> Read [our blog post](https://node-llama-cpp.withcat.ai/blog/v3.19-gemma-4) to learn more about using Gemma 4 with `node-llama-cpp`

Static quants of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).


## Quants
| Link | [URI](https://node-llama-cpp.withcat.ai/cli/pull) | Quant | Size |
|:-----|:--------------------------------------------------|:------|-----:|
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q2_K.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q2_K` | Q2_K | 10.6GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q3_K_S.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q3_K_S` | Q3_K_S | 12.2GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q3_K_M.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q3_K_M` | Q3_K_M | 13.3GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q3_K_L.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q3_K_L` | Q3_K_L | 13.8GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q4_0.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q4_0` | Q4_0 | 14.4GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q4_K_S.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q4_K_S` | Q4_K_S | 15.5GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q4_K_M.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q4_K_M` | Q4_K_M | 16.8GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q5_0.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q5_0` | Q5_0 | 17.5GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q5_K_S.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q5_K_S` | Q5_K_S | 18.0GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q5_K_M.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q5_K_M` | Q5_K_M | 19.1GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q6_K.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q6_K` | Q6_K | 22.6GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.Q8_0.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q8_0` | Q8_0 | 26.9GB |
| [GGUF](https://huggingface.co/giladgd/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it.BF16.gguf) | `hf:giladgd/gemma-4-26B-A4B-it-GGUF:BF16` | BF16 | 50.5GB |

> [!TIP]
> Download a quant using `node-llama-cpp` ([more info](https://node-llama-cpp.withcat.ai/cli/pull)):
> ```bash
> npx -y node-llama-cpp pull <URI>
> ```


# Usage
## Use with [`node-llama-cpp`](https://node-llama-cpp.withcat.ai) (recommended)
Ensure you have node.js installed:
```bash
brew install nodejs
```

### CLI
Chat with the model:
```bash
npx -y node-llama-cpp chat hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q4_K_M
```

### Code
Use it in your project:
```bash
npm install node-llama-cpp
```

```typescript
import {getLlama, resolveModelFile, LlamaChatSession} from "node-llama-cpp";

const modelUri = "hf:giladgd/gemma-4-26B-A4B-it-GGUF:Q4_K_M";


const llama = await getLlama();
const model = await llama.loadModel({
    modelPath: await resolveModelFile(modelUri)
});
const context = await model.createContext();
const session = new LlamaChatSession({
    contextSequence: context.getSequence()
});


const q1 = "Hi there, how are you?";
console.log("User: " + q1);

const a1 = await session.prompt(q1);
console.log("AI: " + a1);
```

> [!TIP]
> Read the [getting started guide](https://node-llama-cpp.withcat.ai/guide/) to quickly scaffold a new `node-llama-cpp` project


## Use with [llama.cpp](https://github.com/ggml-org/llama.cpp)
Install llama.cpp through brew (works on Mac and Linux)

```bash
brew install llama.cpp
```

### CLI
```bash
llama-cli -hf giladgd/gemma-4-26B-A4B-it-GGUF:Q4_K_M -p "The meaning to life and the universe is"
```

### Server
```bash
llama-server -hf giladgd/gemma-4-26B-A4B-it-GGUF:Q4_K_M -c 2048
```
