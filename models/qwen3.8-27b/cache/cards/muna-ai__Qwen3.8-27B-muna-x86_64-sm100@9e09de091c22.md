---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: cc-by-nc-nd-4.0
library_name: muna
pipeline_tag: image-text-to-text
tags:
- muna
- compiled
- nvfp4
- fp4
- 4-bit
- blackwell
- sm100
- b200
- nvidia
- cuda
- linux
- x86_64
- qwen3
- conversational
- vision
- reasoning
- tool-calling
---

# Qwen3.8-27B-muna-x86_64-sm100

Run Qwen 3.8 27B locally on an NVIDIA B200, with no API key: 320 tok/s single-user decode, cold starts in 11 seconds from local disk.

**Runs on**: NVIDIA B200 · Linux x86_64 · ~28 GB disk

An NVFP4 build with DFlash2 speculative decoding, compiled from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with [Muna](https://muna.ai), using the [RadixArk NVFP4 checkpoint](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) and the [DFlash2 drafter by Inco AI](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). See the [source code](https://github.com/muna-ai/models/blob/main/large-language-models/qwen_3_8_27b.py) used to compile it.

- **Throughput**: 320 tok/s median single-user decode at default sampling, up to 603 tok/s with greedy decoding.
- **Cold start**: 10.8 s median from load to ready, with weights on local disk.
- **Co-location**: stayed loaded alongside Gemma 4 26B A4B and two embedding models on one B200, for 150 hours of production serving in the past two weeks.

## Usage

Both SDKs include OpenAI- and Anthropic-compatible clients.

### Python

```bash
pip install "muna>=0.1.34"
```

```python
from muna import Muna

openai = Muna().beta.openai
stream = openai.chat.completions.create(
    model="hf:muna-ai/Qwen3.8-27B-muna-x86_64-sm100",
    messages=[{ "role": "user", "content": "What is the capital of France?" }],
    acceleration="local_gpu",
    stream=True
)
for chunk in stream:
    delta = chunk.choices[0].delta
    print(delta.reasoning_content or delta.content or "", end="", flush=True)
```

### Rust

```toml
[dependencies]
muna = "0.0.32"
futures = "0.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use futures::StreamExt;
use muna::beta::anthropic::{
    ContentBlockDelta, MessageContent, MessageCreateParams, MessageParam, RawMessageStreamEvent
};
use muna::{Acceleration, Muna, MunaError};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), MunaError> {
    let muna = Muna::default();
    let mut stream = muna.beta.anthropic.messages.stream(MessageCreateParams {
        model: "hf:muna-ai/Qwen3.8-27B-muna-x86_64-sm100".to_string(),
        max_tokens: 1024,
        messages: vec![MessageParam {
            role: "user".to_string(),
            content: MessageContent::Text("What is the capital of France?".to_string()),
        }],
        acceleration: Some(Acceleration::LocalGpu),
        ..Default::default()
    }).await?;
    while let Some(event) = stream.next().await {
        if let RawMessageStreamEvent::ContentBlockDelta { delta, .. } = event? {
            match delta {
                ContentBlockDelta::ThinkingDelta { thinking } => print!("{thinking}"),
                ContentBlockDelta::TextDelta { text } => print!("{text}"),
                _ => continue,
            }
            std::io::stdout().flush().ok();
        }
    }
    Ok(())
}
```

The first run downloads the model to the Hugging Face cache, which Python and Rust share. No Muna access key is needed.

## Performance

### Throughput

Single-user decode speed, one request at a time:

| Sampling | Median | Range |
|---|---|---|
| Default (temperature 1.0, top-p 0.95, top-k 20) | 320 tok/s | 210 to 467 tok/s |
| Greedy (temperature 0) | 424 tok/s | 289 to 603 tok/s |

NVIDIA B200 on Modal, October 21, 2026, `muna==0.1.34`. Three prompts (code, a long essay, a technical explanation), two runs each, up to 2,048 output tokens, `reasoning_effort="none"`, after one warmup request. Decode speed is measured from the first output token to the last. Speed depends on how many drafted tokens are accepted, so code runs fastest and open-ended prose slowest.

### Cold start

| Compiled predictor | CUDA libraries | Weights | Load to ready |
|---|---|---|---|
| 56 MB | 1.96 GB | 25.8 GB | 10.8 s (median), 14.0 s (p90) |

The compiled predictor, the inference engine and model code in one native library, is only 56 MB. The rest of the code is standard CUDA libraries.

Muna hosted deployment of the same predictor, 18 loads on single-B200 nodes, October 9 to 21, 2026. Weights were already on local disk and no other model was loading, so download time is excluded. Measured from creating the predictor in a running server to the engine being ready, including CUDA graph capture; process launch is excluded, and the first token then takes about 35 ms.

### Co-location

Over the past two weeks, Muna's hosted fleet kept this model loaded alongside other models on single-B200 machines. Each row is the time Qwen 3.8 27B spent loaded with a given set of co-tenants on the same GPU, and its latency on production traffic during that time.

| Sharing one B200 with | Hours | Median time to first token | Median interval between streamed outputs |
|---|---|---|---|
| Nothing (alone) | 107.5 | 353 ms | 9.5 ms |
| [Qwen3 Embedding 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 45.2 | 265 ms | 10.4 ms |
| [Gemma 4 26B A4B](https://huggingface.co/google/gemma-4-26B-A4B-it), Qwen3 Embedding 8B, [Nomic Embed Text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 40.6 | 351 ms | 8.0 ms |
| Gemma 4 26B A4B | 33.2 | 607 ms | 9.6 ms |
| Gemma 4 26B A4B, Qwen3 Embedding 8B | 23.1 | 140 ms | 7.5 ms |
| Gemma 4 26B A4B, Nomic Embed Text v1.5 | 7.9 | 495 ms | 7.3 ms |

All models stay loaded and serve from a single GPU, each with its own engine.

## Let us host it for you

Don't want to source GPUs or keep them busy? We can serve this model for you as an endpoint on our shared GPU pool.

In our fleet, this build runs behind KV-aware routing: the compiler also produces a sidecar that tokenizes each request exactly as the engine does, so requests go to the replica that already holds their prefix in cache. Over the past two weeks, 86% of prompt tokens were served from cache.

- **Hosted API**: call Qwen 3.8 27B per token on the [Muna API](https://muna.ai), with nothing to provision.
- **Your fine-tune, hosted**: running a fine-tune of Qwen 3.8 27B? We can host it the same way, kept warm, for a monthly fee plus per-token pricing. No GPUs to source, and no SGLang or vLLM to operate. [Talk to us](mailto:hi@muna.ai).
- **Your own fleet**: running this build as a multi-replica service on your own GPUs? The KV-aware routing sidecar is a Muna predictor too, available through the Muna API with per-deployment pricing. [Talk to us](mailto:hi@muna.ai).

Evaluating this build at your company? We're happy to help: hi@muna.ai.

If this build is useful, a star on [GitHub](https://github.com/muna-ai/muna-py) helps others find it.

## Limitations

- This build runs only on NVIDIA B200 GPUs with x86_64 hosts. For other hardware, the [Muna API](https://muna.ai) serves the right build for your hardware automatically.
- Images are supported as base64 `data:` URLs in `image_url` parts; remote image URLs are rejected.
- `tool_choice`, `response_format`, `frequency_penalty`, and `presence_penalty` are ignored. Tool calls behave as `tool_choice: "auto"`.

## License

The compiled Muna binary in this repository is released under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) for research and evaluation. For commercial licensing, contact hi@muna.ai.

The model weights are licensed separately under the [Apache License 2.0](LICENSE-MODEL.md). To tell which files are which, consult `muna.json`: resources with type `dso` are compiled Muna binaries, and all other resources are model weights.

The weights are redistributed unmodified from [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), an NVFP4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Copyright 2026 Alibaba Cloud), and from [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), a mirror of [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) by Inco AI. The DFlash2 drafter weights are licensed under the [Apache License 2.0](LICENSE-DRAFTER.md).
