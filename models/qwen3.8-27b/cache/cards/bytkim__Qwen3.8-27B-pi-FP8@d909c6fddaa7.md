---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mtp
- multi-token-prediction
- speculative-decoding
- vision
- image
- multimodal
- image-text-to-text
- text-generation-inference
- qwen
- qwen3.8
- qwen38
- 27b
- pi
- coding
- coder
- code-generation
- agent
- tool-use
- tool-calling
- function-calling
- reasoning
- conversational
- sft
- grpo
- reinforcement-learning
- fp8
- quantized
license: apache-2.0
library_name: transformers
---

<img src="assets/benchmark-hero-gpqa-v2.png" alt="Qwen3.8-27B-pi and Base: Terminal-Bench 2.1 xhigh preview and Pi GPQA Diamond xhigh attempt pass rate; SciCode Pi/S8 subproblem scores" width="100%" />

<p align="center">
  <a href="https://huggingface.co/bytkim/Qwen3.8-27B-pi">BF16</a> ·
  <strong>FP8</strong> ·
  <a href="https://huggingface.co/bytkim/Qwen3.8-27B-pi-GGUF">GGUF</a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#performance-by-reasoning-level">Benchmarks</a> ·
  <a href="#advanced">Advanced</a> ·
  <a href="#license">License</a>
</p>


## Built for pi

**Qwen3.8-27B-pi** builds on [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for coding work in the Pi agent harness. It is tailored to the loop of reading a repository, editing files, running tools, and responding to their feedback. The aim is more completed work with less generated text, while retaining the base model's familiar interface.

Fine-tuned for the Pi agent harness, the model is designed to work through coding tasks as an iterative process: inspect the code, make a change, check the result, and use tool feedback to guide the next step. Adjustable reasoning effort lets you balance responsiveness with deeper problem-solving, from focused edits to more involved debugging and implementation work.

<img src="assets/benchmark-by-reasoning-gpqa-v1.png" alt="Qwen3.8-27B-pi versus Base across low, medium, and xhigh reasoning: Terminal-Bench 2.1 score, turns, tool calls, reasoning tokens, and total output tokens; Pi GPQA Diamond overall attempt pass rates; SciCode Pi/S8 subproblem scores" width="100%" />

## Qwen3.8-27B-pi Highlights

- **Built for Pi:** Fine-tuned for the everyday coding loop—reading repositories, editing files, running tools, and working through feedback in the Pi harness—with an emphasis on turning plans into working, checked implementations.
- **Curated coding experience:** Supervised fine-tuning on filtered, successful Pi sessions helps the model learn complete coding workflows, rather than isolated answers or code snippets, including how to adapt to existing environments and check results against task requirements.
- **Task quality matters:** Development also included reviewing, repairing, and exploring more demanding coding tasks—with a focus on clear requirements and checks that distinguish working solutions from incorrect ones.
- **Refined through reinforcement learning:** A second training stage pairs verified task outcomes with a custom reasoning-efficiency reward built on GRPO, encouraging successful low- and medium-effort solutions to reason more economically while leaving xhigh focused on correctness.
- **Available in practical formats:** Choose a deployment format that fits your hardware, including GGUF quantizations calibrated using complete Pi coding sessions.

Pi’s development followed a staged process, from trajectory curation and supervised fine-tuning to reinforcement learning and checkpoint evaluation. Each stage was assessed against the same practical goal: helping the model complete useful work in Pi while managing the resources it spends. Checkpoints were compared using actual agent outcomes alongside generated tokens, tool calls, and completion time—not training loss alone.

## Performance by reasoning level

<img src="assets/terminal-bench-score-tokens.png" alt="Terminal-Bench 2.1 task success versus mean output tokens for Pi and Base at low, medium, and xhigh reasoning" width="100%" />

> Curated Pi sessions and reinforcement learning emphasized completed, checked coding work. Pi shows a steadier rise in completion from low to xhigh than Base, with fewer output tokens at every matched effort level. In these selected results, Pi’s medium setting matches Base’s xhigh completion rate with about 41% fewer output tokens.

<img src="assets/gpqa-score-tokens.png" alt="GPQA Diamond attempt pass rate versus mean output tokens for Pi FP8 and Base at low, medium, and xhigh reasoning; y-axis 75–90 percent" width="100%" />

> Pi’s RL stage encouraged economical low/medium reasoning while keeping xhigh focused on correctness. The graph shows a smoother rise in Pi’s attempt success as effort increases, while Base peaks at medium. Pi achieves the highest xhigh score, but Base retains the medium-effort edge.

<img src="assets/scicode-score-tokens.png" alt="SciCode passed subproblems out of 337 versus mean output tokens for Pi FP8 and Base at low, medium, and xhigh reasoning; y-axis 35–50 percent" width="100%" />

> Pi’s development prioritized successful solutions alongside resource use—not shorter responses alone. Both models improve with higher effort, but Pi solves more subproblems at every level. At xhigh, Pi scores higher with about 23% fewer output tokens; at medium, its higher score requires more tokens.

## Quickstart

```bash
vllm serve bytkim/Qwen3.8-27B-pi-FP8 \
  --host 127.0.0.1 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

> By default, vLLM loads this repository’s `generation_config.json`, which sets thinking-mode sampling to `temperature=1.0`, `top_p=0.95`, and `top_k=20`. No additional sampling flags are needed; client-provided values override these defaults, and `--generation-config vllm` uses vLLM’s built-in defaults instead.

### Non-thinking mode

To serve with non-thinking mode and its recommended sampling settings as defaults, use this command instead of the one above:

```bash
vllm serve bytkim/Qwen3.8-27B-pi-FP8 \
  --host 127.0.0.1 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --override-generation-config '{"temperature":0.7,"top_p":0.8,"presence_penalty":1.5}'
```

Alternatively, keep the thinking-mode server above and override individual API requests:

```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "presence_penalty": 1.5,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

> These overrides follow Qwen’s recommended non-thinking settings. The remaining sampling settings are inherited from the model and vLLM defaults; turning thinking off alone does not change sampling.

## Advanced

SGLang serving is also supported, including DFlash2 speculative decoding and FP8 KV cache.

> DFlash2 companion weights are mirrored unchanged from [Inco AI](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), with attribution and license included in [`dflash2/`](dflash2/).

The examples below use DFlash2 instead of the Quickstart's MTP configuration, with FP8 KV cache and a 256K context limit. Use a runtime build with DFlash2 support and sufficient GPU memory; these are documentation-based examples.

### Download the DFlash2 companion

Download the mirrored [DFlash2 companion](dflash2/) once before starting either server. The ~3.85 GB BF16 draft works as the companion for either target format.

```bash
hf download bytkim/Qwen3.8-27B-pi-FP8 \
  --include "dflash2/*" \
  --local-dir models/pi-fp8
```

### SGLang: DFlash2 + FP8 KV cache

```bash
python -m sglang.launch_server \
  --model-path bytkim/Qwen3.8-27B-pi-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --context-length 262144 \
  --max-running-requests 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path models/pi-fp8/dflash2 \
  --speculative-num-draft-tokens 8
```

### vLLM: DFlash2 + FP8 KV cache

```bash
vllm serve bytkim/Qwen3.8-27B-pi-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"dflash","model":"models/pi-fp8/dflash2","num_speculative_tokens":7}'
```

Run one server at a time. SGLang's eight-token verification block corresponds to seven draft tokens in the DFlash2 example; vLLM specifies the seven draft tokens directly. FP8 KV cache is independent of weight precision. For default-precision KV cache, change `--kv-cache-dtype fp8_e4m3` to `--kv-cache-dtype auto`.

References: [DFlash2 model and launch examples](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2#quick-start), [SGLang speculative decoding](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/speculative_decoding.mdx#dflash-decoding), [SGLang server arguments](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/server_arguments.mdx), and [vLLM FP8 KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/).

## License

Qwen3.8-27B-pi is fine-tuned from [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), whose base-model weights are licensed under the Apache License 2.0. See [LICENSE](LICENSE) for the full terms and [NOTICE](NOTICE) for upstream attribution and Pi modification details. Mirrored DFlash2 companions retain their license and provenance in [`dflash2/`](dflash2/).

## Acknowledgements

Built on Qwen3.8-27B from the Qwen team and adapted for the Pi agent harness. Thanks to the open-source training, inference, quantization, and evaluation projects—and dataset contributors—that supported its development.
