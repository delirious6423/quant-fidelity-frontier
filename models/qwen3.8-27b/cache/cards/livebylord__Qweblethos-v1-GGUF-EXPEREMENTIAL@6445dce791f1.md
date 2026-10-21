---
license: other
license_name: mixed-upstream-terms
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
language:
  - en
tags:
  - qwen3.8
  - qwen
  - gguf
  - llama.cpp
  - coding
  - agentic
  - tool-use
  - fine-tuned
  - distillation
---

# Qweblethos v1 — Qwen3.8-27B Fable Agentic Fine-Tune (Q4_K_M GGUF) **TEST VERSION**!

**Qweblethos v1** is a `Q4_K_M` GGUF of `Qwen/Qwen3.8-27B`, fine-tuned for coding-agent behavior, multi-step tool use, debugging, and verification-oriented workflows. 

The name combines **Qwen + Fable + Mythos**. That is the project identity, not a provenance claim: **this v1 checkpoint was trained on Fable 5 traces only; no Mythos-generated samples were used.** A future version may add a separately audited Mythos component.

## What this model is

- Base model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Fine-tuning method: LoRA, rank 32, alpha 32
- Trainable parameters: `159,383,552` (about `0.58%` of `27,516,112,112` parameters)
- Training context length: `16,384` tokens
- Training examples: `6,471`
- Evaluation examples used during the run: `128`
- Training: one epoch, `809` optimizer steps, one NVIDIA B200, about 2 hours
- Export: LoRA merged into the BF16 base, then quantized to GGUF v3 `Q4_K_M`
- Runtime target: llama.cpp-compatible applications
- Vision and MTP: not included in this text-only export
- **Experemental finetune. The new v2 is coming!**

This is response/trajectory distillation through supervised fine-tuning on synthetic agent traces. It is not logit-level distillation and it is not a reproduction of the teacher model.

## Training data

The final training mixture contains:

| Source | Selected training examples | Purpose |
| --- | ---: | --- |
| [`lordx64/agentic-distill-fable-5-sft`](https://huggingface.co/datasets/lordx64/agentic-distill-fable-5-sft), derived from [`Glint-Research/Fable-5-traces`](https://huggingface.co/datasets/Glint-Research/Fable-5-traces) | 4,392 | Fable 5 coding-agent reasoning and tool-use traces |
| [`greghavens/fable-5-coding-and-debugging-traces`](https://huggingface.co/datasets/greghavens/fable-5-coding-and-debugging-traces) | 2,079 | Verified full coding/debugging trajectories |

The preparation pipeline removed duplicates, filtered invalid or overlong samples, excluded service/meta noise, converted tool calls to the native Qwen3.8 format, supervised assistant tokens only, and kept train/evaluation task clusters disjoint.

## Evaluation

The same 128 held-out examples were evaluated with the base model and with the trained adapter enabled:

| Variant | Loss | Perplexity |
| --- | ---: | ---: |
| Base Qwen3.8-27B | 0.7328 | 2.0808 |
| Base + Qweblethos LoRA | 0.4634 | 1.5895 |

This is a `36.76%` relative loss reduction on the prepared held-out distribution. It shows that the adapter learned that distribution. **It does not prove an improvement on independent coding benchmarks such as SWE-bench, LiveCodeBench, or Terminal-Bench.** No independent benchmark score is claimed for v1.

## Downloaded file

| File | Format | Size | SHA-256 |
| --- | --- | ---: | --- |
| `Qweblethos-v1-Q4_K_M.gguf` | GGUF v3, Q4_K_M | 16,547,399,872 bytes (15.41 GiB) | `e0867b0c1b04ceb5d098b4892c89a19bd9049a171ce7e72500a0b967ba05932a` |

The export was produced with llama.cpp commit `0d9ceae1e38291035605613ab41a8f5e693d6fcd` and passed a CPU smoke test by generating the exact requested text `hello world`.

## Running with llama.cpp

```bash
llama-cli \
  -m Qweblethos-v1-Q4_K_M.gguf \
  -c 16384 \
  --jinja \
  -p "Fix this bug, explain the cause, and add focused tests."
```

OpenAI-compatible local server:

```bash
llama-server \
  -m Qweblethos-v1-Q4_K_M.gguf \
  -c 16384 \
  --jinja \
  --host 127.0.0.1 \
  --port 8080
```

Adjust GPU offload (`-ngl`) to match your hardware. The model can run with CPU/RAM offload, but a 27B model remains heavy: 24 GB of system RAM is marginal, while 32 GB or more is the practical target. Small GPUs can offload only part of the model.

## Intended use

Qweblethos v1 is intended for local experimentation with:

- coding-agent workflows;
- debugging and test-driven fixes;
- multi-step tool use;
- repository inspection and verification-oriented responses;
- llama.cpp and OpenAI-compatible local serving.

## Limitations

- The training distribution is narrow and heavily focused on coding-agent traces.
- General chat, multilingual quality, vision, and unrelated reasoning domains may regress relative to the base model.
- Tool-call formatting still depends on the runtime and chat-template implementation.
- The model may hallucinate commands, files, test results, or tool outputs. Treat generated actions as untrusted and review them before execution.
- This release has a smoke test and held-out loss comparison, not a comprehensive independent benchmark suite.
- Quantization can reduce quality compared with the merged BF16 checkpoint.

## License and attribution

This release combines components with different upstream terms. The Qwen3.8-27B base is Apache-2.0; the selected Fable datasets include AGPL-3.0 and CC BY 4.0 material. See [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md) before redistribution or commercial use.

## Reproducibility summary

- Base revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Dataset preparation revisions: `lordx64 9df06dd`, `Crownelius/Glint b9f20b5`
- LoRA: rank 32, alpha 32, dropout 0
- Learning rate: `8e-5`
- Effective batch size: `8`
- Max sequence length: `16,384`
- Final train loss: `0.51743`
- Final evaluation loss: `0.46328`

---

**Qweblethos is an independent community fine-tune and is not affiliated with Qwen, Alibaba, Anthropic, Fable, Mythos, Glint Research, or the dataset authors.**
