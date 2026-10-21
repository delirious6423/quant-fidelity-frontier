---
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: openvino
tags:
  - openvino
  - nncf
  - awq
  - int4
  - weight-only-quantization
  - vision-language
  - moe
  - gemma4
license: apache-2.0
---

# Gemma 4 26B-A4B-it — OpenVINO INT4 (AWQ)

An [OpenVINO](https://docs.openvino.ai/) **INT4 weight-only (AWQ)** conversion of
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) — a vision-language
mixture-of-experts model (~25 B total parameters, ~3.8 B active per token). Produced with
[optimum-intel](https://github.com/huggingface/optimum-intel) + [NNCF](https://github.com/openvinotoolkit/nncf)
for efficient CPU (and Intel GPU) inference via
[`openvino-genai`](https://github.com/openvinotoolkit/openvino.genai).

Released under the **Apache License 2.0**, inherited from the base model
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (tagged `apache-2.0`).
This is a derivative quantization only; all model rights and terms follow the base model.

## Files

Multi-component OpenVINO IR (~14.3 GB total), as emitted by the exporter:

| Component | Precision | Notes |
|---|---|---|
| `openvino_language_model.{xml,bin}` | **INT4 (AWQ)** | the decoder (~13.0 GB) — the bulk |
| `openvino_text_embeddings_model.{xml,bin}` | INT8 | ~0.69 GB |
| `openvino_vision_embeddings_model.{xml,bin}` | INT8 | vision tower (~0.53 GB) |
| `openvino_text_embeddings_per_layer_model.{xml,bin}` | — | Gemma per-layer embedding scaling |
| `openvino_tokenizer.{xml,bin}`, `openvino_detokenizer.{xml,bin}` | — | OpenVINO tokenizer IR |
| `config.json`, `openvino_config.json`, `generation_config.json`, `chat_template.jinja`, `tokenizer*`, `preprocessor_config.json` | — | |

## How the quantization was produced

**Data-aware AWQ INT4 weight compression** (NNCF, via optimum-intel). Exact per-module scheme
(from `openvino_config.json`, `dtype: "int4"`):

- **Language model — INT4 AWQ**, `bits = 4`, `group_size = 64`, `ratio = 1.0`, `sym = false`,
  `quant_method = "awq"`.
- **MoE routers are kept at higher precision** — `ignored_scope.patterns: [".*router.*"]`, so the
  expert-gating layers are not quantized to INT4 (preserves routing accuracy).
- **Embedding / vision modules — INT8**, weight-only.

### Reproduce

> **The export requires a PRE-RELEASE toolchain.** As of this writing the `gemma4` OpenVINO exporter
> is **not in any released `optimum-intel`** — it lives only on git `main`, and pairs with a specific
> transformers version + the **nightly** OpenVINO wheels. (Released `optimum-intel` returns
> *"gemma4 is not supported yet"*; transformers < 5.5 raises `KeyError: 'gemma4'`.) The combination below
> is the one from the official OpenVINO `gemma4` notebook and is what produced this IR.

```bash
# Toolchain (pre-release — install into an isolated venv)
pip install "git+https://github.com/huggingface/optimum-intel.git" \
    --extra-index-url https://download.pytorch.org/whl/cpu
pip install -U --pre openvino-genai openvino openvino-tokenizers nncf \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
pip install "transformers==5.5.0"

optimum-cli export openvino \
  --model google/gemma-4-26B-A4B-it \
  --task image-text-to-text \
  --weight-format int4 \
  ./gemma-4-26B-A4B-int4-ov
```

Versions used for this conversion:

| Package | Version |
|---|---|
| OpenVINO | 2026.3.0.dev (nightly) |
| openvino-genai / openvino-tokenizers | 2026.3.0.dev (nightly) |
| NNCF | 3.2.0 |
| optimum / optimum-intel | 2.2.0 / git `main` |
| transformers | **5.5.0** |
| Python | 3.14 |

### Gotchas worth knowing

- **`--task image-text-to-text` is required** — Gemma 4 is a VLM (`model_type: gemma4`,
  `Gemma4ForConditionalGeneration`). The plain text task is not registered for this architecture.
- **The exporter is pre-release; the runtime need not be.** This IR was *exported* with nightly
  OpenVINO, but it **loads and serves fine under released `openvino-genai` 2026.2** — you do not need
  the nightly runtime to run it.
- **Conversion is RAM-heavy.** The exporter loads the full bf16 weights and upcasts during tracing
  (peak host memory on the order of ~200 GB for this model); configure a large pagefile/swap. Running
  the resulting INT4 IR needs far less.

## Usage

```python
import openvino_genai as ov_genai

pipe = ov_genai.VLMPipeline("gemma-4-26B-A4B-int4-ov", "CPU")   # or "GPU" / "AUTO"
cfg = ov_genai.GenerationConfig()
cfg.max_new_tokens = 512
print(pipe.generate("Explain how a rocket engine works.", generation_config=cfg))
```

- **Validated on CPU** at **~12.9 tok/s** decode on a 24-core desktop CPU (Intel Core Ultra 9 285K) —
  fast for a ~25 B model because only ~3.8 B parameters are active per token (A4B MoE). The INT4 weights
  stay compressed in memory (~16 GB resident for this engine).
- **GPU note:** Intel GPU inference depends on the OpenVINO build's compressed-weight support for this
  architecture's MoE/vision operators. Verify the resident footprint on your hardware before relying on it.

### Note on stopping / chat template

This is **not** a `<think>`-style reasoning model — it answers directly. When serving via a chat API,
supply Gemma 4's stop tokens / chat template explicitly; a bare `VLMPipeline.generate` may over-run the
intended end of turn without the model's EOS handling wired up.

## Acknowledgements

Base model © Google — [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).
Quantization tooling: OpenVINO, NNCF, and optimum-intel.
