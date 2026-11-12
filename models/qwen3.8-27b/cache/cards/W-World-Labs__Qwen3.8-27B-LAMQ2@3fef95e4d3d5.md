---
language:
- en
- zh
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- quantized
- w-modelforge
- lookatmyquant
- trellis-quantization
- multimodal
- vision-bf16
- custom-packed
- gguf
---

# Qwen3.8-27B · W-ModelForge / LookAtMyQuant PTQ

[中文模型说明](README.zh-CN.md)

**Public experimental LAMQ release. Independent original/packed comparisons at 2048 and 4096 tokens are complete. Current 4096 retention is 94.33%; material PPL loss and non-exact dense reconstruction are disclosed below.**

This is an experimental **language-quantized multimodal model with its original BF16 visual encoder and merger/projector preserved**, based on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0). It is maintained by [W-World-Labs](https://huggingface.co/W-World-Labs) using [W-ModelForge](https://github.com/W-World-Labs/w-modelforge) and Shin Megami Boson's [LookAtMyQuant](https://github.com/ShinMegamiBoson/lookatmyquant) method.

**Packed language directory: 9.643 GB, or 2.86438 bits per text parameter for weights plus manifest. Original BF16 vision adds 0.9215 GB.** At the current 4096-token cap, original/actual packed scores are **194/216 and 183/216**, a **94.33% relative correct-answer count**, with **17 original / 26 packed cap hits**. At 2048, original/packed counts are 160/155, or 96.88% retention; historical 1024 remains 84/86. These restricted synthetic-arithmetic results do not establish general reasoning equivalence. All 333 original BF16 visual tensor hashes match after loading, and two basic image examples pass.

<!-- adaptive-gguf:start -->
## Optional adaptive standard GGUF

A separate **fresh-original-BF16 standard GGUF** is now supplied under [gguf-adaptive/](gguf-adaptive/README.md), with [Chinese documentation](gguf-adaptive/README.zh-CN.md). Language **13.287 GB**, matching unquantized vision mmproj **0.931 GB**. Native math 4096 is **193→191/216 (98.96% relative count)**, with **20→19** cap hits; independent raw reports, PPL/KL, licenses and launch scripts accompany it. This uses per-tensor standard Q/IQ allocation from original weights, rather than conversion of the trellis artifact. The original native baseline and profile differ from the Torch scores below. The following packed-format recipe, file sizes and evaluations describe the original LAMQ release; its weights are unchanged. Download using `MODEL_FORMAT=gguf bash inference/download.sh W-World-Labs/Qwen3.8-27B-LAMQ2 OUTPUT`, or leave the default for packed files.
<!-- adaptive-gguf:end -->

## Format and practical limits

- `packed/artifact.json` and `packed/weights.bin` use the custom **`opgraph-packed-1`** format. The packed directory includes config, tokenizer, generation configuration, chat template and model license assets.
- Use the W-ModelForge reference loader. Standard Transformers, vLLM and llama.cpp cannot directly load this packed format; it is not GGUF.
- The reference loader **expands weights into a dense Torch model**. BF16 language plus original BF16 vision require approximately **54.7 GB of parameter storage**, plus decoding workspace, activations and KV/cache overhead. Download size is not inference VRAM usage.
- Dense HF export and separate GGUF conversion are provided below. They expand the weights and do not implement a native trellis inference kernel. No native packed throughput or engine performance claim is made.
- MTP is excluded. No image calibration, distillation or QAT was performed. Full vision/video quality and broad coding, tool-use and instruction-following quality are unbenchmarked.

## Original BF16 vision

`vision.safetensors` contains **all 333 original `model.visual.*` tensors**, including the visual encoder and merger/projector, from the same source revision. They retain their original BF16 bytes, with **no visual quantization or dtype conversion**. The file is **921,497,384 bytes**, SHA256 `d798b0c8c7432ec8164095ba6c1c36ee1c357f17b0f232b6950126bf76c6eb84`.

Original full configuration, tokenizer, image/video processor settings and chat template are included separately from the language packed directory. Actual full-model loading verified **all 333 visual tensor hashes against the original source**, with BF16 parameters and the visual RoPE buffer retained in FP32. See [vision provenance](vision-provenance.json) and the [actual 27B multimodal report](multimodal-validation.json).

Two image examples were run with the prompt: “Name the color of the large shape. Answer with one lowercase color word.” The visual forward module was called in both cases:

| Included image | Actual 27B response |
|---|---|
| [Red square](examples/vision-red-square.png) | `red` |
| [Blue circle](examples/vision-blue-circle.png) | `blue` |

These establish basic image functionality, **not broad visual or video quality**. The assembled full wrapper reports PPL 27.2671534 and mean KL 0.000690891 against the dense quantized reference on the same four 128-token windows, matching the independently loaded language packed measurements. This remains approximate reconstruction, with 98.6328% top-1 agreement and three of four greedy prompts matching the dense reference. Mathematics below was measured through the language execution path; it is not a multimodal benchmark.

## Evaluation

Original and actual packed models used the same frozen **216 synthetic arithmetic problems in 12 families**, BF16, batch **8**, greedy decoding and `enable_thinking=False`. This is **not a 216-question sample of the MATH test dataset**. MATH training questions were used separately for calibration. Each cap generated its own complete answers; the 1024 reports remain unchanged.

| Maximum new tokens | Original correct | Actual packed correct | Original cap hits | Packed cap hits | Relative correct-answer count |
|---|---:|---:|---:|---:|---:|
| 1024, historical | 84/216 | 86/216 | 131 | 130 | 102.38% |
| 2048 | 160/216 | 155/216 | 55 | 58 | 96.88% |
| **4096, current** | **194/216** | **183/216** | **17** | **26** | **94.33%** |

At 4096, original/packed accuracy is 89.81% / 84.72%; rounded mean generated lengths are 1545/1636 tokens. At 2048 the means are 1268/1280. Both requested longer caps meet the selected ≥90% relative threshold. Loss varies by family: at 4096 binomial falls 10→5 out of 18, multiplication 14→12, while modular exponentiation changes 17→18. Aggregate retention is not uniform retention across tasks.

All four 216-row generations were independently rescored; complete IDs, model/weight hashes, text assets, environment, dtype and batches were checked. Original generation used the initial driver; packed used a JSON-validation-only repair. Each raw report retains its real source fingerprint. The original complete-file SHA and the completed packed 2048 checkpoint remain unchanged. A Drive 403 during original 4096 backup recovered through retry without changing the generation. See [source/recovery notes](provenance/long-cap-recovery.json) and [independent audit](provenance/long-cap-audit.json).

| Measurement | Original | Quantizer's dense reconstruction | Actual packed |
|---|---:|---:|---:|
| Additional WikiText validation PPL, 4 × 128 | 18.4191 | 27.1521 | **27.2672** |
| Additional-window original-to-model mean KL, nats | — | 0.377257 | **0.379149** |
| Main WikiText held-out PPL, 8 × 512 | Not reported here | **12.3367** | **Not measured** |
| Main original-to-dense KL, nats | — | **0.335608** | **Not measured** |

**Actual packed PPL increases by 48.04% on the additional windows; quality loss is material.** The short validation and main held-out windows are different. Main English/Chinese/code held-out KL values are **0.287448 / 0.454961 / 0.202069 nats**, mean **0.314826**; those are **dense-only measurements**, not exact packed-equivalent scores.

### Reconstruction is not exact

On four fixed 128-token windows, dense-to-packed mean KL is **0.000690891 nats**, maximum KL **0.00753962**, top-1 agreement **98.6328%**, mean absolute log-probability difference **0.0363558**, and maximum difference **1.11023**. **Three of four** 32-token English/Chinese/code/arithmetic greedy samples match; the arithmetic prompt differs. These short samples are consistency checks, not coding or instruction-following benchmarks.

A [single CPU reconstruction sample](provenance/reconstruction-sample.json) found **319 differing BF16 values out of 5,242,880** in one small coded K projection, with maximum absolute difference **0.000244141**. This confirms non-identical weights in that sample; it does **not** establish the full cause of the output gap or whole-model/GPU equivalence. Byte-level packed integrity and numerical equality with the quantizer's dense checkpoint are separate properties.

See [evaluation.json](evaluation.json), current [2048/4096 original answers](evaluations/math-caps/original.json), [actual packed answers](evaluations/math-caps/packed.json), [comparison](evaluations/math-caps/comparison.json), historical [complete original math rows](evaluations/math216-original-1024.json), [complete packed validation rows](evaluations/packed-validation-1024.raw.json), [dense quality](evaluations/quality.json), [reference windows](evaluations/release-reference.json), [frozen protocol](evaluations/math216-protocol.json) and [actual full-wrapper/image report](multimodal-validation.json).

Both requested longer-cap mathematical thresholds, integrity checks and two basic image examples passed review. PPL/KL are finite, with material quality loss and non-exact reconstruction explicitly retained. **Approved as an experimental release under the selected criterion; this is not a claim of broad quality equivalence.** There are no broad reasoning, coding, tool-use, instruction-following, visual or video benchmarks, and no confidence interval.

## Quantization recipe

- BF16 source/text model execution; core quantization algebra stays FP64. CPU offload avoids keeping two complete 27B models on the GPU.
- Input-side randomized Hadamard transforms; trellis-coded quantization with L=12, row-mode width/lookahead 16/8, Viterbi decoding and LDLQ error feedback.
- Upstream cached EMA/e-means-inspired Gaussian codebooks, `trellis_fit=ema`. No FAISS/e-means implementation is vendored or imported by this release.
- Fisher-based row-group allocation using a `4^-bits` distortion proxy and 2/3/4-bit options; linear target 2.7 bpw, measured **2.716401 bpw**.
- Input embedding and untied output head each use **eight frequency buckets**, all 4-bit Lloyd–Max: seven × 32,768 rows plus 18,944 rows. The quantizer reports **4.003125 bpw** for these tables; final packed accounting separately includes the preserved FP32 scales, metadata and alignment. Per-bucket seeds follow the recorded plan; this is the actual memory-adapted recipe, not a bit-identical subdivision of a single table.
- 128 self-generated greedy solutions to MATH **training** questions produced 107 calibration windows of 1024 tokens; 32 windows were used for linear calibration. Digit-token weighting is 10× with floor 0.1; QEP alpha is 0.5. Fisher estimates use 12 × 512-token windows.
- Mixed English/Chinese/code text supplies vocabulary frequency ordering and held-out evaluation. Linear calibration uses the model's own MATH solutions.
- **PTQ only: no distillation or QAT.** This does not reproduce the original author's tuned 4B release recipe or benchmark results.

The exact plan is in [quantization-plan.json](provenance/quantization-plan.json). Documented memory handling also released 402 saved CPU tensor references after the dense checkpoint was written, before evaluation; weights and saved codes were not changed. The [resource scope](provenance/resource-scope.json) covers this retry build/pack trace only. Its 64,305 MiB whole-card peak overlaps a concurrent 9B validation and cannot be attributed to 27B alone or used as an overall project cost.

## Download and reference inference

Validated build environment: **PyTorch 2.14.1+cu130, Transformers 5.16.1, NumPy 2.4.6**, RTX PRO 6000 Blackwell. Install a CUDA-compatible PyTorch for your own machine. BF16 is this artifact's build dtype and the dtype used for actual packed/full-wrapper verification.

This public release supports anonymous download. From a W-ModelForge checkout:

```bash
git clone https://github.com/W-World-Labs/w-modelforge.git
git -C w-modelforge checkout 15b9d7ec56b0ad51dae136e05ffc684eead7cfb6
cd w-modelforge
python -m pip install transformers==5.16.1 numpy==2.4.6 huggingface_hub safetensors pillow
bash inference/download.sh W-World-Labs/Qwen3.8-27B-LAMQ2 ./models/qwen38-27b
bash inference/run_reference.sh ./models/qwen38-27b   --prompt 'Please introduce yourself in one sentence.' --max-new-tokens 128
# Use your own local image:
bash inference/run_reference.sh ./models/qwen38-27b   --image ./your-image.jpg --prompt 'Describe this image briefly.'
```

The download script verifies packed-file and original-vision sizes/SHA256. The reference script selects BF16 and passes the complete processor output, including multimodal token types. For a CPU with sufficient memory, set `DEVICE=cpu DECODE_DEVICE=cpu`; this is slower. Scripts copied from this model repository need `W_MODELFORGE_ROOT=/absolute/path/to/w-modelforge` so they can find the loader.

See the complete [English inference guide](inference/README.md) or [Chinese guide](inference/README.zh-CN.md).

## Standard HF export and optional engines

These paths are **implemented but have not been executed or quality/speed-validated for this 27B artifact**. They require expanded weights and additional disk/RAM; current packed-model scores must not be assigned to another engine or further quantization.

```bash
python methods/lookatmyquant/export_hf.py ./models/qwen38-27b ./expanded/qwen38-27b   --device cuda --decode-device cuda
# In a separate environment with a compatible vLLM installation:
bash inference/serve_vllm.sh ./expanded/qwen38-27b
```

The export creates standard sharded HF safetensors, full config, processor assets and a provenance receipt. It preserves BF16 language and original BF16 visual weights in the exported files, and does not redownload original language weights. The output directory must be new. Saving moves the model to CPU; reserve memory and disk for the approximately 54.7 GB expanded weights plus inputs/workspace. The vLLM wrapper defaults to BF16, `127.0.0.1:8000`, context 4096 and service name `w-modelforge`. Native support for the architecture is not proof this exported artifact has passed backend validation.

For llama.cpp, build the pinned upstream version and use separate language/mmproj files. Keep GGUF conversion in its own environment: its requirements select Torch 2.11 and Transformers 4.57.6, while `export_hf.py` needs the reference environment above. On Colab images without ensurepip, replace `python -m venv .venv-gguf` with `uv venv --seed --python python .venv-gguf`.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 93cf736a8d1d46aa922e27eb64aa712bcd5ebc30
python -m venv .venv-gguf
.venv-gguf/bin/python -m pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j 8
PATH="$PWD/.venv-gguf/bin:$PATH" \
  bash inference/convert_llamacpp.sh ./expanded/qwen38-27b ./llama.cpp ./gguf/qwen38-27b
LLAMA_SERVER=./llama.cpp/build/bin/llama-server \
  bash inference/serve_llamacpp.sh ./gguf/qwen38-27b/language.gguf ./gguf/qwen38-27b/mmproj.gguf --parallel 1 --reasoning off
```

On Apple Silicon, omit `-DGGML_CUDA=ON` to use the default Metal build. The wrapper requests BF16 language and BF16 mmproj, retaining converter-required F32 tensors. Only the language conversion needs `--no-mtp`; the vision converter already filters MTP. It splits visual convolutions and reorders language attention, so GGUF tensor layout/bytes differ from the original safetensors. No additional Q4/IQ/ternary quantization is applied. Any later requantization introduces a new loss and needs evaluation. Backend arguments, image requests and environment options are documented in the inference guides.

## Size and integrity

| Item | Bytes |
|---|---:|
| `packed/weights.bin` | **9,626,791,936** |
| `packed/artifact.json` | **3,248,591** |
| Complete packed directory, 8 files | **9,642,884,893** |
| Original BF16 `vision.safetensors` | **921,497,384** |
| Text parameter count | **26,895,998,464** |

`weights.bin` SHA256: `122ef326655feaf125765152885eaf07c22d616da7ee8db8da9bc936a254fd6c`.

`artifact.json` SHA256: `f297423d6549829eaa05082a7bd0088416d1d8658ac7abcd43008e25bcae0fdf`.

All 8 packed files and all 13,745 stored tensor hashes were checked; see [checksums.json](checksums.json). The private preservation snapshot `63ca6eb382f444fd8884b1afea8a6f828fa3ca8c` was remotely verified as **51 files / 10,587,799,867 bytes** including visual assets and documentation. Later metadata updates change that snapshot total; packed and visual sizes above are reported separately. Transfer verification is not a logits or quality assessment.

## Sources and licenses

- Base model: Qwen/Qwen3.8-27B, revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`, **Apache-2.0**. See [LICENSE](LICENSE) and [modifications](MODEL_MODIFICATIONS.md).
- Upstream compiler: ShinMegamiBoson/lookatmyquant, commit `4716ffee3da205a8d1df94e4abc629040893fa53`, **MIT**. Copyright and license are retained in W-ModelForge.
- Actual quantization core: W-ModelForge `5ecf73f499d206fc004619a5625a1848263983d1`; vocabulary allocator `fc5dfd78ea67236b4831d745d52757a39c81c5e5`.
- Language release evaluation: `ce4b06948c38af77a7026262b59af6c78943fb4e`. Original-vision extraction/full-wrapper loader: `e69f4f1fd9edee81ca4e212ee45074ea47a83901`; actual 27B loading verified 333 original BF16 tensor hashes and two basic image examples.
- Download, reference and export/engine scripts: W-ModelForge [`15b9d7ec56b0ad51dae136e05ffc684eead7cfb6`](https://github.com/W-World-Labs/w-modelforge/tree/15b9d7ec56b0ad51dae136e05ffc684eead7cfb6), **MIT**. Exact source metadata, the table-memory fix and runtime cleanup are under [provenance](provenance/README.md); compiler license: [MIT text](provenance/compiler-LICENSE-MIT.txt).
- vLLM and llama.cpp are separate external projects. Official support/conversion references are in the inference guide. The maintained [NOTICE](https://github.com/W-World-Labs/w-modelforge/blob/15b9d7ec56b0ad51dae136e05ffc684eead7cfb6/NOTICE.md) and [UPSTREAM](https://github.com/W-World-Labs/w-modelforge/blob/15b9d7ec56b0ad51dae136e05ffc684eead7cfb6/UPSTREAM.md) retain method/dependency/data attribution.

Model weights retain their Qwen model license; the compiler's MIT license does not replace it. Datasets retain their own terms. This release implies no endorsement by Qwen, Shin Megami Boson or Meta FAIR. Current results are from one run, without broad reasoning/coding/tool/vision/video benchmarks or confidence intervals. Completed measurements, their limitations and publication review must accompany public-quality claims; this experimental release remains limited to the measurements disclosed here.
