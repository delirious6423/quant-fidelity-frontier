---
base_model: zai-org/GLM-5.3-Flash
license: mit
library_name: gguf

tags:
  - gguf
  - llama.cpp
  - backpack

  - multimodal
  - image-text-to-text

---

# GLM-5.3-Flash — Backpack GGUF

GGUF quantizations of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), packaged for llama.cpp-compatible image-and-text inference and Backpack. 

## Model

| Property | Value |
| --- | --- |
| Original model | `zai-org/GLM-5.3-Flash` |
| Original publisher | zai-org |
| Upstream revision | `690b705278a3a58e538fcb37c2ca8b5f9511213c` |
| Architecture | Glm5NextForConditionalGeneration |
| Parameters | 321,323,031,390 |
| Context length | Not declared |
| Input modalities | text, image |
| Output modalities | text |
| License | mit |

## Available packages

| Quantization | Size | Approx. RAM | Recommended for |
| --- | ---: | ---: | --- |
| Q4_K_M | 180.5 GiB | 263.78 GB | Most users |


Memory values are estimates, not guarantees. Runtime configuration and context length change actual use.


### Multimodal projector

| File | Precision | Size |
| --- | --- | ---: |
| `GLM-5.3-Flash-mmproj-F16.gguf` | F16 | 1.1 GiB |

The projector is required for image input and must be used with one of the language-model GGUF files above.


## Backpack recommendation

**Recommended: Q4_K_M.** It usually offers a practical quality, size, and speed balance for local inference.

## Run with llama.cpp

Using the llama.cpp revision recorded below:

```bash
llama-mtmd-cli --model GLM-5.3-Flash-Q4_K_M.gguf --mmproj GLM-5.3-Flash-mmproj-F16.gguf --image image.jpg --prompt "Describe this image."
```

## Run with Backpack

These artifacts and `backpack-model.yaml` are prepared for the [Backpack AI workspace](https://backpack.run).

## Validation

Artifact integrity and GGUF metadata validation are the publication requirements. Runtime load,
inference, and tokenizer results are reported independently and do not imply a certification or
endorsement.

| Package | Integrity | Load | Inference | Tokenizer |
| --- | --- | --- | --- | --- |
| Q4_K_M | passed | failed | skipped | skipped |


Runtime execution validation has not completed successfully for every artifact. Treat the affected
package as experimental with the pinned toolchain until downstream runtime testing is complete.




- Packaged: 2026-09-06T07:07:13.528564+00:00
- llama.cpp toolchain revision: `8134115f88ed8018474e7db69afcfe97fb097fc4`
- SHA-256 checksums: see `checksums.sha256`

- `GLM-5.3-Flash-Q4_K_M.gguf`: `3e1f1720e869d98acd55a8f94b5efd78814a6ba0a2c2e4e609d637e9cca60406`


- `GLM-5.3-Flash-mmproj-F16.gguf`: `f64a2e935c899224054258d2372d9ad4c19b760141044292fa6ee0fb5ff36624`


## Provenance

The source model was resolved to immutable revision `690b705278a3a58e538fcb37c2ca8b5f9511213c`. It was converted with llama.cpp's `convert_hf_to_gguf.py`, including its multimodal projector, and quantized with `llama-quantize`; the exact toolchain revision is recorded above and in `backpack-model.yaml`.

## License and attribution

Upstream declares **mit**. Review the [upstream model card](https://huggingface.co/zai-org/GLM-5.3-Flash) and comply with all applicable terms.



Backpack does not claim ownership of the original model. These artifacts are packaged and quantized distributions of the upstream model.

## Disclaimer

Quantization can alter output quality. Memory estimates vary with runtime configuration, context length, and hardware.
