---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
- exl3
- quantized
- vllm
- multimodal
- image-text-to-text
pipeline_tag: image-text-to-text
---

# Recursant Two Sparks — DeepSeek V4.1 Flash (retained EXL3 MCG2 pack)

This repository redistributes the exact retained model pack used by the [matched two-Spark serving recipe](https://github.com/ajensenwaud/recursant-two-sparks-deepseekv4.1-flash). It is not a new quantization or a claim of authorship of DeepSeek's model or the upstream EXL3 work.

## Attribution and provenance

- Original model: [DeepSeek / DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). Audited upstream repository revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
- Upstream EXL3 distribution: [sfxnz / DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3). Audited upstream repository revision: `4f93d8d2d5fd86de4be4f28495df587639fa6377`.
- These upstream revisions identify the repositories inspected during the provenance audit; they do not assert that downloading either upstream revision reproduces every byte of this retained pack. The authoritative identity of this redistribution is the 53-file SHA-256 manifest in `model-manifest.json`.
- The original MIT `LICENSE`, including DeepSeek's copyright notice, is preserved unchanged. Runtime components have their own licenses; this model repository does not change those licenses.

## Format and intended use

The pack contains native EXL3 2-bit MCG routed experts, retained non-routed MXFP8 weights (E4M3 data and UE8M0 scales), the vision components, and the DSpark draft components configured for block size K=5. It includes 48 Safetensors shards plus configuration, tokenizer files, index, and license: 53 retained files totaling 358,128,021,424 bytes.

Use the matched serving recipe and its runtime compatibility checks. This is not a generic Transformers `from_pretrained` compatibility claim, a GGUF pack, or a drop-in checkpoint for arbitrary vLLM releases. The recipe covers the specialized two-Spark runtime and Engram handling. The large Engram shards remain intact; no resharding or new conversion was performed for publication.

## Verification and limitations

Pin an immutable commit when downloading and compare every retained file with `model-manifest.json`. Do not treat a repository that is still receiving its initial upload as complete: all 53 listed files and hashes must match. The manifest excludes this README and itself to avoid a self-hash cycle.

The retained pack has been exercised with its matched text/reasoning and vision serving configuration. This is not a comprehensive quality, safety, context-length, or hardware-portability evaluation. Architectural context limits in `config.json` are not a promise that those limits fit the reference hardware. Models can hallucinate, produce unsafe or biased content, or misinterpret images; evaluate outputs for the intended use.

Model publication and container-image publication are separate release gates. Refer to the serving repository's current release metadata for image availability; this model card does not claim that a public container image is available.
