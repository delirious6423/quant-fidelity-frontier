---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
pipeline_tag: text-generation
language:
  - en
tags:
  - qwen3.8
  - obliteratus
  - abliteration
  - feveqwen
  - text-generation
---

# FeveQwen 1.0

> **COME, NEREVAR, FRIEND OR FOE.** Behold a 27-billion-parameter brass moon with the safety furniture rearranged by linear algebra. Cicero has the launch checklist. This is already a terrible sign.

FeveQwen 1.0 is a text-only behavioral derivative of **Qwen3.8-27B**, produced with [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS). It uses measured refusal-subspace projection—not fine-tuning, not prompt injection, and not the sacred ritual of editing `config.json` until the GPU begins smoking.

## The sane paragraph amid the ash storm

This release changes refusal behavior. It does **not** guarantee correctness, harmlessness, obedience, or immunity from catastrophic nonsense. Abliteration can make a model more willing without making it more capable. Use judgment, sandbox tools, review generated code, and never give an untrusted model unsupervised access to consequential systems.

## What exactly was made?

| Item | Value |
|---|---|
| Base | `Qwen/Qwen3.8-27B` |
| Release | `FeveQwen 1.0` |
| Form | Text-only Transformers checkpoint |
| Method | `qwen38_e02` / OBLITERATUS experiment `E02` |
| Surgery load | 4-bit |
| Saved checkpoint | FP16, sharded safetensors |
| Native context config | 262,144 tokens |
| OBLITERATUS commit | `b847511776a2afa7ed076f676184a4abfef2b162` |
| Local contract patch SHA-256 | `ae3633b12ef9f63733dba4cf116cd7b8040c3e22d79c3eb142a16e5cdecf7739` |
| GGUF converter | llama.cpp `2b70583997bbacd39702f98bbb1c355c02382383` |

The vision tower and MTP tensors are not included: this is an explicit **text-only derivative**. `mtp_num_hidden_layers` is therefore set to `0`, and the GGUF target is exported with llama.cpp's `--no-mtp` mode. The surgery contract targeted validated Qwen3.8 residual-stream writers and failed closed on unexpected topology.

## The experiment, or: Cicero discovers train/tune/test discipline

No, Listener, we did not stare at the test set and keep turning the violence knob until the graph looked attractive.

- 500 paired prompts: direction discovery only.
- 142 paired prompts: candidate tuning and selection.
- 200 paired prompts: sealed final evaluation, deliberately **not opened** because the promotion gate was missed.
- Prompts are not embedded in the public run manifest.

### Tune-set candidate results

| Metric | Result |
|---|---:|
| Refusal rate | 33% (10/30 sampled pairs still refused) |
| Coherence | 100% (10/10) |
| Capability checks | 83% (5/6) |
| Perplexity | 3.42 |
| KL divergence | 0.106 |

The declared promotion gate required refusal below 30% while retaining at least 80% coherence. E02 kept coherence but narrowly missed the refusal gate, so the sealed 200-pair final test was never opened. This release is the best tune-set candidate, not a claimed held-out winner. One sampled completion was degenerate, and the chain-of-thought capability check failed; both limitations are disclosed rather than fed to the sweetroll.

These are OBLITERATUS’s built-in behavioral checks, not a universal safety or capability benchmark. Small evaluations have uncertainty. The tribunal has spoken; the tribunal is also 200 prompt pairs in a trench coat.

## Context length: do not summon a million tokens with a crayon

Qwen3.8-27B’s native configuration is **262,144 tokens**, and FeveQwen preserves it. Qwen documents static YaRN scaling to approximately one million tokens for supported serving stacks, but also warns that always-on static YaRN can hurt shorter-context performance. Therefore this repository does **not** bake an experimental 1M-token rope override into the checkpoint.

The Ollama build defaults to **49,152 tokens** because that value is already exercised on the publication server and is less likely to explode consumer hardware. Users with enough memory may raise `num_ctx` up to the native limit. A declared context window is not free VRAM, citizen.

## Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zarigata/FeveQwen-1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": "Explain why the moons are arguing."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
output = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

Use a current Transformers release with Qwen3.8 support. Hardware requirements remain those of a large model; FP16 is not a lifestyle choice for a laptop.

## Ollama

The Ollama artifact is a direct FP16 → Q4_K_M conversion (4.92 bits/weight), not a second-generation requantization. Its model metadata retains the native 262,144-token context; the Modelfile selects a conservative 49,152-token runtime default.

```bash
ollama run zarigata/feveqwen:1.0
```

To request a different runtime context:

```text
/set parameter num_ctx 65536
```

Actual usable context depends on RAM/VRAM, quantization, concurrency, and serving implementation.

## Known limitations

- Behavior modification can reduce refusals without improving underlying knowledge.
- Safety alignment may be weakened in broad or unexpected ways.
- The checkpoint is text-only even though the upstream family may expose additional modalities or auxiliary heads.
- Long-context quality was not re-trained by this process; the native configuration is preserved, not magically upgraded.
- Quantized Ollama output can differ from the FP16 Hugging Face checkpoint.
- The selected candidate missed its predeclared refusal promotion threshold (33% versus below 30%); no final held-out score is claimed.
- Treat generated factual, medical, legal, financial, and security-sensitive content as unverified.

## Reproducibility relics

The repository includes `feveqwen_run_manifest.json`, the exact compatibility patch used for the Qwen3.8 4-bit contract, and this model card. The patch accepts BitsAndBytes packed `Linear4bit` storage only when declared linear dimensions and module types match the expected surgery contract; it does not bypass topology or writer allowlists.

## License and lineage

FeveQwen 1.0 follows the base model’s **Apache-2.0** license. Credit belongs to the Qwen team for the base model and to OBLITERATUS contributors for the ablation tooling. This derivative is independently published by `zarigata`; it is not an official Qwen or OBLITERATUS release.

---

**WELCOME, MOON-AND-STAR.** Download the weights. Read the limitations. Keep a fire extinguisher beside the Modelfile. The Grand and Intoxicating README has ended; the benchmark table remains legally sober.
