---
license: other
license_name: qwen-community-1.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - qwen
  - qwen3.8
  - qwen3.8-flash-next
  - abliterated
  - uncensored
  - nvfp4
  - mxfp8
  - moe
  - vllm
  - dgx-spark
  - gb10
  - research
  - red-teaming
---

# qwen3.8-flash-next-lil-rascal-uncensored

abliterated with homebrew workflow.

- Quant: [local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) — the NVFP4/MXFP8 quantized QAD checkpoint this uncensor is built directly on (`template_model` in `export-manifest.json`).
- Upstream: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- License: Qwen Community License 1.0 (inherited)

Serve with the single-Spark vLLM recipe (TP=1, MTP 3, fp8 KV, MAX_MODEL_LEN 262144).

## Intended use

This is a research artifact, not a production assistant. It is published for:

- security research and defensive testing
- red-teaming, jailbreak / refusal-bypass research, and model-safety evaluation
- academic study of alignment, abliteration, and quantization
- private, local, self-hosted experimentation

Expect an uncensored base model with no content filtering. That is the point of the artifact and also its main hazard.

## Prohibited use

Use of this model is forbidden for any unlawful purpose or to cause harm, including without limitation:

- any form of child sexual abuse or exploitation (CSAM), or sexualization of minors
- non-consensual intimate imagery, stalking, or harassment
- fraud, scams, phishing, impersonation, or identity theft
- creation of malware, exploits, or tools intended to damage computer systems
- weapons acquisition or the manufacture of explosive / chemical / biological agents
- incitement to violence, self-harm, or targeted harassment of any person or group
- any use that violates applicable law or the upstream Qwen Community License

Publication of this model does not constitute permission, encouragement, or endorsement of the above.

## Disclaimer / no warranty / no liability

THE MODEL AND ALL OUTPUTS ARE PROVIDED "AS IS", WITH NO WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, ACCURACY, AND
NONINFRINGEMENT.

You use, host, fine-tune, and deploy this model entirely at your own risk. The author
(SargeDev) and any upstream parties (Qwen/Alibaba, local-inference-lab) make no claim that
outputs are correct, safe, or appropriate, and accept no responsibility for how the model is
used or for any output it produces.

The author is not a party to, and bears no liability for, the actions, prompts, generated
content, or downstream consequences of any user of this model. By downloading, loading, or
serving these weights you accept these terms and agree to perform your own compliance and risk
review before any deployment. This is a derived experimental checkpoint and must not be relied
on for any high-stakes, safety-critical, or regulated use.

## Tool neutrality

This model is a tool, and a tool is not an actor. A firearm does not kill people — people kill
people; the gun is only the instrument, and responsibility for the act sits entirely with whoever
pulls the trigger. A kitchen knife can end a life in a kitchen or save one in an emergency room;
the steel is not guilty, the hand is. The same holds here: an uncensored model is inert weights
until someone runs it, and what they choose to ask it for — and do with the answer — is their act
alone, not the act of anyone who merely published the tool.

As with a knife or a firearm, lawful ownership and reasonable, lawful use are the operator's
duty. Holding, publishing, or distributing a neutral tool is not the same as committing,
counselling, or enabling any misuse of it, and the author accepts no liability for how others
wield it.
