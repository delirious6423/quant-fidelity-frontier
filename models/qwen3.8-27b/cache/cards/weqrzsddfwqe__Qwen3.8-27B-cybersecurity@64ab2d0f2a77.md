
---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen3.8-27B
tags:
- qwen3.8
- 27b
- cybersecurity
- security
- code-review
- pentesting
- ctf
- reasoning
- thinking
- distillation
- conversational
- unsloth
---

A cybersecurity / code-review specialist fine-tuned from Qwen3.8-27B by distilling reasoning and answers from larger open models (GLM-5.3, Kimi K3).

What it is good at
Security code review: finding vulnerabilities, explaining impact, suggesting fixes
Practical security knowledge from penetration-testing writeups and CTF walkthroughs
Step-by-step reasoning before answering (Qwen thinking mode, <think> ... </think>)
Training
	
Base model	Qwen/Qwen3.8-27B
Method	QLoRA (4-bit base), LoRA rank 32 on all attention + MLP layers, merged to 16-bit
Data	~20,000 examples: Kimi K3 distillation data, GLM-5.3 answers to domain questions, and Q&A that GLM-5.3 wrote from public security writeups
Sequence length	6,144 tokens
Epochs	1
Hardware	1x RTX 5090 (32 GB), ~5 hours

Loss was computed only on the assistant's reasoning + answer, not the prompt.

How to use

llama.cpp (GGUF repo: weqrzsddfwqe/Qwen3.8-27B-cybersecurity):

bash
llama-server -m Qwen3.8-27B-cybersecurity-Q4_K_M.gguf --jinja --reasoning-format deepseek -ngl 99 -fa on -c 262144

vLLM:

bash
vllm serve weqrzsddfwqe/Qwen3.8-27B-cybersecurity --max-model-len 262144 --reasoning-parser qwen3

Recommended sampling: temperature 0.6, top_p 0.95, top_k 20.

Intended use and limitations
For education, defensive security, code review and authorized security testing only.
Fine-tuned without tool-calling data: test tool use in your agent before relying on it.
Can be confidently wrong. Verify findings before acting on them.
Knowledge comes from the training material and the base model; it does not know recent CVEs.