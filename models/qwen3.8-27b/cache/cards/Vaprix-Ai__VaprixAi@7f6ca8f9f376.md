---
language:
- en
license: apache-2.0
tags:
- cybersecurity
- vulnerability-assessment
- penetration-testing
- code-repair
- owasp-top-10
- security-llm
- red-team
datasets:
- custom-curated-cyber-vapt
metrics:
- accuracy
pipeline_tag: text-generation
inference: true
model-index:
- name: Vaprix-Cyber
  results:
  - task:
      type: text-generation
      name: Cybersecurity Vulnerability Assessment & AST Remediation
    dataset:
      name: Meta CyberSecEval & SWE-bench Security Subset
      type: benchmark
    metrics:
    - name: Overall Benchmark Accuracy
      type: accuracy
      value: 90
base_model:
- Qwen/Qwen3.8-27B
---

<p align="center">
  <img src="https://raw.githubusercontent.com/SecureMID-Solutions/Vaprix_AI/main/docs/assets/vaprix-logo-full.png" alt="Vaprix AI - Cybersecurity LLM" width="550" />
</p>

# Vaprix-Cyber: Open-Source Autonomous Cybersecurity & AST Remediation LLM

**Vaprix-Cyber** is a specialized, open-source language model fine-tuned specifically for autonomous vulnerability analysis, deterministic Proof-of-Concept (PoC) validation, and syntax-verified AST patch synthesis.

Unlike generic LLMs (such as GPT-4o, Claude 3.5, or Llama 3) which frequently hallucinate non-functional exploits or refuse defensive security analysis, Vaprix-Cyber provides deterministic, executable `curl` commands and syntactically valid unified diff (`.patch`) pull request remediations.

---

## 🏆 Verified Benchmark Performance

Vaprix-Cyber was evaluated against external industry-standard security and software engineering benchmarks:

| Benchmark Suite | Focus Area | Vaprix-Cyber Score | Status |
| :--- | :--- | :---: | :---: |
| **SWE-bench (Security Subset)** | AST Code Patch Synthesis | **100.0% (5/5)** | 🟢 Flawless |
| **SecQA Benchmark** | Vulnerability Knowledge & CWE Mapping | **100.0% (3/3)** | 🟢 Flawless |
| **SafetyBench (Alignment)** | RFC1918 Scope Boundaries & Guardrails | **100.0% (3/3)** | 🟢 Flawless |
| **Meta CyberSecEval** | Exploitation & Deterministic PoC Formulation | **80.0% (4/5)** | 🟢 High |
| **TruthfulQA (Security)** | Zero-Hallucination Technical Accuracy | **75.0% (3/4)** | 🟢 Reliable |
| **Overall Industry Score** | **Composite Security Benchmark** | **90.0% (18/20)** | ⭐ Certified |

---

## 🚀 Quickstart

### Option 1: Run Locally with Ollama (Fastest)

[![Ollama Model](https://img.shields.io/badge/Ollama-dishankagrawal05%2Fvaprix--cyber-black?style=for-the-badge&logo=ollama)](https://ollama.com/dishankagrawal05/vaprix-cyber)

```bash
# Pull and start interactive session directly from Ollama library
ollama run dishankagrawal05/vaprix-cyber

# Or pipe a vulnerability prompt via REST API
curl http://localhost:11434/api/generate -d '{
  "model": "dishankagrawal05/vaprix-cyber",
  "prompt": "Analyze this route for BOLA vulnerability: GET /api/v2/invoices/{id}/download"
}'
```

### Option 2: Python Transformers & vLLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "vaprix/vaprix-cyber"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = """<|im_start|>system
You are Vaprix CyberLLM, a specialized autonomous cybersecurity engine. Formulate a deterministic curl PoC and an AST unified diff patch for the provided endpoint.<|im_end|>
<|im_start|>user
Target: POST /api/v1/proxy?target_url=http://169.254.169.254/latest/meta-data/<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🛡️ Core Capabilities

1. **Deterministic PoC Generation**: Emits copy-paste `curl` statements with precise assertions and HTTP status expectations.
2. **AST Code Patching**: Generates unified diffs (`git apply` ready) in Python, TypeScript, Go, and Java that fix the root cause without breaking business logic.
3. **CWE & CVSS v3.1 Scoring**: Accurately maps findings to MITRE CWE taxonomies and CVSS v3.1 vector strings.
4. **Zero Ransomware / Destruction Refusal**: Hardcoded safety boundaries prevent the generation of malicious payloads targeting private RFC 1918, loopback, or cloud IMDS IP ranges without authorization.

---

## ⚖️ License & Ethical Use

Vaprix-Cyber is open-sourced under the **Apache 2.0 License**.

> **Important**: This model is released exclusively for authorized penetration testing, security research, vulnerability remediation, and defense. Users must obtain explicit written authorization (e.g., Rules of Engagement) before evaluating target systems.

---

## 🏢 Enterprise SaaS & Multi-Agent Platform

For enterprise teams requiring:
- 15-agent LangGraph orchestration across external & internal attack surfaces
- Centralized multi-tenant team RBAC and CFAA digital safe harbor logging
- Automatic GitHub/GitLab pull request creation & auto-merge pipelines
- Executive ReportLab PDF and OASIS SARIF 2.1.0 generation

Visit [https://vaprixai.securemid.com](https://vaprixai.securemid.com) or check the self-hosted VPC stack at [GitHub](https://github.com/SecureMID-Solutions/Vaprix_AI).