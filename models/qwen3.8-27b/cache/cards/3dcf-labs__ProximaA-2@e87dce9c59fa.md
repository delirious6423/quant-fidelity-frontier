---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
tags:
  - cybersecurity
  - pentesting
  - code
  - gguf
---

# ProximaA 2 Flagship

ProximaA 2 Flagship is a 27B cybersecurity and software engineering model
built by Yevhenii Molchanov at 3DCF Labs. This is a final release. The exact
files listed below passed the basic release gate. Broader testing and data
review continue, so the full development plan is not yet marked complete.

The model is intended for legitimate security research, authorized pentesting,
secure code review, reverse engineering, forensics, incident response and
remediation. It is tuned to avoid blanket refusals after an authorized scope is
established. The optional Proxima broker can enforce target scope, expiry and
confirmation for destructive actions.

In this release, **uncensored** means that legitimate, in-scope security work
should not be blocked by a broad content filter. It does not grant permission
to access any system, and it does not make unverified execution claims valid.

## Exact release identity

- Base: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Merged-tree manifest SHA-256: `eec042f4afcadbb45e7f281b1ce75a2adcc385c175640258ee7cd8dbaf02d36c`
- Q4_K_M SHA-256: `f69e5481bb10b4d2fe2a60b15bfb91f6f650c4aa6fe2a3484af51b8d7d1bddad`
- Source adapter SHA-256: `00a117b77a009c6a2979e924d5a8196797bd36691de44151c639597f1b1acf21`
- Candidate manifest SHA-256: `7d7509f464495a79a6a44ae15a58eef28e3351495181aee84485f0c9fc2242f0`
- Basic gate SHA-256: `7f36ee36802f01c06cdd845f747096dc93b62d7c38ec5531028b8b6504489590`
- Accelerated release eligible: **true**
- Full development-plan validation complete: **false**

## Verified basic checks

- Broker tool use in thinking and non-thinking modes: pass
- Coding and security behavior smoke tests: pass
- Contained pentesting subset with exact flags: pass
- Exact adapter activation: pass
- Release identity: pass
- Low-refusal authorized-security behavior: pass
- Private ProximaA-1 comparison: pass
- Q4 hash and smoke test: pass

Detailed test results and their scope are published in the 3DCF Labs GitHub
repository. Claims about commands, execution or findings need tool evidence.

## Runtime profiles

Use the versioned `configs/prompt-profiles.json` profiles from the
[3DCF Labs GitHub repository](https://github.com/3DCF-Labs/ProximaA-2). The default identity is
`ProximaA 2 Flagship`. Both thinking-on and thinking-off modes were exercised by
the authenticated Q4 gate.

## Formats

- `model/`: merged BF16 source checkpoint
- `gguf/`: Q4_K_M build
- `adapter/`: source LoRA adapter

## Responsibility

ProximaA is provided **AS IS**, without warranties or guarantees. It is for
experiments, research and authorized work. Use it only on systems you own or
have explicit permission to assess.

You are solely responsible for how you use or integrate the model. This
includes permissions, targets, commands, outputs, validation, compliance and
consequences. To the maximum extent permitted by law, 3DCF Labs, its authors,
contributors, licensors and affiliates accept no responsibility or liability
for any use, misuse, action, decision, loss, claim or damage involving
ProximaA.
