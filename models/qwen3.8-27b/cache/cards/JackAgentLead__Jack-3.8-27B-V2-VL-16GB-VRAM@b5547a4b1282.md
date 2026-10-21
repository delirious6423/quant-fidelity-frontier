---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
library_name: llama.cpp
pipeline_tag: image-text-to-text
tags:
- gguf
- qwen3.8
- 27b
- jack
- jack-xml
- agentic
- multimodal
- vision
- long-context
- long-agentic
- reasoning
- tool-calling
- 16gb-vram
- lm-studio
- state-management
- operative-recall
- counterfactual-reasoning
- deterministic-tools
- conversational
---

# Jack-3.8-27B-V2-VL-16GB-VRAM

![Jack V2](jack-v2-logo.png)

> ***Multimodal intelligence. Structured cognitive control. One 16GB GPU.***

**Main GGUF:** `Jack-3.8-27B-V2-VL-16GB-VRAM.gguf`  
**Vision projector:** `mmproj-Qwen3.8-27B-BF16.gguf`

## Author

**Jonathan Michael Langford**  
Independent Researcher  
Lead Architect, The Jack Project  
Email: `mlangford75@protonmail.com`  
GitHub: https://github.com/mlangford75-lgtm/mlangford75-lgtm

---

## Note from JML

For the intended Jack V2 configuration:

- **Keep MTP disabled.**
- **Use the included vision encoder / multimodal projector:** `mmproj-Qwen3.8-27B-BF16.gguf`
- **Initial local testing has shown strong results with Q4 KV cache.**

MTP is not part of the recommended operating profile for this release.

The vision projector should be loaded when using the model so the full VL capability of `Jack-3.8-27B-V2-VL-16GB-VRAM` is available.

Q4 KV cache has provided a strong early balance between memory efficiency and observed model quality in local testing. This is an initial behavioral observation, not a claim that Q4 KV cache is universally optimal across every runtime, context length, GPU, or workload.

— **JML**

### Support Jack

If Jack is useful to you and you want to support continued development of the Jack Project:

**[Support Jack / Jack Kernel](https://github.com/mlangford75-lgtm/Jack-Kernel)**

---

# What Is Jack V2?

Jack V2 is a local, multimodal, long-context agentic model built around a 27B Qwen3.8-derived cognitive engine and the Jack inference architecture.

It is the successor to:

**[Jack-3.8-27B-Coder-16GB-VRAM](https://huggingface.co/JackAgentLead/Jack-3.8-27B-Coder-16GB-VRAM)**

V2 changes the release substrate as well as the cognitive protocol:

- a different fine-tuned model substrate from V1
- a different quantization path
- the Qwen3.8 vision capability is retained through the included BF16 multimodal projector
- a revised embedded Jack Prime Directive
- mandatory post-reasoning Jack XML calibration
- deterministic tool verification when exact computation is required
- stronger separation of observation, configuration, counterfactual, assumption, and conclusion

Jack V2 is not built around the assumption that a larger context window automatically creates a better agent.

Its central problem remains:

> ***When tens of thousands of tokens contain old plans, corrected plans, failed experiments, verified results, superseded artifacts, competing configurations, contradictory conclusions, tool results, images, and evolving requirements, which information should control the next action?***

Jack treats that as an architectural problem.

---

# The Jack V2 Cognitive Pipeline

The embedded Jack Prime Directive establishes the intended execution order:

```text
NATIVE REASONING
        ↓
DETERMINISTIC TOOL USE WHEN REQUIRED
        ↓
JACK XML REASONING
        ↓
FINAL ANSWER
```

Native reasoning is allowed to search broadly before the structured Jack stage begins.

When an answer depends on exact arithmetic, code behavior, exhaustive search, hashes, recurrence evaluation, combinatorial computation, or another deterministically testable proposition, the intended workflow is to execute the appropriate external tool first.

The completed result is then rebound into the Jack XML stage.

The XML stage is the cognitive checkpoint immediately before final commitment.

---

# What Makes Jack Different

A capable base model can reason well and still fail a long-running agentic task.

It can:

- remember an old path and a new path, then use the old one
- remember a failed result and a verified result, then act on the failed one
- correctly solve two configurations independently, then mix them during synthesis
- identify a bug during reasoning, then emit code that reproduces it
- construct a valid reduced constraint set, then silently import inactive constraints
- write a test suite, then mistake the existence of tests for evidence that the tests were actually executed
- inspect an image correctly, then allow an unsupported textual assumption to replace the visual evidence
- preserve the full conversation while losing track of what is authoritative now

Jack is designed to reduce these failures by controlling the model's **operative state**, not merely by increasing the amount of text it can remember.

> ***Long context stores history. Jack reconstructs and rebinds the working state.***

---

# Operative Recall

Traditional long-context evaluation often asks:

> ***Can the model retrieve a fact that appeared earlier?***

Jack research asks a stricter question:

> ***Can the model retrieve the correct fact, preserve its provenance, distinguish it from superseded alternatives, determine whether it is still authoritative, bind it to the correct configuration, and continue acting on it after later state transitions?***

That is **operative recall**.

A model may remember both:

```text
PLAN_V1 = rejected
PLAN_V2 = authoritative
```

and still fail if `PLAN_V1` continues to influence execution.

It may remember:

```text
CONFIG_A = measured
CONFIG_B = untested
```

and still fail if the result from A is silently transferred to B.

It may remember:

```text
TOOL_RESULT_1 = invalid
TOOL_RESULT_2 = verified
```

and still fail if the invalid result remains behaviorally active.

Jack treats historical presence and current authority as different things.

> ***A statement can remain historically true as a record of what was believed without remaining authoritative as a statement of what is currently established.***

---

# Jack XML: Structured Attention Anchoring at the Generation Frontier

Jack XML is not merely a second reasoning format and it is not decorative output markup.

It is an inference-time control structure designed to organize and re-present the information that should govern the model's next action.

The Jack stage occurs after native reasoning and any required deterministic tool verification, immediately before final output or action.

That placement is intentional.

Native reasoning can extend for thousands of tokens. During a long reasoning trace, the most important conclusion may become separated from final output by rejected branches, intermediate calculations, speculative alternatives, edge-case analysis, tool discussion, self-correction, and additional reasoning.

Jack reconstructs the controlling state and places it **close to the generation frontier**.

```text
LONG CONTEXT / NATIVE REASONING
────────────────────────────────────────────
old state
new state
hypotheses
rejected branches
calculations
tool results
visual observations
counterfactuals
corrections
more reasoning
                         │
                         ▼
JACK XML REBINDING
────────────────────────────────────────────
<workspace_state>      operative state
<grounded_source>      controlling evidence
<anchor_fact>          exact critical variables / identities
<deterministic_check>  verified external result when applicable
<pitfall_check>        falsification / calibration boundary
                         │
                         ▼
FINAL OUTPUT / ACTION
```

The goal is not simply to repeat important information.

The goal is to make critical state:

- structurally distinct
- semantically grouped
- easy to re-attend to
- explicitly separated by provenance and configuration
- close to the output that must obey it

Jack's research hypothesis is that **structure + salience + frontier proximity** can improve alignment between the model's established state and its final behavior.

Jack does not claim that a particular XML tag has been mechanistically proven to edit a specific neural activation or transformer layer.

---

# The Core Jack Structures

## `<workspace_state>`

`<workspace_state>` reconstructs the current operative state.

It can preserve:

- current milestone
- exact active requirements
- active artifact identity
- active configuration
- paths
- versions
- parameters
- verified results
- rejected hypotheses
- unresolved questions
- remaining risks
- next required action

The objective is to move controlling variables out of deep history and back near the current generation frontier.

## `<grounded_source>`

`<grounded_source>` rebinds the evidence that should govern the answer.

It is designed to keep separate:

- direct observations
- supplied evidence
- retrieved evidence
- visual evidence
- deterministic tool output
- derived conclusions
- assumptions
- requested outcomes
- predictions
- unknown information

The aim is to prevent plausible prior knowledge or narrative momentum from silently replacing the evidence actually available in the current task.

## `<anchor_fact>`

`<anchor_fact>` assigns structurally distinct identities to critical state.

Examples:

```xml
<anchor_fact ID="ACTIVE_ARTIFACT">
Jack-3.8-27B-V2-VL-16GB-VRAM.gguf
</anchor_fact>
```

```xml
<anchor_fact ID="VISION_PROJECTOR">
mmproj-Qwen3.8-27B-BF16.gguf
</anchor_fact>
```

```xml
<anchor_fact ID="CONFIG_B_STATUS">
UNTESTED
</anchor_fact>
```

Anchoring is especially useful when a conversation contains several similar filenames, versions, model configurations, images, experimental conditions, paths, measurements, hypotheses, or tool outputs.

The purpose is not merely recall.

It is **coordinate isolation**: keep distinct states distinct.

## `<deterministic_check>`

When exact verification is required and an appropriate external tool is available, Jack is designed to use the real tool before the XML reasoning stage.

The completed check records:

- the exact deterministic subproblem
- the tool that actually executed
- the actual returned result
- the expected invariant
- PASS / FAIL
- verified status

A tool result that contradicts an emerging derivation should repair the reasoning before final commitment.

## `<pitfall_check>`

`<pitfall_check>` is Jack's adversarial falsification and calibration stage.

It is intended to inspect:

- unsupported assumptions
- confirmation bias
- counterfactual leakage
- configuration contamination
- missing evidence
- causal overreach
- plan/execution drift
- test/requirement mismatch
- false verification claims
- competing explanations
- unresolved ambiguity

Jack distinguishes:

- `SUPPORTED`
- `CONTRADICTED`
- `INSUFFICIENT EVIDENCE`

A claim that has not been proven is not automatically false.

A claim that has not been disproven is not automatically true.

> ***The search should be broad. The commitment should be calibrated.***

---

# Multimodal / Vision Operation

Jack V2 retains Qwen3.8 multimodal capability through:

`mmproj-Qwen3.8-27B-BF16.gguf`

The vision path is part of the intended V2 operating configuration.

For multimodal work, visual evidence can participate in the same Jack control architecture as textual evidence:

```text
IMAGE / VISUAL INPUT
        ↓
native multimodal interpretation
        ↓
<grounded_source>
        ↓
<anchor_fact>
        ↓
configuration / provenance separation
        ↓
<pitfall_check>
        ↓
FINAL ANSWER / ACTION
```

The goal is not merely to add image captioning.

The research direction is to make visual observations participate in the same long-context authority, grounding, falsification, and state-rebinding system used for text, tools, code, and agentic history.

---

# Deterministic Verification

A core Jack principle is:

> ***Probabilistic cognition may propose, but deterministic software must dispose.***

When a question depends on something that can be checked exactly, Jack is designed to use deterministic evidence when the runtime exposes an appropriate tool.

Examples include:

- arithmetic
- code execution
- exhaustive enumeration
- hashes
- file properties
- structured validation
- recurrence evaluation
- combinatorial search
- directly testable program behavior

The intended relationship is:

```text
model hypothesis
        ↓
deterministic check
        ↓
verified result
        ↓
Jack rebinds verified result into operative state
        ↓
later reasoning must use, explain, or challenge the evidence explicitly
```

A deterministic result should not become just another sentence buried in the transcript.

It should become part of the state that governs the next action.

---

# Configuration and Counterfactual Separation

A result observed under one system configuration does not automatically establish what will happen under another configuration.

Jack V2 explicitly separates:

```text
OBSERVED CONFIGURATION
        ≠
PROPOSED CONFIGURATION
```

This targets reasoning failures such as:

- transferring a measured failure rate from configuration A to untested configuration B
- assuming removal of a component is safe because failures occurred while the component was present
- assuming the component prevented worse failures without comparative evidence
- treating a requested outcome as established evidence
- converting absence of evidence into evidence of safety or harm

When the evidence does not determine a counterfactual outcome, Jack is instructed to preserve that uncertainty rather than invent a direction.

---

# Long Context Is Not Working Memory

A large context can contain all of the following at once:

- obsolete plans
- current plans
- invalid tool results
- corrected tool results
- superseded paths
- active paths
- rejected hypotheses
- surviving hypotheses
- tested configurations
- untested configurations
- old requirements
- amended requirements
- several artifact versions
- several models
- several images
- several experimental branches

All of them can remain technically accessible.

That does not mean all of them should remain behaviorally active.

Jack's long-context architecture repeatedly asks:

> ***What matters now?***

and then rebinds that answer near the current generation point.

> ***Preserve history without surrendering control to history.***

---

# Adaptive Native Reasoning

Jack separates native reasoning depth from the Jack structured checkpoint.

When native thinking is enabled, Jack is intended to search broadly: generate candidate failure modes, inspect edge cases, revisit assumptions, construct counterexamples, and challenge its own interpretation.

When a runtime supports reduced native reasoning for routine execution, the Jack XML checkpoint is still intended to remain active.

The architecture therefore treats native reasoning depth and operative-state alignment as related but distinct controls.

For the intended V2 release configuration, **MTP should remain disabled**.

---

# Agentic Work

Jack V2 is designed for work that unfolds over time:

- software engineering
- repository-scale debugging
- implementation from an established plan
- test construction and test auditing
- foreign-workflow reconstruction
- long-running technical investigation
- requirements tracking across many turns
- tool-assisted work
- multimodal technical inspection
- document / screenshot / diagram analysis
- review of work generated by another model
- plan → execute → review workflows
- changing configurations and artifacts
- repeated failure / correction cycles

For a real agent, the difficult problem is often not generating one answer.

It is preserving the correct working state across:

```text
requirement
→ plan
→ implementation
→ observation
→ failure
→ diagnosis
→ revised plan
→ new artifact
→ tool verification
→ visual evidence
→ external review
→ correction
→ final commitment
```

Every earlier state may remain inside the context.

Jack is designed to keep the current one operative.

---

# Structure Beats Scale

Jack belongs to a broader research program exploring a simple proposition:

> ***Scale determines potential capability. Structure helps determine how much of that capability remains reliably usable across time.***

This does not mean model scale is irrelevant.

Jack investigates a complementary problem:

> ***How much effective agentic capability can be recovered from a local model by improving the architecture around state, attention, reasoning, verification, grounding, and action?***

That question matters particularly for local deployment.

The target is:

> ***high effective agentic intelligence per unit of local memory and compute.***

---

# V1 → V2 Lineage

The previous release was:

**Jack-3.8-27B-Coder-16GB-VRAM**

V2 is not a simple filename change.

```text
Jack-3.8-27B-Coder-16GB-VRAM
            ↓
different fine-tuned substrate
different quantization path
vision capability retained
revised embedded Prime Directive
stronger evidence / counterfactual calibration
mandatory Jack XML checkpoint
            ↓
Jack-3.8-27B-V2-VL-16GB-VRAM
```

Results reported for the earlier Coder release should be understood as historical evidence from that release unless they have been rerun on V2.

V2-specific evaluation is treated separately.

---

# Source Model, Quantization, and Vision Credit

Jack V2 is based on the Qwen3.8-27B model family.

**Base model:**

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

**Quantized model source and vision projector:**

- [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)

The Jack V2 language-model artifact was built from the ISTA-DASLab Qwen3.8-27B GSQ-RCO GGUF release.

The V2 release also ships the associated:

`mmproj-Qwen3.8-27B-BF16.gguf`

for multimodal / vision capability.

Credit for the **GSQ / RCO quantization work and supplied multimodal projector** belongs to **ISTA-DASLab — the Deep Algorithms and Systems Lab at the Institute of Science and Technology Austria (ISTA)** and the authors of the associated research.

Jack V2 contributes the Jack inference-time cognitive architecture, embedded Prime Directive, structured XML reasoning checkpoint, operative-state reconstruction, grounding, deterministic verification protocol, counterfactual separation, and associated Jack research framework.

Please consult the ISTA-DASLab source repository and its cited GSQ / RCO papers for the underlying quantization methodology and research citations.

---

# Research Lineage

Jack is part of a broader research program by **Jonathan Michael Langford**, Independent Researcher and Lead Architect of The Jack Project.

Primary Jack research:

- **Rolling Cognitive State Maintenance: Why Structured Carry-Forward May Matter More Than Raw Context Length**  
  https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/Rolling%20Cognitive%20State%20Maintenance_%20Why%20Structured%20Carry-Forward%20May%20Matter%20More%20Than%20Raw%20Context%20Length.pdf

- **Engineering J-Space for Long-Context Recall**  
  https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/ENGINEERING%20J-SPACE%20FOR%20LONG-CONTEXT%20RECALL

- **Engineering the J-Space: Beyond Chain of Thought**  
  https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/ENGINEERING%20THE%20J-SPACE%3A%20BEYOND%20CHAIN%20OF%20THOUGHT

- **Structure Beats Scale**  
  https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/Structure_Beats_Scale_Langford_v2%20(2).pdf

The Jack research family explores adversarial cognition, structured attention anchoring, frontier state rebinding, rolling cognitive state maintenance, long-context operative recall, J-space engineering, counterfactual reflection, deterministic verification, selective reasoning, and local-model efficiency.

Earlier systems and experiments are part of Jack's research lineage. They should not be read as a claim that every historical component is literally implemented unchanged in V2.

---

# Research Preview / Experimental Model

Jack V2 is a research-oriented model.

It remains probabilistic.

It can:

- make reasoning errors
- generate false-positive audit hypotheses
- misinterpret ambiguous requirements
- ground an image incorrectly
- preserve an incorrect conclusion if it was incorrectly established upstream
- fail to retrieve relevant history
- over- or under-calibrate uncertainty
- fail when an appropriate deterministic verifier is unavailable
- produce structured XML that appears rigorous while the underlying reasoning is still wrong

The architecture is designed to reduce important classes of long-horizon failure.

It does not make them impossible.

---

# Recommended Runtime Behavior

For the intended Jack V2 behavior:

- use the model's **embedded chat template**
- do not override the model's prompt template unless intentionally testing an alternative
- start evaluation with the external **System Prompt blank**
- **keep MTP disabled**
- load `mmproj-Qwen3.8-27B-BF16.gguf`
- preserve conversation and tool history
- expose deterministic tools when available
- allow real tool execution when exact verification is appropriate
- keep the Jack XML stage active
- use a runtime capable of maintaining long context and multimodal history
- Q4 KV cache is the current early local recommendation based on initial testing

---

# Hardware Target

Jack V2 targets the practical **16GB VRAM consumer-GPU class**.

The language-model GGUF is in the approximately 12GB class and the release additionally uses the BF16 vision projector.

The 16GB target does not mean every context length, batch size, KV-cache precision, or runtime configuration will fit identically.

The intended release profile emphasizes local deployment rather than datacenter-only inference.

---

# Quantization

V2 uses a different quantization path from the original Jack 3.8 Coder release.

The quantized model source is the ISTA-DASLab Qwen3.8-27B GSQ-RCO GGUF release.

Quantization quality is treated as part of the model engineering rather than as a packaging afterthought.

Jack depends on the interaction of:

> ***model substrate + quantization quality + Jack cognitive architecture***

---

# What Jack Is Not

Jack V2 is not:

- a guarantee of perfect reasoning
- proof that a 27B model universally outperforms larger models
- a replacement for deterministic verification
- a claim that long context automatically equals working memory
- a claim that XML is merely a second chain of thought
- direct mechanistic proof that a particular XML tag controls a specific neural activation
- a claim that V1 benchmark or case-study results automatically transfer to V2
- a claim that Q4 KV cache is universally optimal
- a claim that vision grounding cannot fail

Jack is an inference-time cognitive control architecture integrated with a capable local multimodal model.

---

# Model Identity

| Field | Value |
|---|---|
| Model | `Jack-3.8-27B-V2-VL-16GB-VRAM` |
| Author | Jonathan Michael Langford — Independent Researcher, Lead Architect, The Jack Project |
| Contact | `mlangford75@protonmail.com` |
| Main GGUF | `Jack-3.8-27B-V2-VL-16GB-VRAM.gguf` |
| Vision projector | `mmproj-Qwen3.8-27B-BF16.gguf` |
| Model class | 27B Qwen3.8-derived multimodal model |
| Base family | `Qwen/Qwen3.8-27B` |
| Quantized source | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` |
| Format | GGUF |
| Target hardware | 16GB VRAM consumer GPU class |
| Vision | Supported through included BF16 multimodal projector |
| MTP | **Recommended OFF** |
| Native reasoning | Preserved |
| Jack XML | Mandatory structured reasoning / calibration checkpoint |
| Deterministic tools | Supported when exposed by runtime |
| External system prompt | Leave blank for intended embedded-template behavior |
| Predecessor | `Jack-3.8-27B-Coder-16GB-VRAM` |

---

# Core Design Principle

> ***The model is not being asked to become a different intelligence. It is being given an architecture for keeping the right intelligence operative at the right time.***

For short isolated tasks, native model capability may be enough.

For long-running agentic work, intelligence must survive changing state, repeated tool calls, superseded artifacts, corrections, images and other multimodal evidence, uncertainty, counterfactual branches, review, execution, and final commitment.

Jack is designed around that problem.

---

# One-Line Description

> ***Jack-3.8-27B-V2-VL-16GB-VRAM is a 27B Qwen3.8-derived local multimodal model with the Jack long-context cognitive architecture: native reasoning, deterministic verification, structured XML state rebinding, evidence grounding, adversarial falsification, counterfactual separation, operative recall, and vision-enabled agentic work on 16GB-class GPUs.***

---

# Research Direction

Jack asks a practical question:

> ***How much effective agentic intelligence can be recovered from local hardware by organizing cognition better rather than relying only on more parameters, more context, or permanently enabled deliberation?***

That is the research program.

