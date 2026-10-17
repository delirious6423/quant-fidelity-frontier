---
tags:
- 27b
- gguf
- jack
- jack-xml
- agentic
- coding
- coder
- long-context
- long-agentic
- multi-turn
- tool-calling
- reasoning
- 16gb-vram
- lm-studio
- llama.cpp
- state-management
- operative-recall
- counterfactual-reasoning
- deterministic-tools
- global-workspace
library_name: llama.cpp
pipeline_tag: text-generation
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---

# Jack-3.8-27B-Coder-16GB-VRAM

<p align="center">
  <img src="https://huggingface.co/JackAgentLead/Jack-3.8-27B-Coder-16GB-VRAM/resolve/main/jack-3.8-coder.png" alt="Jack 3.8 Coder" width="600">
</p>

> **Coding intelligence. Structured cognitive control. One 16GB GPU.**

**Final GGUF:** `Jack-3.8-27B-Coder-16GB-VRAM.gguf`

## Author

**Jonathan Michael Langford**  
Independent Researcher  
Lead Architect, The Jack Project  
Email: `mlangford75@protonmail.com`  
GitHub: https://github.com/mlangford75-lgtm/mlangford75-lgtm

---

Jack 3.8 Coder is a local, long-context, agentic coding model built around a 27B Qwen3.8-derived cognitive engine and the Jack inference architecture.

It is designed for software engineering that unfolds over time:

- planning
- implementation
- debugging
- code review
- test design
- repository-scale work
- tool use
- long-running technical investigations
- multi-stage agent workflows
- work in which requirements, artifacts, configurations, and conclusions change as the task progresses

Jack is not built around the assumption that a larger context window automatically creates a better agent.

Its central problem is harder:

> **When tens of thousands of tokens contain old plans, corrected plans, failed experiments, verified results, superseded artifacts, competing configurations, and contradictory conclusions, which information should control the next action?**

Jack treats that as an architectural problem.

---

# What Makes Jack Different

A capable base model can reason well and still fail a long-running engineering task.

It can:

- remember an old path and a new path, then use the old one
- remember a failed result and a verified result, then act on the failed one
- correctly solve two configurations independently, then mix them during synthesis
- identify a bug during reasoning, then emit code that reproduces it
- construct a valid reduced constraint set, then silently import inactive constraints
- write a test suite, then mistake the existence of tests for evidence that the tests were actually executed
- preserve the full conversation while losing track of what is authoritative now

Jack is designed to reduce these failures by controlling the model's **operative state**, not merely by increasing the amount of text it can remember.

A useful distinction is:

> **Long context stores history. Jack reconstructs and rebinds the working state.**

---

# Operative Recall

Traditional long-context evaluation often asks:

> Can the model retrieve a fact that appeared earlier?

Jack research asks a stricter question:

> **Can the model retrieve the correct fact, preserve its provenance, distinguish it from superseded alternatives, determine whether it is still authoritative, bind it to the correct configuration, and continue acting on it after later state transitions?**

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

> **A statement can remain historically true as a record of what was believed without remaining authoritative as a statement of what is currently established.**

---

# Jack XML: Structured Attention Anchoring at the Generation Frontier

Jack XML is not merely a second reasoning format and it is not decorative output markup.

It is an inference-time control structure designed to organize and re-present the information that should govern the model's next action.

The Jack stage occurs immediately before final output or action.

That placement is intentional.

Native reasoning can extend for thousands of tokens. During a long reasoning trace, the most important conclusion may be separated from the final output by:

- rejected branches
- intermediate calculations
- speculative alternatives
- edge-case analysis
- tool discussion
- self-correction
- additional reasoning

Jack reconstructs the controlling state and places it **close to the generation frontier**, where the next tokens are produced.

Conceptually:

```text
LONG CONTEXT / NATIVE REASONING
────────────────────────────────────────────
old state
new state
hypotheses
rejected branches
calculations
tool results
counterfactuals
corrections
more reasoning
more reasoning
more reasoning
                         │
                         ▼
JACK XML REBINDING
────────────────────────────────────────────
<workspace_state>   operative state
<grounded_source>   evidence that controls the conclusion
<anchor_fact>       exact critical variables and identities
<deterministic_check> verified external results when applicable
<pitfall_check>     active falsification / failure boundaries
                         │
                         ▼
FINAL OUTPUT / ACTION
```

The goal is not simply to repeat important information.

The goal is to make critical state:

- structurally distinct
- semantically grouped
- easy to re-attend to
- close to the output that must obey it

Jack's research hypothesis is that this combination of **structure + salience + frontier proximity** can improve alignment between the model's established state and its final behavior.

The behavioral evidence in Jack testing supports the usefulness of this architecture.

Jack does **not** claim that these behavioral results alone prove that a particular XML tag directly edits a specific internal neural activation.

---

# The Core Jack Structures

## `<workspace_state>`

`<workspace_state>` reconstructs the current operative state.

It is not intended to summarize the entire conversation.

It can preserve:

- current milestone
- exact active requirements
- active artifact identity
- active configuration
- paths
- versions
- verified results
- rejected hypotheses
- unresolved questions
- remaining risks
- next required action

The objective is to move the controlling variables out of deep history and back near the current generation frontier.

---

## `<grounded_source>`

`<grounded_source>` rebinds the evidence that should govern the answer.

It is designed to keep separate:

- direct observations
- supplied evidence
- retrieved evidence
- deterministic tool output
- derived conclusions
- assumptions
- requested outcomes
- predictions
- unknown information

This targets a common model failure:

```text
plausible prior knowledge
        ↓
silently replaces
        ↓
the evidence actually supplied in context
```

---

## `<anchor_fact>`

`<anchor_fact>` assigns structurally distinct identities to critical state.

Examples:

```xml
<anchor_fact ID="ACTIVE_ARTIFACT">
Jack-3.8-27B-Coder-16GB-VRAM.gguf
</anchor_fact>
```

```xml
<anchor_fact ID="CONFIG_D_STATUS">
UNTESTED
</anchor_fact>
```

```xml
<anchor_fact ID="VERIFIED_RESULT">
PASSED
</anchor_fact>
```

Anchoring is especially useful when a conversation contains several similar:

- filenames
- versions
- model configurations
- experimental conditions
- paths
- measurements
- hypotheses
- tool outputs

The purpose is not merely recall.

It is **coordinate isolation**: keep distinct states distinct.

---

## `<pitfall_check>`

`<pitfall_check>` is Jack's adversarial falsification stage.

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

The goal is not automatic disagreement.

Jack distinguishes:

- `SUPPORTED`
- `CONTRADICTED`
- `INSUFFICIENT EVIDENCE`

A claim that has not been proven is not automatically false.

A claim that has not been disproven is not automatically true.

---

# Exhaustive Reasoning Is Intentional

When native thinking is enabled, Jack is intended to search broadly.

It may:

- generate many candidate failure modes
- inspect low-probability edge cases
- revisit apparently settled assumptions
- search for contradictions
- construct counterexamples
- challenge its own interpretation
- investigate the tests as well as the code
- examine the claimed evidence for correctness

This increases the number of hypotheses Jack considers and therefore also increases the number of opportunities for a candidate hypothesis to be wrong.

That is not the same as poor auditing.

For exhaustive review, the more useful measures are:

- true failures discovered
- important failures uniquely discovered
- severity-weighted recall
- false positives relative to search breadth
- confidence calibration
- whether weak hypotheses are separated from established defects

> **The search should be broad. The commitment should be calibrated.**

---

# Adaptive Native Thinking in Agentic Workflows

Jack can be used inside an agentic workflow that changes native reasoning depth according to the task.

Jack does **not** require native thinking to remain enabled for every execution token.

A typical workflow can look like:

```text
PLAN / NOVEL PROBLEM
Native thinking: ON
Jack XML: ACTIVE
        │
        ▼
expansive planning, search, falsification

ROUTINE EXECUTION
Native thinking: OFF
Jack XML: ACTIVE
        │
        ▼
aligned execution under anchored state

NEW FAILURE / UNCERTAINTY
Native thinking: ON
Jack XML: ACTIVE
        │
        ▼
deep debugging and hypothesis search

CONTINUED EXECUTION
Native thinking: OFF
Jack XML: ACTIVE
        │
        ▼
efficient implementation

INDEPENDENT REVIEW
Native thinking: ON
Jack XML: ACTIVE
        │
        ▼
adversarial audit and recalibration
```

The important distinction is:

> **Native thinking changes reasoning depth. Jack XML maintains cognitive alignment and can still drive structured self-correction with native thinking disabled.**

Thinking OFF is therefore not "Jack turned off."

The Jack architecture remains active.

The XML structures continue to provide:

- attention anchors
- operative-state reconstruction
- constraint salience
- evidence grounding
- failure boundaries
- configuration separation
- final-output alignment

This makes Jack suitable for agentic workflows where expensive deliberation is allocated selectively rather than regenerated during every routine execution step.

---

# Thinking-OFF Ablation: Jack Still Self-Corrects

A controlled state-machine audit was run with **native thinking disabled** while the embedded Jack XML architecture remained active.

The task required Jack to reconstruct a versioned distributed state machine after a forensic rollback, preserve original transaction base versions, evaluate a later correction, classify competing auditor claims, solve a counterfactual history, and state the governing invariant.

During the Jack XML stage, the model initially produced two incorrect intermediate anchor states. It then independently re-entered the problem through `<pitfall_check>`, re-evaluated the authoritative rollback rules, explicitly marked the earlier anchor as wrong, and repaired the state before final commitment.

Observed trajectory:

```text
native thinking: OFF
        ↓
<workspace_state>
        ↓
<grounded_source>
        ↓
initial <anchor_fact> values
        ↓
<pitfall_check>
        ↓
detect inconsistency
        ↓
supersede incorrect anchors
        ↓
correct final authoritative state
```

The final rollback state was reconstructed correctly:

```text
Version = 2
A = 11
B = 9
C = 2
P = true
Q = false
```

This matters because the correction occurred **without native deliberative reasoning being enabled**.

The run also exposed an important boundary. Jack with thinking OFF still accepted one auditor claim whose overall conclusion was correct but whose exact causal rationale was false. In a matched run with native thinking ON, Jack rejected that claim **as written**, preserving the finer semantic distinction.

The current behavioral interpretation is therefore:

> **Jack XML provides structured state reconstruction, grounding, adversarial checking, self-correction, and final-state rebinding in both modes. Native thinking adds additional search depth for difficult semantic and causal distinctions.**

This is an empirical behavioral result, not a claim that a specific XML token has been causally localized to a specific internal neural coordinate.


---

# Coding and Software Engineering Focus

The final Jack 3.8 release is explicitly a **Coder** model.

Primary intended workloads include:

- long-running coding sessions
- implementation from an established plan
- repository-scale debugging
- multi-file engineering
- refactoring
- test construction
- test auditing
- foreign-code review
- review of work generated by another model
- configuration debugging
- build and runtime diagnosis
- tool-assisted software engineering
- long-horizon project continuation
- requirements tracking across many turns
- debugging after repeated failed attempts

Jack is especially relevant when the coding problem is not isolated to one prompt.

For a real software agent, the difficult problem is often not generating a function.

It is preserving the correct engineering state across:

```text
requirement
→ plan
→ implementation
→ failure
→ diagnosis
→ revised plan
→ new artifact
→ test
→ external review
→ correction
→ final verification
```

Every earlier state may remain inside the context.

Jack is designed to keep the current one operative.

---

# Foreign-Workflow Reconstruction and Independent Audit

One of the strongest observed behaviors of Jack 3.8 is its ability to enter a workflow **after another model has already done the work**.

In controlled coding evaluations, Jack was given foreign trajectories containing:

- another model's interpretation of the task
- another model's plan
- another model's implementation
- another model's tests
- another model's correctness claims

Jack then had to reconstruct the authority structure from scratch and audit the complete chain.

The review target is broader than ordinary code inspection:

```text
original requirement
        ↕
derived plan
        ↕
implementation
        ↕
tests
        ↕
claimed verification
```

Jack has demonstrated the ability to inspect all of these levels, including weaknesses that a previous model's own self-review did not surface.

Examples observed in controlled evaluations include scrutiny of:

- whether a test actually proves the property it claims to test
- whether an invalid record still participates in global duplicate semantics
- whether truthy values are accidentally accepted where exact `True` is required
- whether test code was merely written versus actually demonstrated as executed
- whether a plan introduced a restriction not present in the original requirements
- whether implementation silently changed the planning state

These are controlled case studies, not a universal leaderboard claim.

They demonstrate the kind of auditing behavior Jack is designed to perform.

---

# State-Preserving Plan → Execute → Review

Jack 3.8 has also been tested in workflows where native reasoning mode changes between stages.

A representative pattern:

```text
PLANNING
Native thinking: ON
        ↓
Jack reconstructs and anchors the authoritative plan

EXECUTION
Native thinking: OFF
        ↓
Jack carries out the established plan under active XML alignment

REVIEW
Native thinking: ON
        ↓
Jack independently falsifies the result
```

In observed coding tests, Jack preserved plan constraints through the no-thinking execution stage without silently redesigning the solution.

This is important because the architectural goal is not merely to make thinking shorter.

It is to make **reasoning depth adaptive while keeping the controlling state aligned across modes**.

---

# Deterministic Verification

A core Jack principle is:

> **Probabilistic cognition may propose, but deterministic software must dispose.**

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
- several experimental branches

All of them can remain technically accessible.

That does not mean all of them should remain behaviorally active.

Jack's long-context architecture repeatedly asks:

> **What matters now?**

and then rebinds that answer near the current generation point.

This is why Jack's long-context goal is not simply "remember more."

It is:

> **Preserve history without surrendering control to history.**

---

# Structure Beats Scale

Jack belongs to a broader research program exploring a simple proposition:

> **Scale determines potential capability. Structure helps determine how much of that capability remains reliably usable across time.**

This does not mean model scale is irrelevant.

Larger models generally provide more raw representational and reasoning capacity.

Jack investigates a complementary problem:

> How much effective agentic capability can be recovered from a local model by improving the architecture around state, attention, reasoning, verification, and action?

That question matters particularly for local deployment.

Jack 3.8 is designed to fit entirely within the practical **16GB VRAM consumer-GPU class** while still supporting large-context agentic coding work.

The target is not merely "a model that fits."

The target is:

> **high effective agentic intelligence per unit of local memory and compute.**

---

# Relationship to Global Workspace Research

Anthropic's 2026 research, **A Global Workspace in Language Models**, reports evidence for a small, causally important internal workspace associated with deliberate, flexible and reusable computation in language models.

Anthropic research:
- https://www.anthropic.com/research/global-workspace
- https://transformer-circuits.pub/2026/workspace/index.html

Related Jack research:
- https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/Rolling%20Cognitive%20State%20Maintenance_%20Why%20Structured%20Carry-Forward%20May%20Matter%20More%20Than%20Raw%20Context%20Length.pdf
- https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/ENGINEERING%20J-SPACE%20FOR%20LONG-CONTEXT%20RECALL
- https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/ENGINEERING%20THE%20J-SPACE%3A%20BEYOND%20CHAIN%20OF%20THOUGHT
- https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/Structure_Beats_Scale_Langford_v2%20(2).pdf

The Jack research family explores a complementary engineering question:

> **If language models contain a limited internal workspace used for deliberate cognition, can inference-time structure help determine which information is salient, separated, reconstructed, and positioned to govern downstream action?**

Jack's approach is external and behavioral.

It uses:

- structured XML
- semantic grouping
- active-state reconstruction
- attention anchoring
- evidence rebinding
- counterfactual separation
- adversarial falsification
- frontier proximity

The current Jack 3.8 behavioral results do not by themselves prove that a specific XML tag directly edits a specific neural representation or transformer layer.

The stronger claim is:

> **Structured Jack inference changes the information environment immediately preceding action, and controlled evaluations show meaningful differences in state preservation, auditing, execution fidelity, and long-horizon behavior.**

---

# Research Lineage

Jack 3.8 is the current model in a broader family of work by **Jonathan Michael Langford**, Independent Researcher and Lead Architect of The Jack Project.

Author / project links:

- **Jonathan Michael Langford**  
  Independent Researcher  
  Lead Architect, The Jack Project  
  Email: `mlangford75@protonmail.com`
- **GitHub:**  
  https://github.com/mlangford75-lgtm/mlangford75-lgtm

Primary Jack research:

- **Rolling Cognitive State Maintenance: Why Structured Carry-Forward May Matter More Than Raw Context Length**  
  https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/Rolling%20Cognitive%20State%20Maintenance_%20Why%20Structured%20Carry-Forward%20May%20Matter%20More%20Than%20Raw%20Context%20Length.pdf

- **Engineering J-Space for Long-Context Recall**  
  https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/ENGINEERING%20J-SPACE%20FOR%20LONG-CONTEXT%20RECALL

- **Engineering the J-Space: Beyond Chain of Thought**  
  https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/ENGINEERING%20THE%20J-SPACE%3A%20BEYOND%20CHAIN%20OF%20THOUGHT

- **Structure Beats Scale**  
  https://github.com/mlangford75-lgtm/mlangford75-lgtm/blob/main/Structure_Beats_Scale_Langford_v2%20(2).pdf

These works form the research lineage behind Jack's approach to:

- adversarial cognition
- structured attention anchoring
- frontier state rebinding
- rolling cognitive state maintenance
- long-context operative recall
- J-space engineering
- counterfactual reflection
- deterministic verification
- selective reasoning
- local-model efficiency

Earlier systems and experiments described in these papers are part of Jack's research lineage.

They should not be read as a claim that every historical component is literally implemented unchanged in Jack 3.8.

---

# Observed Comparative Coding Evaluations

Jack 3.8 has been evaluated in controlled planning, execution, and review workflows against other local models.

The most informative comparisons have included dense models in a similar deployment class, including:

- Muse Glimmer 30B
- Gemma 4 31B

The objective was not simply "which model can write a Python function."

The evaluations examined:

- planning precision
- preservation of original requirements
- plan → execution fidelity
- malformed-input handling
- test quality
- independent review
- foreign-workflow reconstruction
- adversarial failure discovery
- state preservation across mode changes

In these observed workflows, Jack's clearest advantage was not raw syntax generation.

It was **control over the reasoning process**:

- reconstructing authoritative state
- finding weaknesses in another model's work
- inspecting the tests as evidence rather than merely accepting them
- preserving constraints into execution
- separating observation from assumption
- maintaining configuration boundaries
- returning to deep reasoning when review warranted it

These are controlled demonstrations, not proof that Jack universally outperforms the compared models on all tasks.

Community replication and broader evaluation are encouraged.

---

# Research Preview / Experimental Model

Jack 3.8 is a research-oriented model.

It is intended to produce real work, but the architecture is also an active research program.

Jack remains probabilistic.

It can:

- make reasoning errors
- generate false-positive audit hypotheses
- misinterpret ambiguous requirements
- preserve an incorrect conclusion if that conclusion was incorrectly established upstream
- fail to retrieve relevant history
- over- or under-calibrate uncertainty
- fail when an appropriate deterministic verifier is unavailable

The architecture is designed to reduce important classes of long-horizon failure.

It does not make them impossible.

---

# Recommended Runtime Behavior

For the intended Jack behavior:

- use the model's embedded chat template
- do not add a competing external system prompt unless intentionally testing interaction effects
- preserve the conversation history
- expose deterministic tools when available
- allow Jack to use those tools when exact verification is appropriate
- use native thinking selectively according to workflow requirements
- keep Jack XML active in both thinking-ON and thinking-OFF operation
- use a runtime capable of maintaining long context and tool history

For LM Studio testing, use a fresh chat with the external **System Prompt empty** so the embedded Jack template is not duplicated or overridden.

---

# Thinking Mode Guidance

## Thinking ON

Best suited for:

- initial architecture
- novel planning
- difficult debugging
- ambiguous requirements
- independent review
- high-risk changes
- adversarial audit
- uncertainty
- hypothesis generation
- complex causal reasoning

Expected behavior:

> **Search broadly. Falsify aggressively. Calibrate before commitment.**

## Thinking OFF

Best suited for:

- execution of an established plan
- repetitive edits
- straightforward implementation
- routine continuation
- tool-oriented steps
- actions whose governing constraints are already known

Jack XML remains active.

Thinking OFF is not "unaligned base-model mode."

It is reduced native deliberation under the same Jack attention-anchoring and operative-state architecture.

---

# Usage

## LM Studio

1. Download:

```text
Jack-3.8-27B-Coder-16GB-VRAM.gguf
```

2. Load the model in LM Studio.

3. Start a fresh chat.

4. Leave the external **System Prompt blank**.

5. Select the desired context length based on available memory and runtime configuration.

6. Toggle native thinking according to the stage of the workflow when your runtime exposes that control.

---

## llama.cpp

Example local server:

```bash
llama-server \
  -m Jack-3.8-27B-Coder-16GB-VRAM.gguf \
  -c 32768
```

Increase context according to your hardware, KV-cache configuration, and runtime limits.

Example terminal use:

```bash
llama-cli \
  -m Jack-3.8-27B-Coder-16GB-VRAM.gguf \
  -p "Review this implementation against the original requirements."
```

---

# Large-Context Validation

Jack 3.8 has been exercised in large-context local testing, including a **120,000-token runtime context configuration**.

That figure describes a tested runtime setting, not a guarantee that every 120K-token workload will have identical quality or memory behavior across different runtimes, GPUs, KV-cache settings, or prompts.

The Jack research target is not merely maximum token count.

It is maintaining **operative state** as the historical context becomes large.

---

# Hardware Target

Jack 3.8 is built for the practical **16GB VRAM GPU class**.

The final model file is approximately in the 12GB-class range, leaving the remaining GPU-memory budget for runtime overhead and KV cache depending on context configuration.

Actual memory use depends on:

- runtime
- GPU backend
- context length
- KV-cache type
- offload settings
- batching
- flash-attention support
- tool/runtime integration

The model is intended for local consumer hardware rather than datacenter-only deployment.

---

# Quantization

The Jack 3.8 release uses a deliberately selected GGUF quantization chosen to preserve the native reasoning and coding capability needed by the Jack architecture while fitting the model into the practical 16GB VRAM class.

The exact quantization configuration is intentionally not disclosed.

Quantization is treated as part of the engineering of the release rather than as a packaging afterthought.

A weak quantization can destroy capabilities that no prompt architecture can recover.

Jack therefore depends on the interaction of:

> **model substrate + quantization quality + Jack cognitive architecture**

---

# What Jack Is Not

Jack 3.8 is not:

- a guarantee of perfect reasoning
- proof that a 27B model universally outperforms larger models
- a replacement for deterministic verification
- a claim that long context automatically equals memory
- a claim that native thinking should always be ON
- a claim that native thinking should always be OFF
- a claim that XML is merely a second chain of thought
- direct mechanistic proof that a particular XML tag controls a specific neural activation
- a claim that earlier Jack research components are all implemented unchanged in this release

Jack is an inference-time cognitive control architecture integrated with a capable local coding model.

---

# Model Identity

| Field | Value |
|---|---|
| Model | `Jack-3.8-27B-Coder-16GB-VRAM` |
| Author | Jonathan Michael Langford — Independent Researcher, Lead Architect, The Jack Project |
| Contact | `mlangford75@protonmail.com` |
| Project / Research | https://github.com/mlangford75-lgtm/mlangford75-lgtm |
| GGUF | `Jack-3.8-27B-Coder-16GB-VRAM.gguf` |
| Model class | 27B Qwen3.8-derived |
| Primary specialization | Agentic coding and long-context software engineering |
| Format | GGUF |
| Target hardware | 16GB VRAM consumer GPU class |
| Native reasoning | Preserved |
| Adaptive thinking | Supported by workflow/runtime when exposed |
| Jack XML | Active structured attention/state architecture |
| Deterministic tools | Supported when exposed by runtime |
| External system prompt | Leave blank for intended embedded-template behavior |
| Validated local runtime | LM Studio Chat |
| Large-context test setting | 120,000 tokens |
| Exact quantization | Intentionally undisclosed |
| Vision | Text-only release |

---

# Core Design Principle

> **The model is not being asked to become a different intelligence. It is being given an architecture for keeping the right intelligence operative at the right time.**

For short isolated tasks, native model capability may be enough.

For long-running agentic coding work, intelligence must survive:

- changing state
- repeated tool calls
- superseded artifacts
- corrections
- uncertainty
- counterfactual branches
- review
- execution
- final commitment

Jack is designed around that problem.

---

# One-Line Description

> **Jack-3.8-27B-Coder-16GB-VRAM is a 27B Qwen3.8-derived local coding model with the Jack long-context cognitive architecture: structured attention anchoring, frontier state rebinding, operative recall, adversarial falsification, configuration separation, deterministic verification, and adaptive native reasoning for agentic software engineering on 16GB GPUs.**

---

# Research Direction

Jack asks a practical question:

> **How much effective agentic intelligence can be recovered from local hardware by organizing cognition better rather than relying only on more parameters, more context, or permanently enabled deliberation?**

That is the research program.

Special Acknowledgement to https://empero.org