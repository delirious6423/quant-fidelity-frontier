---

license: mit
language:
- multilingual
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
tags:
- large-language-model
- reasoning
- mixture-of-experts
- agentic-coding
- code-generation
- tool-use
- long-context
---

# Ornith-1.5-35B-A3B

Ornith-1.5-35B-A3B is a mixture-of-experts reasoning model from the Ornith family, designed for demanding software-engineering and agentic workloads. With approximately 35 billion total parameters and around 3 billion active parameters per token, the model combines a large parameter pool with a comparatively small active computation footprint.

This repository provides GGUF quantized variants of Ornith-1.5-35B-A3B for efficient local inference. The available quantization formats substantially reduce storage and memory requirements while retaining the model's reasoning, coding, and agent-oriented capabilities.

Ornith-1.5-35B-A3B is intended for multi-step development workflows where the model may need to understand existing code, reason through problems, generate or modify implementation details, and interact with tools through a compatible inference or agent framework.

---

# Model Overview

- **Model Name:** Ornith-1.5-35B-A3B
- **Base Model:** ornith-ai/Ornith-1.5-35B-A3B
- **Architecture:** Mixture-of-Experts language model
- **Parameter Count:** Approximately 35 Billion total parameters
- **Active Parameters:** Approximately 3 Billion parameters per token
- **Model Type:** Reasoning
- **Modalities:** Text
- **Context Length:** 262,144 tokens
- **Developer:** Ornith AI
- **License:** MIT

---

# Quantization Formats

This repository provides multiple GGUF quantized versions of Ornith-1.5-35B-A3B. Each format offers a different trade-off between model storage, memory consumption, inference requirements, and preservation of the original model's behavior.

## IQ3_M

- Size reduction of approx 79.53% (14.72 GB) compared to 16-bit (71.90 GB)
- Aggressive 3-bit quantization intended for substantially lower memory requirements
- Suitable for local agentic coding and reasoning workloads where deployment resources are constrained
- Makes the large 35B-parameter MoE model considerably more practical for local inference
- Fine-grained reasoning, complex code generation, and long multi-step workflows may experience greater degradation than higher-precision variants

## IQ4_NL

- Size reduction of approx 73.55% (19.02 GB) compared to 16-bit (71.90 GB)
- 4-bit non-linear quantization providing a stronger balance between model footprint and output fidelity
- Suitable for coding agents, software-engineering tasks, repository analysis, and reasoning workloads
- Retains more representational capacity than the more aggressive IQ3_M format while remaining substantially smaller than the F16 model
- Appropriate for deployments where memory efficiency and reasoning quality both need to be considered

## IQ4_XS

- Size reduction of approx 74.88% (18.06 GB) compared to 16-bit (71.90 GB)
- Compact 4-bit quantization designed for efficient local deployment of the MoE reasoning model
- Provides a practical compromise between memory consumption and retention of coding and reasoning behavior
- Suitable for software-development assistants, agentic workflows, and multi-step problem solving
- Offers a substantially smaller deployment footprint than the original 16-bit representation

---

# Training Background (Original Model)

Ornith-1.5-35B-A3B belongs to the Ornith model family focused on reasoning and agentic software engineering. Its mixture-of-experts architecture separates total model capacity from the amount of computation activated for each token, allowing the model to maintain a large parameter pool while using only a subset of parameters during individual forward passes.

## Mixture-of-Experts Architecture

- Uses a sparse MoE architecture with approximately 35B total parameters
- Approximately 3B parameters are active for each token
- The sparse activation pattern reduces per-token computational requirements relative to a dense model containing the same total parameter count
- Provides a large parameter capacity for reasoning and coding while maintaining a comparatively smaller active computation footprint


---

# Key Capabilities

- **Agentic Coding**
  Designed for software-engineering workflows involving iterative reasoning, implementation, debugging, and interaction with development tools.

- **Code Generation**
  Generates implementation code for software-development tasks while reasoning about the surrounding problem context.

- **Code Modification**
  Can be used in workflows that require modifying existing implementations rather than generating standalone snippets.

- **Repository-Level Reasoning**
  Supports analysis of larger software projects and tasks involving relationships between multiple source files.

- **Complex Reasoning**
  Designed for multi-step problems where solving the task requires planning, intermediate reasoning, and iterative refinement.

- **Tool Use**
  Can be integrated with compatible agentic runtimes that provide external tools, APIs, terminals, or development environments.

- **Mixture-of-Experts Efficiency**
  The 35B total parameter capacity is combined with approximately 3B active parameters per token, providing sparse computation during inference.

- **Long-Context Workflows**
  Intended for extended interactions and software-engineering tasks where substantial contextual information may need to remain available.

---

# Recommended Usecases

- **Agentic Coding Systems**
  Build coding agents capable of reasoning through software-development tasks and interacting with external development tools.

- **Repository-Level Software Engineering**
  Analyze and work with larger codebases containing multiple files, dependencies, and implementation components.

- **Automated Debugging**
  Assist with investigating software problems, identifying likely causes, and developing implementation-level fixes.

- **Code Generation and Refactoring**
  Generate new implementations or assist with restructuring and improving existing software components.

- **Terminal-Based Coding Agents**
  Integrate the model with terminal environments and agent frameworks for execution-oriented software development.

- **Developer Productivity**
  Automate repetitive development tasks such as code generation, modification, analysis, and debugging assistance.

- **Tool-Calling Applications**
  Connect the model to external functions, APIs, development tools, or agent orchestration frameworks.

- **Research and Experimentation**
  Evaluate sparse MoE reasoning models, quantization techniques, local inference, and agentic coding workflows.

---

# Usage Example

## Using llama.cpp

```bash
./llama-cli \
  -m SandLogicTechnologies/Ornith-1.5-35B-IQ3_M.gguf \
  -mmproj SandLogicTechnologies/mmproj-Ornith-1.5-35B-A3B-bf16.gguf
```


# Acknowledgments

These GGUF quantized models are based on the original work of the **Ornith AI** team.

Special thanks to:

- The Ornith AI team for developing and releasing the Ornith-1.5-35B-A3B model.

- The developers and contributors of **llama.cpp** and the broader GGUF ecosystem for enabling efficient local inference and quantized model deployment.

---

# Contact

For questions, feedback, or support, please refer to the official Ornith AI project and model repository.
