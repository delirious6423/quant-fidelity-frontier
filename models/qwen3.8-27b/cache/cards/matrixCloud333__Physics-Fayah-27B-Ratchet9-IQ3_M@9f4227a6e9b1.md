---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language:
- en
- zh
pipeline_tag: text-generation
library_name: gguf
tags:
- physics
- scientific-computing
- numerical-simulation
- continual-learning
- gguf
- iq3_m
---

# Physics-Fayah 27B — sequence-16 calibration refresh (IQ3_M)

An experimental, locally adapted GGUF checkpoint intended for scientific programming and computational-physics exploration. It is a derivative checkpoint, not a validated numerical solver or a substitute for independent derivation, tests, and convergence analysis.

## Artifact and provenance

- File: `Physics-Fayah-27B-Ratchet16-IQ3_M.gguf`
- Quantization: IQ3_M
- Size: 14,532,048,416 bytes
- SHA-256: `34d0bfaf614158d4936ffc28a1c01009a187c8351f8fd12c3f463f76294036bd`
- Calibration date: 2026-10-20
- Calibration sequence: 16

The local source file retained the old `Ratchet9` filename. Its pre-update backup has SHA-256 `65c9fec12c02a153c7fbde788171361d6269b4a902d79c602ba447274cb6b329`, matching the earlier public Hub artifact. The October 20 calibration report records sequence 16, and the Fayah-specific calibration log records a directional update. The Hub filename is updated to reflect that lineage.

## Calibration evidence and limits

The sequence-16 probe used three equivalent prompts. Its recorded values were:

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| Learning loss | 0.2147887469 | 0.2100177841 |
| Retention loss | 0.2489672313 | 0.2496686303 |
| MTP draft acceptance | 0.7482014388 | 0.7481481481 |

The candidate output-equivalence check passed, but its MTP gate did **not** pass; the behavioral gate was not evaluated. The seed was admitted using an explicit learned-seed override. These small-sample measurements do not establish a general capability or acceptance improvement. This release makes no throughput, perplexity, or physical-model-validation claim. Earlier Ratchet9 performance figures were not re-measured for this exact weight hash.

This sequence-16 calibration record is not the result of the currently running ten-model physical-model campaign, and does not mean those candidate models passed review. The campaign’s validation results must be reported separately from this weight-probe record.

## Use

Use a current llama.cpp build with GGUF support. Context length, memory requirements, and generation speed depend on runtime, hardware, KV-cache policy, and speculative-decoding configuration; this repository does not guarantee a particular throughput or quality level.

Treat generated equations, code, and scientific explanations as hypotheses. Independently verify equations, units, boundary conditions, numerical convergence, and invariants before relying on results.

## License

Apache-2.0, consistent with the Qwen3.8 base model license and the prior project declaration. Review the upstream model and any merged-source terms when redistributing derivative artifacts.
