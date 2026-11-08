---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- gguf
- research
- biology
- coding
- arcas-dynamics
- reasoning
- persephone
---

Persephone is Arcas Dynamics' research and coding model, built to bring frontier-style reasoning to the hardware people already own: a single 16 GB graphics card, or a 16 GB Apple Silicon Mac.

Persephone is post-trained by Arcas Dynamics from the open-weight Qwen3.8-27B base, through our own multi-stage pipeline: reasoning distillation, preference optimisation and reinforcement learning against executed tests. It is the default local model in AXON, Arcas Dynamics' desktop AI workspace.

Reasoning-first: thinks through algorithms, edge cases and time limits before it answers.
Trained against reality: Persephone's RL rounds reward code that actually passes hidden tests, not code that only looks right.
Built for your machine: 3-bit builds with Arcas' own importance matrix, about 11 GB, fast on a single consumer GPU.
Better with AXON: AXON runs Persephone's code against the problem's examples and its own self-written tests, and sends it back to fix what fails.
Files
File	Size	Best for
Persephone-27B-v6.1-UD-IQ3_S.gguf	11.2 GiB	16 GB NVIDIA GPUs (tested on RTX 4080 SUPER)
Persephone-27B-v6.1-UD-IQ3_XXS.gguf	10.2 GiB	16 GB Macs (M3 and later)
How Persephone is made
Arcas Dynamics trains Persephone in stages:

Reasoning distillation: supervised training on full-length reasoning traces for coding and research problems.

Preference optimisation: learning to prefer correct answers over plausible wrong ones.

Reinforcement learning with verifiable rewards: Persephone writes thousands of solutions to programming problems; every one is executed against hidden tests; correct answers are reinforced over wrong and overlong ones. Each round must beat the previous version on held-out problems before it ships.

Quantization for consumer hardware: an importance matrix computed on Persephone's own outputs, with per-tensor precision chosen to keep the most sensitive weights sharp.

Training updates are applied as low-rank adapters merged into the base weights. Coding problems are drawn from LiveCodeBench (AtCoder, LeetCode); evaluation problems are held out from all training.

Results
Held-out programming problems, UD-IQ3_S build, 16K-token answer limit, temperature 1.0.

Problem set	Persephone	Persephone in AXON (verify-and-retry)
Easy (50 problems x 3)	93.3% (140/150)	99.3% (146/147)
Medium (80)	80.0% (64/80)	86.3% (69/80)
Hard (70)	38.6% (27/70)	40.0% (28/70)
These are Arcas Dynamics' own held-out evaluation sets, not official leaderboard results. On the easy set, the unmodified Qwen3.8-27B base scored 147/150 under the same conditions; a paired comparison on the medium and hard sets is planned but has not been run yet.

Quick start
With llama.cpp:


llama-server -m Persephone-27B-v6.1-UD-IQ3_S.gguf -ngl all -fa on -c 32768 --jinja
Recommended sampling: temperature 0.6, top_p 0.95. Thinking is on by default.

On a 16 GB Mac, use the IQ3_XXS build, raise the GPU memory limit with sudo sysctl iogpu.wired_limit_mb=12288, and use a context of 8192.

Or skip setup entirely: Persephone comes ready to run in AXON.

Limitations
Multi-step agent and tool-use work is the focus of Persephone's next training round and is not yet specialised.
The hardest problems often need more than 16K tokens of reasoning; allow a longer limit for them.
3-bit builds trade a little accuracy for the ability to run on consumer hardware.
Persephone can be confidently wrong. Check code and claims that matter.
License and attribution
Persephone 27B is released under the Apache License 2.0.

Persephone is built on Qwen3.8-27B by the Qwen team, Alibaba Cloud, used under the Apache License 2.0. We thank the Qwen team for releasing a base model of this quality openly. See NOTICE for attribution details.

Persephone, AXON and the Persephone training pipeline are developed by Arcas Dynamics.