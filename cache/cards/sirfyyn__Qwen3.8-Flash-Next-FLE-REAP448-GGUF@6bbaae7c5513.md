---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
quantized_by: sirfyyn
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - moe
  - llama.cpp
  - fle
  - reap
  - expert-pruning
  - kl-divergence
---

# Qwen3.8-Flash-Next, FLE-IQ3_M-REAP448 (GGUF): 448 of 512 experts per layer, 82.00 GB

FLE-IQ3_M-REAP448 is FLE-IQ3_M with 64 of the 512 routed experts per layer removed, ranked by REAP saliency: 82.00 GB
instead of 88.87 GB and 161.84 B instead of 176.94 B parameters. Nothing is re-quantized; the kept experts and their
router rows are copied byte for byte from FLE-IQ3_M. It is a standard GGUF file with 448 instead of 512 experts per
layer and loads in upstream llama.cpp; the speed numbers below and the MTP head need the FLE engine
(<https://github.com/sirfyyn/FastLocalExperts>).

This file has its own repo so that the Hub lists it on its own. The unpruned FLE hybrids (FLE-IQ3_M, FLE-IQ3_XXS,
FLE-IQ2_M), the MTP head with its draft vocabulary and the eight short films are in
[sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF); how FLE-IQ3_M was
built from Unsloth's and ISTA-DASLab's public quantizations is described there. How the files and the engine were built
and measured is in the report: <https://sirfyyn.github.io/FastLocalExperts/> (also as
[PDF](https://sirfyyn.github.io/FastLocalExperts/fle-paper.pdf)).

| | FLE-IQ3_M-REAP448 | FLE-IQ3_M |
|---|---|---|
| Size | 82.00 GB | 88.87 GB |
| Parameters | 161.84 B | 176.94 B |
| Routed experts per layer | 448 | 512 |
| KLD against BF16, texts E / W / K | 0.080 / 0.086 / 0.185 | 0.051 / 0.050 / 0.129 |
| MKQA, correct answers German / English | 21.5 % / 29.8 % | 23.2 % / 33.0 % |
| Decode on FLE, against FLE-IQ3_M | +0.7 % [−0.3 .. +1.8] (bracket B3e6) | reference |

## Files

| File | Size | SHA-256 |
|---|---|---|
| `Qwen3.8-Flash-Next-FLE-IQ3_M-REAP448-00001-of-00002.gguf` | 44.91 GB (44,914,807,136 B) | `b0d002c1c68896618ce3b481a811f3245d152a276cec947204c7f0590bed9516` |
| `Qwen3.8-Flash-Next-FLE-IQ3_M-REAP448-00002-of-00002.gguf` | 37.09 GB (37,087,295,680 B) | `27f52c8db7725da78c0b03bbc4e90ac5b4bb3cd1419f97e2d74e6e5203f5cbec` |

Give llama.cpp or FLE the first part and it loads the second. The same list is in `SHA256SUMS`. Until 2026-10-10 the
same two files (same SHA-256) were in the folder `IQ3_M-REAP448/` of the main repo.

For speculative decoding with FLE, take the MTP head and its draft vocabulary from the main repo
(`mtp/mtp-Qwen3.8-Flash-Next-FLE-IQ4_NL.gguf`, `mtp/fle-mtp-vocab-32k.txt`); the head has its own experts and is the
same for all FLE files.

## How the pruning works

`llama-fle-reap` (in the engine repo) runs FLE-IQ3_M over a calibration text (256 sections of 2,048 tokens) and records,
per layer and expert, how often it was routed and its REAP saliency, the mean router-weighted norm of the expert's
output over the tokens routed to it ([Lasby et al., arXiv:2510.13999](https://arxiv.org/abs/2510.13999)). The
calibration text is FLE-IQ3_M's own answers to public prompts (German part of OASST, SWE-Gym/SWE-Zero, UltraChat,
GSM8K), mixed as 45 % German conversation, 25 % code and tool calls, 18 % English conversation and 12 % math; neither
the KLD texts nor the probe questions are in it. `reap_prune.py` keeps the 448 experts per layer with the highest
saliency, in their original order, and copies their weights and router rows byte for byte; nothing is re-quantized. On
the calibration text the kept experts receive on average 96.1 % of the summed router weight (lowest layer 90.5 %).

Bits per weight, measured from the GGUF header on 2026-10-09: 4.05 in total, routed experts 3.62, per-layer token
embedding table 4.50, attention, shared expert and norms 9.00, embeddings and output 7.54 (FLE-IQ3_M: 4.02, 3.62, 4.50,
9.05, 7.54).

## Quality: KL divergence against BF16

Mean KL divergence (KLD) of the next-token distribution against BF16, with top-1 agreement, measured with **upstream**
`llama-perplexity` (llama.cpp commit `a8c9a4e7c`, no FLE code), `-c 2048 -b 2048 -ub 2048 -t 16 -ngl 99 -cmoe`, on
three public texts: **E** Python language reference, **W** four German Wikipedia articles, **K** seven files of the
German Kubernetes documentation (method, sources and the full table in the main repo's card). FLE-IQ3_M-REAP448 was
measured on 2026-10-08 with the same build, settings and BF16 reference as the others. Lower is better; compare within
a column only.

| Quant | Source | Size (GB) | KLD E | KLD W | KLD K | Top-1 E / W / K (%) |
|---|---|---|---|---|---|---|
| UD-IQ4_XS | Unsloth | 93.7 | 0.049 ± 0.003 | 0.046 ± 0.001 | 0.122 ± 0.004 | 96.8 / 91.7 / 88.6 |
| FLE-IQ3_M | main repo | 88.9 | 0.051 ± 0.003 | 0.050 ± 0.001 | 0.129 ± 0.004 | 96.9 / 91.5 / 89.1 |
| IQ3_S | ISTA GSQ-RCO | 83.6 | 0.069 ± 0.004 | 0.061 ± 0.002 | 0.183 ± 0.005 | 96.4 / 90.6 / 86.8 |
| **FLE-IQ3_M-REAP448** | this repo | **82.0** | **0.080 ± 0.005** | **0.086 ± 0.003** | **0.185 ± 0.006** | 96.0 / 89.7 / 87.2 |
| UD-IQ3_XXS | Unsloth | 82.0 | 0.091 ± 0.005 | 0.091 ± 0.003 | 0.229 ± 0.007 | 95.5 / 88.3 / 84.6 |
| FLE-IQ3_XXS | main repo | 79.5 | 0.090 ± 0.005 | 0.093 ± 0.002 | 0.227 ± 0.007 | 95.7 / 88.1 / 84.2 |

Against FLE-IQ3_M: 64 of the 512 experts per layer removed, 6.9 GB smaller (82.00 against 88.87 GB), +0.029 / +0.036 /
+0.056 KLD. At the size of UD-IQ3_XXS its KLD is lower on all three texts; ISTA IQ3_S, 1.6 GB larger, is lower on E and
W. KLD understates what pruning removes, see the knowledge probe below.

## Knowledge probe: MKQA

Removing experts can remove facts that three short texts never touch. So every file also answered 400 factual
questions from MKQA ([Longpre, Lu and Daiber, arXiv:2007.15207](https://arxiv.org/abs/2007.15207); open-domain export
`nthakur/mkqa-open-domain`, answerable questions only), the same 400 questions in German and in English, with the
system prompt asking for the answer only or exactly "I don't know". Sampling as on Qwen's card (temperature 1.0, top-p
0.95, top-k 20, min-p 0), thinking off, up to 256 tokens, one fixed seed per question for every file, so the comparison
is paired. Server: the production FLE build (Patch 51) with all experts on the CPU and without MTP; measured
2026-10-08.

| File | Size (GB) | Correct, German | Correct, English | Against FLE-IQ3_M, German / English (points, paired 95 %) |
|---|---|---|---|---|
| FLE-IQ3_M | 88.87 | 23.2 % | 33.0 % | – |
| FLE-IQ3_XXS | 79.48 | 21.5 % | 30.8 % | −1.8 [−4.2 .. +0.5] / −2.2 [−5.0 .. +0.2] |
| **FLE-IQ3_M-REAP448** | 82.00 | 21.5 % | 29.8 % | −1.8 [−4.5 .. +0.8] / −3.2 [−6.5 .. +0.0] |
| FLE-IQ3_M, REAP to 384 experts (not released) | 75.13 | 17.5 % | 21.2 % | −5.8 [−9.2 .. −2.5] / −11.8 [−16.0 .. −7.8] |
| FLE-IQ3_M, 384 experts by mean activation norm (not released) | 75.13 | 11.2 % | 15.5 % | −12.0 [−15.8 .. −8.5] / −17.5 [−21.8 .. −13.2] |

The absolute rates are low by design (many questions are US-specific, and the model abstains when unsure); compare the
rows with each other. FLE-IQ3_M-REAP448 loses about what FLE-IQ3_XXS loses (−1.8 / −3.2 against −1.8 / −2.2 points),
the English drop at the edge of its interval. Between the two, FLE-IQ3_XXS is 2.5 GB smaller and FLE-IQ3_M-REAP448 has
the lower KLD. With 384 experts per layer the loss was three to five times that of FLE-IQ3_XXS, and pruning by REAP
saliency clearly beat pruning by the mean activation norm (MAN, [Liu et al.,
arXiv:2606.15716](https://arxiv.org/abs/2606.15716)) at the same size; therefore the released file keeps 448 experts and
was built with REAP. A second probe (60 short messages per language that must repeat an invented, rare full name) was
passed in all messages. One model and one probe; we do not generalize this.

## Speed

Bracket B3e6 of 2026-10-09: FLE-IQ3_M as the reference, the final engine configuration (`p49-ring`), one GPU expert
cache budget of 17,035 MiB for both files, 24 paired requests (code, math, physics, prose; greedy, MTP on), reference
drift +1.64 %. FLE-IQ3_M-REAP448 decoded as fast as FLE-IQ3_M: **+0.7 % [−0.3 .. +1.8]** started cold and +0.5 %
[−0.6 .. +1.7] with the warm-start profile its cold run had written. Like all final speed numbers it was measured with
the internal re-trained MTP head c2; the published head is the untrained one (see the main repo's card). Hardware: AMD
Ryzen 9 9950X3D (16 cores), 120 GB DDR5, one NVIDIA RTX PRO 4500 Blackwell (32 GB), Linux.

## Running

Upstream llama.cpp, for example with the experts on the CPU (no MTP: in our test of 2026-10-05, upstream could not use
the separate head, see the main repo's card):

```bash
llama-server -m Qwen3.8-Flash-Next-FLE-IQ3_M-REAP448-00001-of-00002.gguf -cmoe -ngl 99 -c 32768 --jinja
```

FLE (release tag [`v1.0`](https://github.com/sirfyyn/FastLocalExperts/tree/v1.0)): the production command and the
meaning of every flag are in the main repo's card under "Running"; use it unchanged and give `-m` the first part of this
file. Memory with that command (measured on 2026-10-09 with the `v1.0` build, published head, 32k vocabulary):

| File | System RAM taken | of it pinned (`--load-mode none`) | n-gram table, memory-mapped | GPU memory | Load time |
|---|---|---|---|---|---|
| FLE-IQ3_M-REAP448 | 48.0 GiB | 46.1 GiB | 26.8 GiB | 31,367 MiB | 127 s |
| FLE-IQ3_M (for comparison) | 54.3 GiB | 52.2 GiB | 26.8 GiB | 31,315 MiB | 167 s |

## Limitations

- One model, one machine (a single 32 GB GPU, 120 GB RAM, Linux, CUDA).
- Quality is KLD on three texts and one 400-question knowledge probe per language. The task benchmarks of the main card
  (IFBench, GPQA Diamond, LiveCodeBench v6) were run with FLE-IQ3_M only, not with this file.
- The experts were ranked on one calibration mix (German and English conversation, code, math). Topics or languages
  that the mix covers poorly may have lost more than the probe shows; we have not measured that.
- The KLD numbers come from the neutral upstream meter; the FLE engine adds a small router bias towards cached experts
  (`FLE_MOE_CACHE_BIAS=0.001`) that is not included in them.
- No support promise. This is a personal project; issues and PRs on the FLE repo are welcome.

## Credits and licence

This file is a derivative of Qwen3.8-Flash-Next and of FLE-IQ3_M, and through it of Unsloth's UD-IQ4_XS and
ISTA-DASLab's GSQ-RCO quantizations; the full list of sources, revisions and licences is in `NOTICE.md` and in the main
repo's card.

- **Qwen team**: Qwen3.8-Flash-Next, Qwen Community License 1.0 (copy in the `LICENSE` file of this repo), Copyright (c)
  2026 Qwen. Architecture paper: [arXiv:2608.30320](https://arxiv.org/abs/2608.30320).
- **Unsloth** ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)) and
  **ISTA-DASLab** ([ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
  GSQ: Dadgarnia et al., [arXiv:2604.18556](https://arxiv.org/abs/2604.18556); RCO: Helcig and Alistarh,
  [arXiv:2605.00649](https://arxiv.org/abs/2605.00649)): the quantizations FLE-IQ3_M is spliced from.
- **Expert pruning**: REAP, Lasby et al., "REAP the Experts: Why Pruning Prevails for One-Shot MoE compression",
  [arXiv:2510.13999](https://arxiv.org/abs/2510.13999); the comparison criterion MAN, Liu et al.,
  [arXiv:2606.15716](https://arxiv.org/abs/2606.15716). Our implementation (`llama-fle-reap`, `reap_prune.py`) is in the
  engine repo.
- **llama.cpp / ggml and GGUF** (G. Gerganov et al., MIT): <https://github.com/ggml-org/llama.cpp>; the FLE engine is a
  fork of it.
- **Knowledge probe**: MKQA ([apple/ml-mkqa](https://github.com/apple/ml-mkqa), CC BY-SA 3.0), used through the
  open-domain export `nthakur/mkqa-open-domain`; the questions are not redistributed here. **KLD texts**: Python
  documentation (PSF license), German Wikipedia (CC BY-SA 4.0), Kubernetes documentation (CC BY 4.0).

Licence of this repo: the model file is a derivative of Qwen3.8-Flash-Next and is provided under the **Qwen Community
License 1.0** (`license: other`, `license_name: qwen-community-1.0`; the licence text is the `LICENSE` file of this
repo). In short, as we read it: use, copy, modify, distribute and fine-tune are allowed; the copyright and licence
notice must travel with copies; products with more than 100 million monthly active users or US$ 20 million monthly
revenue must display the model name; running a Model-as-a-Service or AI Work Assistant business needs a separate licence
from Qwen. This summary is not legal advice; the licence text decides.

Copyright notice required by the licence: "Copyright (c) 2026 Qwen". The `LICENSE` file of this repo is the unchanged
Qwen licence text and carries the notice, and `NOTICE.md` repeats it.

## Citation

Please cite the report. The engine release with the report is archived on Zenodo:
<https://doi.org/10.5281/zenodo.23272400>.

```bibtex
@misc{bitzer2026fle,
  author       = {Max Bitzer},
  title        = {{FLE}: Fast Local Experts. An 89 GB Mixture-of-Experts model at ~90 tokens/s on one 32 GB GPU},
  year         = {2026},
  month        = oct,
  doi          = {10.5281/zenodo.23272400},
  note         = {Technical report, version 1.0.2. Engine: https://github.com/sirfyyn/FastLocalExperts
                  (tag v1.0.1, same engine code as v1.0).
                  Model files: https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF and
                  https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF},
  howpublished = {\url{https://sirfyyn.github.io/FastLocalExperts/}}
}
```

## Authors

Max Bitzer (sirfyyn). The pruning tools, the measurements and this card were made together with Claude (Anthropic,
Claude Code).
