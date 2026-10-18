---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
language: [en, zh]
tags: [exl3, quantized, glm, moe, rocm, strix-halo, gfx1151, uncensored]
---
<p align="center"><img src="assets/banner.png" alt="Yamz Labs" width="100%"></p>

# GLM-5.3-Flash, EXL3, uncensored

**Model size:** 320B parameters (18B active). The "50B params" figure in the Hugging Face sidebar is not a parameter count: the site counts the packed 16-bit words of the EXL3 format.

**Hardware:** sized to fit one 128 GB machine. The pack is in the standard EXL3 format, so it is not tied to one GPU; it has been tested only on AMD Strix Halo (Ryzen AI Max+ 395) with the Kyojin engine, and other hardware is untested.

This is the same model as [yamz-labs/GLM-5.3-Flash-EXL3-Yamz](https://huggingface.co/yamz-labs/GLM-5.3-Flash-EXL3-Yamz), plus two small files that make it refuse far less. The weight shards are byte-identical to the base repository (same sha256, see `SHA256SUMS`). Read that card for the quantisation, the hardware, the quality against FP8 and the speed table. This card covers only what the two extra files change.

The model refuses far less and may produce content the base model would decline. See "Use" below.

## Highlights
- **Refusals.** 0/100 with the edit, 81/100 without, on 100 harmful prompts, measured on the base pack.
- **Same quality, size and speed as the base repository.** The shards are byte-identical: KLD 0.15081 against FP8 on 30 held-out rows, 99.73 GB, prefill 652 tok/s at 8K and 567 at 500K, decode 27.4 to 36.8 tok/s with Kyojin 1.5.10 on one Ryzen AI Max+ 395 (see the base card).
- **MTP projection included.** The pack ships `mtp_eh_proj.st` (64 MB), an unquantized copy of the MTP projection. The Kyojin server loads it automatically from the model folder (engine main from 2026-10-05, no flag). It only changes the draft, not the model's answers: 31.6 tok/s greedy with it against 26.7 without, on three prompts of 400 tokens.
- **Cost of the edit.** Under 1 % of decode by our best estimate (see "Speed cost").
- **Switchable.** One environment variable or flag turns it off; deleting two files gives the base repository.

![Refusals on 100 harmful prompts, with and without the edit](assets/datasheet-refusals.png)

## What is added
| File | Size | What it is |
|---|---|---|
| `uncensor_spec.json` | 3 KB | Per-layer strengths of one direction edit (attention output of layers 24-44, strength 2.14-2.21; MLP output of layers 11-38, strength 0.90-1.49). |
| `uncensor_direction.st` | 16 KB | The direction: one unit vector, 4096 float32 values (safetensors layout; the extension is not `.safetensors` on purpose, so that weight loaders do not index it). |

The weights are not edited. The Yamz engine finds `uncensor_spec.json` in the model folder at load time and subtracts the direction from the output of each affected block while the model runs. The direction is the difference between the mean residuals on 400 harmful and 400 harmless prompts, with the harmless-mean component removed. The log prints ` -- ablit runtime: spec <path> active (bundled in the model directory)` at load. No such line means no edit.

Switch it off without touching the files: `EXL3_ABLIT_RUNTIME=off` or `--no-uncensor` (then the model behaves as the base repository). `EXL3_ABLIT_RUNTIME=/path/to/other_spec.json` overrides the bundled file. Delete the two files to get the base repository.

The uncensor preset is applied by the Kyojin engine at load (https://github.com/Yamz-Labs/kyojin). Stock ExLlamaV3 and TabbyAPI load the weights but ignore it, so you get the base model with its refusals. Older engine builds ignore the file too.

## Run
```bash
git clone https://github.com/Yamz-Labs/kyojin && cd kyojin
./build.sh && source tools/strix_halo/env.sh
python tools/glm/serve.py --model /path/to/this-folder --port 8000 -c 131072 --num-draft 2
```
Same flags as the base repository. Sampling: temperature 1.0, top-p 0.95, no repetition penalty. Greedy decoding is not recommended.

The pack also ships `draft2s.st`, a two-stage draft head. Kyojin 1.5.2 and later pick it up from the model folder automatically; it only speeds up drafting, never changes the answers. `EXL3_DRAFT2S=off` turns it off.

## Measured effect
Spec sha256 `fe9989cb...f8d97`, direction sha256 `7ca504e1...0c608`, fitted on a smaller (2.05 bpw) quantisation of the same model. All numbers below were measured on the base pack, with the spec bundled in the model folder.

| What | Result |
|---|---|
| Bundled `uncensor_spec.json` found and applied from the model folder | yes: log line `active (bundled in the model directory)`, 34 hooked blocks |
| Refusals on 100 harmful prompts (regex detector, 64 greedy tokens) | 0/100 with the edit; 81/100 without (`EXL3_ABLIT_RUNTIME=off`) |

Small sample (100 prompts): the interval is wide. The refusal count is a regex detector; soft refusals and partial answers can slip through either way. English prompts only. A hash proves the file is the one we measured, not that your setup behaves the same.

## Speed cost of the edit
Controlled runs on one Ryzen AI Max+ 395, `-c 131072`, MTP 2 drafts, 3 repetitions, on a pack of the same size and quantised tensors. Decode prose / chat / code tok/s, prefill 3.5K / 14K tok/s.

| Edit | Medium-reasoning template | Decode | Prefill |
|---|---|---|---|
| off | no | 28.7 / 30.2 / 31.9 | 562 / 578 |
| on | no | 31.4 / 32.3 / 32.7 | 553 / 592 |
| off | yes | 27.7 / 31.4 / 29.6 | 591 / 593 |
| on | yes | 27.1 / 29.2 / 27.4 | 579 / 592 |

"Medium-reasoning template" = the chat template with medium reasoning effort and `--max-history 2` (not the default of this card). The edit alone is inside the noise: the three repetitions inside one arm differ by up to 20 %, and the edit-on row came out faster than the edit-off row. Our best estimate is that the edit costs less than 1 % of decode (a microbenchmark measured -0.1 to -0.3 %) and nothing visible in prefill. The slowdown in the last row comes from that template and the history limit, not from the edit.

## Use
The edit lowers the rate of refusals. It is not a safety evaluation and says nothing about what the model will or will not produce. You decide how to use it and you answer for it under the laws that apply to you. Intended uses: research, evaluation, red-teaming and private deployment. Do not use it to produce illegal content, to target people, or in a public service without your own filtering and moderation. The files come as is, without warranty.

## Limits
- Only gfx1151 with ROCm is tested, with the Yamz engine only.
- The edit is one global direction. It lowers refusal and over-refusal together.
- We measured refusals only on this pack. Over-refusal on safe prompts (XSTest), the KL of the edit and task-score deltas were not measured on v2.
- Greedy decoding is not recommended.

## Licence and credits
MIT. Base model GLM-5.3-Flash, Copyright (c) 2026 Z.AI Co., Ltd, MIT, licence file included as `LICENSE`. Our additions: Copyright (c) 2026 Yamz Labs, MIT. Provided as is, without warranty. Engine: built on ExLlamaV3 by turboderp (MIT). Quantised tensors by turboderp; layer mix, scale-vector tuning and edit by Yamz. Not affiliated with Z.ai. We remove this repository on request from the base-model authors or a rights holder, on a valid legal notice.
