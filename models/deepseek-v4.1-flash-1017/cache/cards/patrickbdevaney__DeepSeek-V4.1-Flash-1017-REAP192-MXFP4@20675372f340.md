---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags: [deepseek, moe, reap, expert-pruning, mxfp4]
---

# DeepSeek-V4.1-Flash-1017-REAP192-MXFP4

This is deepseek-ai/DeepSeek-V4.1-Flash-1017 with 192 of 384 routed experts kept in every decoder layer. The kept experts are **bit-exact**: their packed E2M1 weights and E8M0 scales are copied byte for byte from the base checkpoint at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.

The following parts are unchanged:
- the dense path (FP8 attention, compressor and indexer);
- the shared expert, the hyper-connections, the vision tower and aligner;
- the three DSpark/MTP layers.

The router's `gate.weight`, `gate.bias` and `gate.bias_vl` are sliced to the kept experts. Routing therefore re-normalises over the survivors exactly as the base model's router does.

This repository is the master for every derived build (NVFP4, GGUF / Thor).

## How the experts were chosen

- **Calibration corpus.** 4975198 tokens in 8k sequences, plus a long-context pool of 32k-128k rows, token-balanced across general, multilingual, agentic, code, vision, math, science, finance and long-context text (corpus spec `c1803e70af7db162`). It was run through the teacher layer by layer with Engram switched on. Only human-written or open-model text was used; no Claude-written text.
- **Ranking.** Candidate keep-sets were ranked by HOPE (pairwise co-activation with the REAP-sum diagonal), REAP-sum, routing mass, frequency and random anchors.
  - Each pool, and image tokens ranked separately on their own routing (`bias_vl`), protected its top experts at the measured protect-fraction knee (0.06).
  - Super-Experts and position-sensitive / attention-sink experts were never pruned (long-context drift guard).
- **Arbitration.** The candidates were arbitrated by held-out top-32 KL against the teacher. The router re-selected over the survivors, evaluated in memory, on a screening split.
  - Winner: `hope_sum_pf0`.
  - Keep-set fingerprint: `1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec` (also in `reap_keepset.json`).

**Not healed.** Pruning inflates the routed sum, because weights re-normalise over fewer experts. A measured per-layer gain was measured but not adopted (it did not pass its paired gate), so every gain is 1. An E8M0 scale can only move by powers of two, so this master ships **without** that gain. The NVFP4 and GGUF builds carry it.

## How it was evaluated

Every number below comes from `s10/final/results.json` of the build pipeline. It is a paired, teacher-forced comparison against the unpruned deepseek-ai/DeepSeek-V4.1-Flash-1017 at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, run in its own MXFP4/FP8 numerics. The comparison uses 185565 scored tokens from 26 held-out documents in the confirmation split. That split was never used to choose the model. The intervals are ninety-five percent document-cluster bootstrap intervals.

- **kl32** is the mean over tokens of the KL divergence on the teacher's top-32 support.
- **top-1 agreement** is the share of tokens where this model's argmax equals the teacher's.
- **dNLL** is this model's next-token NLL minus the teacher's.

The pass/fail thresholds were pre-registered before any result existed (`ops/s10_gates.json`, sha256 `204259f4c401037a`). Any threshold amended after a result is listed below with its date, and the result under the original rule is shown beside it.

| arm | kl32 (95% CI) | top-1 agreement | dNLL | confident-flip rate |
|---|---|---|---|---|
| control | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 |
| mxfp4_A | 0.4405 [0.2639, 0.6465] | 0.7980 [0.7431, 0.8481] | 0.2718 [0.1219, 0.4585] | 0.0524 |
| thorA | 0.5194 [0.3261, 0.7423] | 0.7756 [0.7179, 0.8284] | 0.3091 [0.1426, 0.5119] | 0.0619 |
| b128 | 0.8026 [0.5627, 1.0801] | 0.7092 [0.6443, 0.7701] | 0.5469 [0.3271, 0.8010] | 0.0995 |

**Known limitation: multilingual.** Top-1 agreement with the unpruned model on the held-out multilingual documents is lower than in every other domain (confirmation split: mxfp4_A 0.5247, thorA 0.4935, b128 0.4219). The release floor for every other domain is 0.65; for multilingual it was lowered to a collapse guard of 0.4 (2026-10-22, owner decision, before any s10 result). During expert selection, the shipped keep-set scored 0.5233 on the screen split against 0.5592-0.5741 for random keep-sets of the same size. Keep-sets that protected more multilingual experts recovered multilingual only by losing general, finance and science text, so the better all-round keep-set ships. For multilingual use, prefer the unpruned model.

**Gate amended after the result (2026-10-24).** The pre-registered rule required the pruned model to beat a random keep-set of the same size on kl32. Over all domains it did not reach significance (paired difference -0.1014, interval -0.2196 to +0.0400), so this release FAILED its gate as originally registered. The cause is multilingual alone. After seeing that result, the owner amended the rule to exclude multilingual, whose weakness is already documented above. Without it the difference is -0.2144 (interval -0.2633 to -0.1676). Reason: Owner decision AFTER the first s10 final result (confirm split). As registered, mxfp4_A vs random0 paired d_kl32 was -0.1014 [-0.2196, +0.0400]: not significant, entirely because of the multilingual bucket (kl32 1.534 vs random 0.894), the weakness the owner had already accepted on 2026-10-22 with a lowered top-1 floor. In every other bucket the keep-set has roughly half of random's kl32. The as-registered comparison stays in results.json and on the cards beside the amended one; no other threshold changes (thorA top-1 0.78 floor still applies).

`control` keeps all experts and is re-run through the same evaluation machinery as the others. It measures that machinery, not the model.


Here is this model's own breakdown (`mxfp4_A`):

| domain | tokens | kl32 | top-1 agreement | dNLL |
|---|---|---|---|---|
| agentic | 40955 | 0.1889 [0.1645, 0.2120] | 0.8589 [0.8514, 0.8670] | 0.0809 [0.0511, 0.1053] |
| code | 24573 | 0.0462 [0.0310, 0.0709] | 0.9739 [0.9653, 0.9795] | 0.0336 [0.0220, 0.0508] |
| finance | 24573 | 0.3666 [0.1989, 0.5876] | 0.7775 [0.7241, 0.8147] | 0.2240 [0.1065, 0.3626] |
| general | 32764 | 0.4440 [0.3468, 0.5679] | 0.7817 [0.7353, 0.8206] | 0.1493 [-0.0422, 0.3407] |
| math | 8191 | 0.1248 [0.1248, 0.1248] | 0.9390 [0.9390, 0.9390] | 0.0722 [0.0722, 0.0722] |
| multilingual | 24573 | 1.5337 [1.1905, 1.8014] | 0.5247 [0.4667, 0.5842] | 1.2209 [0.8392, 1.5565] |
| science | 24573 | 0.3517 [0.2395, 0.5378] | 0.7724 [0.7342, 0.8130] | 0.2461 [0.1567, 0.3929] |
| vision | 5363 | 0.3661 [0.3480, 0.3904] | 0.8745 [0.8649, 0.8842] | -0.1394 [-0.2389, -0.0489] |

| position | tokens | kl32 | top-1 agreement | dNLL |
|---|---|---|---|---|
| [0,8) (under-sampled) | 180 | 1.4050 [1.0492, 1.7911] | 0.4333 [0.3722, 0.4934] | 0.4165 [-0.0821, 0.9147] |
| [8,64) (under-sampled) | 1232 | 0.7652 [0.5039, 1.0748] | 0.6607 [0.5885, 0.7240] | 0.4124 [0.1856, 0.7043] |
| [64,256) | 4307 | 0.5781 [0.3934, 0.8118] | 0.7407 [0.6751, 0.7959] | 0.3434 [0.1742, 0.5495] |
| [256,1k) | 17145 | 0.4713 [0.3119, 0.6586] | 0.7713 [0.7188, 0.8206] | 0.3060 [0.1557, 0.4918] |
| [1k,2k) | 22935 | 0.4624 [0.2859, 0.6795] | 0.7866 [0.7284, 0.8396] | 0.3001 [0.1385, 0.5008] |
| [2k,4k) | 46798 | 0.3928 [0.2199, 0.6097] | 0.8126 [0.7570, 0.8624] | 0.2333 [0.0828, 0.4262] |
| [4k,8k) | 92968 | 0.4409 [0.2472, 0.6627] | 0.8035 [0.7457, 0.8558] | 0.2724 [0.1170, 0.4591] |

- **Vision caption perplexity:** 7.349, against 8.012 for the teacher, a ratio of 0.917 over 5256 caption tokens.
- **Release gate:** PASS under the gate as amended 2026-10-24 (FAIL as originally registered; see the evaluation).

## Speculative decoding

Not measured yet. The checkpoint's own DSpark head (the three `mtp` layers) is being refit to this keep-set; this section is filled from that result.


## Engram tables (required, not included)

The two Engram tables and their small projection tensors are unchanged by pruning. They are not re-hosted here. Download these two files from the base repository at the pinned revision into this model's directory:

- `model-00047-of-00048.safetensors` (101.5 GB, sha256 `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5`)
- `model-00048-of-00048.safetensors` (101.5 GB, sha256 `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd`)

```
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir .
```

`model.safetensors.index.json` already maps their tensors to these file names. The same information is in `engram_pointer.json`.

## License and provenance

- **License:** MIT, the same as the base model (`LICENSE` is copied verbatim).
- **Teacher:** deepseek-ai/DeepSeek-V4.1-Flash-1017 at `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- **Corpus spec:** `c1803e70af7db162`.
- **Keep-set:** `1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec`.
- **Pre-registered gates:** `204259f4c401037ae381ce93b3aa3372a13cf8783b034cc5626761e5c1e7addc`.
