---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
- n-gram-embedding
- lookup-table-quantization
- trellis-quantization
- 2-bit
- ngq2
---

# Qwen3.8-Flash-Next-NGQ2: Qwen3.8-Flash-Next with a 2-bit n-gram table

A **complete checkpoint** of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next):
- every backbone tensor is copied **byte-for-byte** from the base model (bf16, unchanged);
- the PLE n-gram embedding table is replaced by its **NGQ2** 2-bit packed form (2.10 bits per value). The table holds
  51.2B parameters (320M rows × 160 dims).

It loads with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. No other download is needed.

Derived from Qwen3.8-Flash-Next by Qwen (Copyright (c) 2026 Qwen) under the Qwen Community License 1.0; see License
below.

| | original (bf16) | this repo |
|---|---|---|
| backbone (everything except the n-gram table) | 257.6 GB | 257.6 GB (byte-identical) |
| n-gram table | 102.4 GB (16 bpp) | **13.4 GB** (2.10 bpp) |
| **total checkpoint** | 360.0 GB | **271.0 GB** |

Served with vLLM ([PR #66576](https://github.com/vllm-project/vllm/pull/66576), under review) on 4 × H200, TP4, table on the GPU: weights 84.6 → 63.9 GiB per GPU and KV cache 2.03M → 3.15M tokens (+55%), at the same TPOT within 2.2% (one run each).

Paper: [N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models](https://arxiv.org/abs/2610.21810) ([arXiv:2610.21810](https://huggingface.co/papers/2610.21810)).

## Usage

Requirements: `transformers>=5.8` (built-in `qwen4_exp` code; tested with 5.17), `torch`, `safetensors`,
`huggingface_hub`, `accelerate`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "dnotitia/Qwen3.8-Flash-Next-NGQ2"
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16,
                                             device_map="auto", table_device="gpu")   # "gpu" | "cpu" | "disk"
tok = AutoTokenizer.from_pretrained(repo)
msgs = [{"role": "user", "content": "Name three famous works by Leo Tolstoy."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                              enable_thinking=False)["input_ids"]
out = model.generate(ids.to(model.device), max_new_tokens=64, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

- **`trust_remote_code=True` is required.** `modeling_ngq2.py` subclasses transformers' built-in
  `Qwen4ExpForConditionalGeneration` (unchanged). The 102 GB `nn.Embedding` is never allocated. The table is replaced by an
  `ngq.PackedTable` that decodes rows on lookup.
- **Without `trust_remote_code`,** transformers falls back to its built-in class, because `model_type` stays
  `qwen4_exp`. That class has no n-gram table weights, so the table would be left **uninitialized**: transformers only
  warns about missing `ngram_embedding` weights. Always pass `trust_remote_code=True`.
- **`table_device`** chooses where the packed bytes live. The decode is identical in every placement, and the logits are
  bit-identical (verified below).
  - `"gpu"` (default): HBM.
  - `"cpu"`: host DRAM. Only the looked-up rows go to the GPU.
  - `"disk"`: page-aligned records on local NVMe, read with O_DIRECT. Pass `disk_dir="/nvme/ngq"`. The record file is
    built once (13.4 GB). This uses `native/rowread.c` and needs Linux + gcc.
- **Defaults as in our evaluation:**
  - `eval_compat=True` installs the exact QSA-indexer shortcut of our evaluation. For key lengths ≤ the indexer budget,
    top-k keeps every visible block, so the result is identical and faster. Longer contexts use the original code.
  - `experts_implementation="grouped_mm"`.
- **Expected log line.** transformers logs `Unknown quantization type, got ngq2 ... we will skip the quantization`. The
  generic loader does not know `ngq2`; `modeling_ngq2.py` attaches the packed table itself.

**Alternative without the auto-class indirection** (`ngq.py` from this repo):

```python
import ngq                                   # from this repo
model, tok = ngq.load("dnotitia/Qwen3.8-Flash-Next-NGQ2", table_device="gpu")   # same model, same logits
```

**Tested example:**

```bash
hf download dnotitia/Qwen3.8-Flash-Next-NGQ2 --local-dir Qwen3.8-Flash-Next-NGQ2 && cd Qwen3.8-Flash-Next-NGQ2
python examples/generate.py                                         # AutoModelForCausalLM, table on the GPU
python examples/generate.py --table_device cpu                      # table in host DRAM
python examples/generate.py --table_device disk --disk_dir /nvme/ngq  # table on local NVMe
python examples/generate.py --loader ngq                            # through ngq.py
```

## What NGQ2 is

The method applies a uniform-rate 2-bit trellis code to each table row, then restores each row's norm.

- **Code.** The QTIP bitshift trellis (Tseng et al., 2024): L = 16 state bits, 4 bits per pair of values, QTIP's
  "HYB" (`quantlut_sym`) codebook with a 512×2 base table, tail-biting. Each 160-dim row is one trellis sequence of 80
  steps (40 bytes), so every row decodes independently (row-granular).
- **Norm restoration.** One bf16 gain per row makes the decoded row's L2 norm equal the original row's norm (16 bits per
  160-dim row = +0.10 bpp).
- **What the method leaves out:**
  - no rotation;
  - no Hessian, calibration data or fine-tuning (data-free);
  - uniform rate.
- **Unchanged:** the hashing and addressing. Packed row r replaces row r of `ple.ple_embedding.ngram_embedding`.

**Same codes as evaluated.** The stored codes are the ones behind the numbers below (config `qtip2ncc_norht`).
- All 320,001,536 rows decode **bit-identically** to the evaluated table; see Verification.
- The last 90 rows are padding rows the hash never addresses.

## Checkpoint layout

The repo follows the base layout: `config.json`, `generation_config.json`, `chat_template.jinja`, the tokenizer and
processor files, and `LICENSE`. The weights sit in 100 safetensors shards of ≤ 5 GB, listed in
`model.safetensors.index.json`.

- **Backbone.** 1,530 tensors, byte-identical to the base checkpoint, with the same names and dtypes. This includes the MTP
  and vision weights.
- **Removed.** The 128 tensors `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight`.
- **Packed table.** It keeps the base checkpoint's 128 row shards, next to the tensors it replaces, in shards 98–100:

| tensor | dtype | shape | content |
|---|---|---|---|
| `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{j}.ngq2_symbols` | uint8 | [2,500,012, 40] | rows j·2,500,012 …: 80 4-bit trellis symbols per row (2 per byte, high nibble first) |
| `….ngram_embedding.shard_{j}.ngq2_gain` | bf16 | [2,500,012] | per-row gain |
| `ngq2.codebook` | fp32 | [512, 2] | HYB base table (full codebook = 65536 × 2, see `ngq.hyb_lut`) |

`config.json` is the base config with two additions:
- the `quantization_config` block below;
- an `auto_map` (`AutoModelForCausalLM` / `AutoModelForImageTextToText` → `modeling_ngq2.NGQ2Qwen4ExpForConditionalGeneration`).

`model_type` and `architectures` are unchanged.

```json
"quantization_config": {
  "quant_method": "ngq2", "version": 1, "bits": 2, "bpp": 2.1,
  "code": "qtip_hyb_trellis", "L": 16, "K": 2, "V": 2,
  "codebook": {"tensor": "ngq2.codebook", "type": "hyb", "base_entries": 512},
  "gain": "ncc_bf16", "granularity": "row",
  "modules_to_quantize": ["model.language_model.layers.1.ple.ple_embedding.ngram_embedding"],
  "packed_tensor_suffixes": {"symbols": "ngq2_symbols", "row_gain": "ngq2_gain"},
  "checkpoint_parts": "shard_{j}"
}
```

| field | meaning |
|---|---|
| `quant_method` | `"ngq2"`: 2-bit n-gram / Engram table quantization (this work) |
| `version` | layout version of the packed tensors (1) |
| `bits` | payload bits per table value (2) |
| `bpp` | bits per value including the per-row bf16 gain (2 + 16/160 = 2.10). The 4 KB codebook is not counted. |
| `code` | QTIP bitshift trellis decoded through the HYB (`quantlut_sym`) codebook |
| `L` | trellis state bits (16): state = the last 4 symbols |
| `K` | bits per value (2): one 4-bit symbol per `V` values |
| `V` | values per trellis step = codebook vector length (2) |
| `codebook` | tensor holding the HYB base table (fp32 [512, 2]); the full 65536-entry codebook is derived from it (`ngq.hyb_lut`) |
| `gain` | `ncc_bf16`: per-row norm restoration with one bf16 per row (here a gain g = ‖x‖/‖q‖, row = bf16(q·g)) |
| `granularity` | `row`: one trellis sequence per table row (rows decode independently) |
| `modules_to_quantize` | exact module name of the replaced table |
| `packed_tensor_suffixes` | names of the packed tensors: `<module>[.shard_{j}].<suffix>` |
| `checkpoint_parts` | the module's packed tensors are stored in row parts `<module>.shard_{j}.<suffix>`, concatenated in numeric j order (as the base `shard_{j}.weight`) |

The decode, with the precision of every step, is in [`VLLM_NOTES.md`](VLLM_NOTES.md).

### Where to put the table

These are measured tradeoffs from our serving study (Qwen on 2 × H200, HF eager):

| placement | measured tradeoff |
|---|---|
| `gpu` (default) | 13.4 GB in HBM next to the backbone. The 2-GPU layout is near the HBM limit at large decode batches. |
| `cpu` | Host DRAM: 102 GB (bf16) → 13.4 GB. Even the bf16 table served from pinned host DRAM matched GPU-resident speed in our runs (prefill 10.6k tok/s). On Qwen, compression mainly buys host / GPU memory, not speed. |
| `disk` | Local NVMe, O_DIRECT, one 4 KB page per row (16 rows/token). Not measured end-to-end for Qwen. On DeepSeek, SSD lookups with prefetch hid the latency, but compression gave no SSD speedup. |

## Hardware

- **Tested:** 2 × H200 (141 GB), `device_map="auto"`, transformers 5.17, torch 2.14.
- **Minimum measured: 2 × H200.** Peak GPU memory for the 24 × 2,047-token KL pass plus generation:
  - `table_device="gpu"`: 131.5 + 118.1 GiB;
  - `"cpu"`: 118.9 + 118.1 GiB;
  - `"disk"`: 118.9 + 118.1 GiB.
- The backbone alone is 257.6 GB (240 GiB). 80 GB GPUs need ≥ 4 (not tested).
- **Host RAM:** the loading process peaked at 236 GiB resident (memory-mapped checkpoint pages while loading).
- **Speed:**
  - The first `generate` call runs at ~1.1 tok/s because of one-time compilation.
  - After warm-up we measured 9.4–11.5 tok/s (gpu), 9.3–11.4 (cpu) and 11.8 (disk) in our earlier tests. That was HF eager,
    2 × H200, batch 1, 64 greedy tokens, with the same decode code.

## Results (our shared benchmark)

Numbers are taken from our benchmark result files.
- **Format:** chat format (thinking off) for generative tasks; raw text for log-likelihood / KL.
- **CIs:** 95%, item bootstrap (tasks) or chunk bootstrap (KL).
- **KL:** KL(original ‖ config) per token on 6 text domains.
- **Benefit retained:** (q − removed) / (original − removed), where removed = table zeroed. On this model the table
  matters much less for many tasks than on DeepSeek / LongCat, so several benefit CIs are wide.
- **RTN:** asymmetric g80 (d = 160) with bf16 scale and min.

**Fidelity**

| config | bpp | table GB | gm KL | worst KL | ben gm KL | top-1 agree mean / min (6 dom) | WT2 PPL |
|---|---|---|---|---|---|---|---|
| original (bf16, 16 bpp) | 16.00 | 102.40 | .000 [.000,.000] | .000 [.000,.000] | 1.000 [1.000,1.000] | 1.0000 / 1.0000 | 4.671 [4.336,5.000] |
| table removed (zeroed) | 0.00 | 0.00 | .308 [.296,.320] | .481 [.470,.491] | .000 [.000,.000] | 0.8151 / 0.7450 | 6.476 [6.182,6.764] |
| RTN 2-bit g80 (2.40 bpp) | 2.40 | 15.36 | .059 [.057,.061] | .099 [.090,.109] | .809 [.803,.816] | 0.9230 / 0.9092 | 5.089 [4.791,5.384] |
| QTIP 2.0, published recipe (2.00 bpp) | 2.00 | 12.80 | .035 [.033,.036] | .048 [.043,.052] | .887 [.883,.892] | 0.9392 / 0.9276 | 4.827 [4.515,5.141] |
| **NGQ2 (this repo, 2.10 bpp)** | 2.10 | 13.44 | .033 [.032,.034] | .044 [.040,.048] | .894 [.890,.898] | 0.9404 / 0.9279 | 4.830 [4.515,5.145] |

**Knowledge (absolute / benefit retained)**

| config | mmlu | popqa | tqa | nq | entityq | simpleqa |
|---|---|---|---|---|---|---|
| original (bf16, 16 bpp) | .868 [.848,.889] | .363 [.350,.377] | .823 [.810,.837] | .424 [.409,.440] | .372 [.355,.390] | .137 [.115,.158] |
| table removed (zeroed) | .859 [.838,.881] | .266 [.253,.278] | .734 [.718,.750] | .376 [.360,.391] | .205 [.191,.220] | .077 [.061,.094] |
| RTN 2-bit g80 (2.40 bpp) | .863 [.842,.885] / .444 [-4.000,4.000] | .351 [.338,.364] / .874 [.817,.935] | .820 [.806,.833] / .963 [.896,1.032] | .417 [.402,.432] / .845 [.726,.969] | .354 [.337,.371] / .892 [.839,.947] | .127 [.107,.147] / .833 [.644,1.061] |
| QTIP 2.0, published recipe (2.00 bpp) | .873 [.853,.894] / 1.556 [-3.792,7.000] | .361 [.348,.374] / .979 [.930,1.026] | .824 [.810,.837] / 1.007 [.947,1.074] | .421 [.405,.436] / .937 [.824,1.052] | .372 [.355,.390] / .998 [.954,1.042] | .132 [.111,.152] / .917 [.750,1.103] |
| **NGQ2 (this repo, 2.10 bpp)** | .868 [.848,.889] / 1.000 [-2.075,5.000] | .364 [.351,.378] / 1.014 [.967,1.067] | .822 [.808,.835] / .985 [.932,1.043] | .421 [.406,.436] / .937 [.829,1.046] | .366 [.349,.384] / .964 [.919,1.011] | .139 [.118,.160] / 1.033 [.859,1.255] |

**Multilingual / generation / task groups**

| config | belebele8 | mkqa8 | mgsm6 |
|---|---|---|---|
| original (bf16, 16 bpp) | .939 | .268 | .711 |
| table removed (zeroed) | .919 | .206 | .688 |
| RTN 2-bit g80 (2.40 bpp) | .939 / .975 [.798,1.190] | .269 / 1.008 [.925,1.101] | .707 / .829 [-.086,2.216] |
| QTIP 2.0, published recipe (2.00 bpp) | .939 / 1.000 [.844,1.182] | .266 / .964 [.892,1.041] | .705 / .714 [-.215,1.667] |
| **NGQ2 (this repo, 2.10 bpp)** | .940 / 1.012 [.864,1.200] | .268 / .988 [.914,1.063] | .719 / 1.314 [.658,3.953] |

| config | gsm8k | humaneval |
|---|---|---|
| original (bf16, 16 bpp) | .876 [.846,.904] | .957 [.927,.982] |
| table removed (zeroed) | .830 [.796,.862] | .945 [.909,.976] |
| RTN 2-bit g80 (2.40 bpp) | .864 [.836,.892] / .739 [.316,1.188] | .957 [.927,.982] / 1.000 [1.000,1.000] |
| QTIP 2.0, published recipe (2.00 bpp) | .866 [.836,.894] / .783 [.357,1.222] | .951 [.915,.982] / .500 [-2.000,3.000] |
| **NGQ2 (this repo, 2.10 bpp)** | .868 [.838,.896] / .826 [.478,1.167] | .957 [.927,.982] / 1.000 [1.000,1.000] |

| config | commonsense | knowledge | rc | multilingual | generation |
|---|---|---|---|---|---|
| original (bf16, 16 bpp) | .769 | .496 | .965 | .633 | .917 |
| table removed (zeroed) | .757 | .420 | .961 | .597 | .888 |
| RTN 2-bit g80 (2.40 bpp) | .773 / 1.370 [.908,2.563] | .489 / .906 [.856,.958] | .963 / .597 [-3.437,3.360] | .632 / .970 [.833,1.125] | .911 / .794 [.369,1.148] |
| QTIP 2.0, published recipe (2.00 bpp) | .770 / 1.100 [.693,1.889] | .495 / .991 [.951,1.036] | .965 / 1.031 [-2.490,4.168] | .631 / .928 [.809,1.062] | .909 / .723 [.062,1.230] |
| **NGQ2 (this repo, 2.10 bpp)** | .770 / 1.047 [.626,1.818] | .496 / .994 [.953,1.041] | .963 / .541 [-1.862,2.791] | .635 / 1.050 [.925,1.196] | .913 / .863 [.498,1.133] |

Answer flips vs the original over 35,213 0/1 items:
- NGQ2: 3.03% right→wrong, 4.27% wrong→right, net −0.05 pp.

Reading:
- NGQ2 keeps 89.4% of the table's KL benefit (gm).
- Its KL is 0.56× that of RTN 2-bit (paired, 95% CI 0.55–0.57) and 0.94× that of the published QTIP recipe
  (CI 0.92–0.97), which uses rotation and a global scale at 2.00 bpp.
- Knowledge-group benefit is 99.4%. GSM8K and HumanEval stay within noise of the original.


## Verification of this checkpoint (run before upload)

All checks below were run on this machine (H200) on the exact files in this repo. Scripts: `scripts/release/full/` in
our research repo.

| check | result |
|---|---|
| local base == Hub base | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d`: sizes of all files, git blob sha1 of all non-weight files, and sha256 of shards 00001 / 00005 / 00130 match the Hub metadata |
| backbone tensors | **1,530 / 1,530 tensors byte-identical** to the base shards (257.6 GB compared byte by byte; same names, dtypes, shapes). The key set equals the base keys minus the 128 table shards exactly. Every shard opens with `safetensors`; the index lists every tensor once. |
| packed tensors | byte-identical to the verified tensors of our earlier table-only release (re-split into the base's 128 row shards) |
| decoded rows == evaluated table | **320,001,536 / 320,001,536 rows bit-identical** to the evaluated (dequantized) table `qtip2ncc_norht`, decoded from this checkpoint by the code `from_pretrained` uses |
| how the codes were obtained | Viterbi on the evaluated rows with the stored codec (LUT, k = 4, L = 16), each row verified. The 90 padding rows (norm ≈ 1e-17) were re-derived with the original codec call. |
| KL through `from_pretrained` | Teacher = the stock base model with its bf16 table; student = `AutoModelForCausalLM.from_pretrained(this repo, trust_remote_code=True)`. 24 KL chunks (2,047 tokens each; web, code, math, chat, multi5, ja, sw), batch 1. |
| KL result | KL/token **0.03279** (top-1 0.9434). This equals the previous release loader's .03279 exactly. Recorded benchmark value .03142. With the recorded batching (8 chunks per batch), the evaluation harness reproduces every recorded chunk exactly; the batch-1 gap is run-to-run numerical noise of the model. |
| placements / loaders | `table_device="gpu"`, `"cpu"`, `"disk"` and `ngq.load(...)` give **bit-identical logits** on all 24 chunks |
| generation (greedy, chat template, thinking off) | "Here are three of Leo Tolstoy's most famous works: 1. **War and Peace** (1869) – An epic novel that intertwines the lives of several Russian families with the historical events of the Napoleonic Wars. …" |
| after upload: fresh directory, empty HF cache | The Usage example above, run verbatim, downloads this repo from the Hub (271.0 GB) and generates the same text. All 118 files match the pre-upload **sha256** manifest. `examples/generate.py` from the downloaded snapshot (gpu / cpu / disk / `--loader ngq`) gives the same logits as before upload: KL vs the evaluation harness logits .0040, top-1 97.0%. The harness uses a different SDPA backend / GPU layout; this does not involve the table. Peak 133.4 + 118.1 GiB (gpu) / 120.9 + 118.1 GiB (cpu, disk) on 2 × H200. |

## vLLM

Support for this checkpoint is proposed in [vllm-project/vllm#66576](https://github.com/vllm-project/vllm/pull/66576) (under review, not in a vLLM release yet). With that branch the checkpoint serves as published, with no conversion and no extra flag:

```bash
vllm serve dnotitia/Qwen3.8-Flash-Next-NGQ2 --tensor-parallel-size 4
```

The table stays in pinned host memory by default; `--engram-config '{"cpu_offload": false}'` keeps it on the GPU. Disk offload and non-NVIDIA GPUs are not supported in that PR.

Measured on 4 × H200, TP4, `--gpu-memory-utilization 0.90`, one run per configuration:

| | weights GiB / GPU | KV cache tokens | host RSS GiB | TPOT c=1 / c=32 ms |
|---|---|---|---|---|
| BF16 table on GPU | 84.61 | 2,034,688 | 18.3 | 5.01 / 15.79 |
| **NGQ2 table on GPU** | 63.89 | 3,147,264 | 18.2 | 5.02 / 16.14 |
| **NGQ2 table in host memory** | 60.76 | 3,312,640 | 35.3 | 4.99 / 16.80 |
| BF16 table in host memory | 60.76 | 3,309,568 | 146.3 | not run |

[`VLLM_NOTES.md`](VLLM_NOTES.md) gives the tensor names, shapes and dtypes and the exact decode.

## Limitations

- **Not lossless.** gm KL .033, worst .044; WikiText-2 PPL 4.67 → 4.83.
  - On this model the n-gram table's measured task benefit is small: removed and original differ by only a few points on
    most tasks.
  - So benefit-retained ratios have wide CIs, e.g. mgsm6 1.31 [.66, 3.95]. The KL numbers are the more reliable
    fidelity measure.
- **Decode on every lookup.** Pure-PyTorch decode, no fused kernel shipped.
- **Disk placement.** Experimental for Qwen (not benchmarked end-to-end); Linux-only.
- **transformers path.** `modeling_ngq2.py` overrides `from_pretrained` and builds the built-in `Qwen4ExpTextNGramEmbedding`
  without its table. It is tested with transformers 5.17. `save_pretrained` of the loaded model is not supported: it would
  not write the packed table.

## License

This repo is distributed under the **Qwen Community License 1.0**, the license of the base model. The full text is in
`LICENSE` (Copyright (c) 2026 Qwen).

It contains Qwen3.8-Flash-Next's weights (byte-identical) and a compressed derivative of its n-gram table. Note the
license's conditions, including:
- the attribution / model-name display requirement above 100M monthly active users or US$20M monthly revenue;
- that Model-as-a-Service or AI Work Assistant businesses need a separate license from Qwen for commercial use.

Built with Qwen. The trellis code follows QTIP (Tseng et al., 2024). Paper: [N-Gram Memories at Two Bits](https://arxiv.org/abs/2610.21810).
