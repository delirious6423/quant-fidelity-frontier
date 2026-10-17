---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: text-generation
tags:
- dynquant
- quantization
- mixed-precision
- text-to-sql
- qlora
language:
- en
---

# Qwen3.8-27B-DynQuant-3bit

`Qwen/Qwen3.8-27B`, fine-tuned for text-to-SQL with QLoRA and then quantized with [DynQuant](https://github.com/kambojvikram/dynquant) to **2.999 bits per weight** (9.41 GiB on disk).

DynQuant gives every module its own width, driven by two signals measured *during* the fine-tune: how much activation mass each weight sees, and how unstable its gradient is across optimizer steps. Modules the training dynamics say are load-bearing keep their bits; the rest pay for them.

## Read this first

This architecture has a **floor budget of 4.0196 bits** -- the narrowest average width at which every module can still be given the minimum its role requires. That is measured, not assumed: it is the narrowest budget in this campaign's sweep that breached no floor.

This arm was exported at **2.999 bits, 1.02 bits below it**. So it is not an allocation within the architecture's limits; it is a measurement of what overriding those limits costs. The table further down names every floor it broke. The score was measured exactly as every other arm's was -- what it is a score *of* is a deliberately over-compressed model.

For the smallest arm that breaks nothing, use the 4.02-bit budget instead.

## Results

Execution accuracy on the held-out validation split of `spider`, `gretel`, `wikisql`, 400 problems, greedy decode.

| Model | Bits | Size | Accuracy | vs bf16 | p |
|---|---:|---:|---:|---:|---:|
| bf16 (unquantized) | 16 | -- | 85.50% | -- | -- |
| **this checkpoint** | 2.999 | 9.41 GiB | 79.50% | -6.00 | 1.93e-05 |

The comparison is paired: both arms answered the same problems in the same order, so the difference is a McNemar test on the per-problem outcomes rather than two independent accuracies subtracted.

Decode budget was 1024 new tokens, and 0 generations reached it without finishing. A short decode budget scores a model that deliberates as though it were wrong, so this is reported rather than assumed.

## What the allocator did

498 modules were quantized at group size 128, scored by the `rank_product` allocator.

| Width | Modules | Parameters | Share |
|---|---:|---:|---:|
| 2-bit | 112 | 6,878,658,560 | 25.58% |
| 3-bit | 290 | 19,991,101,440 | 74.33% |
| 8-bit | 96 | 23,592,960 | 0.09% |

### Floors

Each role carries a minimum width below which that role is known to break -- an embedding, an LM head, an attention projection and an MLP gate do not tolerate the same compression. When a budget cannot pay for every floor, DynQuant breaks the cheapest ones and **reports every one it broke**, rather than quietly lowering a floor until the arithmetic works.

310 modules were allocated below the floor their role requires:

| Role | Modules | Parameters | Floor | Given |
|---|---:|---:|---:|---:|
| `mlp.gate` | 64 | 5,704,253,440 | 4b | 2b/3b |
| `lin_attn.qkv` | 48 | 2,516,582,400 | 4b | 2b/3b |
| `mlp.down` | 18 | 1,604,321,280 | 3b | 2b |
| `mlp.up` | 18 | 1,604,321,280 | 3b | 2b |
| `lin_attn.z` | 48 | 1,509,949,440 | 4b | 2b/3b |
| `lin_attn.out` | 48 | 1,509,949,440 | 4b | 2b/3b |
| `lm_head` | 1 | 1,271,398,400 | 8b | 3b |
| `embedding` | 1 | 1,271,398,400 | 4b | 3b |
| `attn.q_gate` | 16 | 1,006,632,960 | 4b | 2b/3b |
| `attn.o` | 16 | 503,316,480 | 4b | 2b/3b |
| `attn.k` | 16 | 83,886,080 | 4b | 2b/3b |
| `attn.v` | 16 | 83,886,080 | 4b | 2b/3b |

## Use it

```bash
pip install "dynquant[hf]"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import dynquant

# Both lines are needed. transformers has no entry-point discovery for
# quantization methods, so without the registration call it skips the
# quantization it does not recognise and hands back a randomly initialised
# model -- fluent-looking output, no exception, no non-zero exit.
dynquant.register_hf_quantizer()

model = AutoModelForCausalLM.from_pretrained("VikramPal/Qwen3.8-27B-DynQuant-3bit", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("VikramPal/Qwen3.8-27B-DynQuant-3bit")
```

That path is measured rather than assumed: this repo was pulled from the Hub and loaded with the released wheel on torch 2.13, and the graph that came back was checked to hold 498 DynQuant modules at the widths in the table above.

### Serving

`vllm serve VikramPal/Qwen3.8-27B-DynQuant-3bit` does **not** work, and the failure belongs here rather than in your terminal:

```
ValueError: There is no module or parameter named 'embed_tokens.offsets' in Qwen3_5Model
```

vLLM builds this architecture's input embedding without offering it to a quantization plugin at all, so a packed `model.embed_tokens` has nowhere to land. It is not a flag you are missing and it is not the plugin failing to register: on vLLM 0.27.1 the DynQuant plugin loads, the engine reports `quantization=dynquant`, and then this raises.

That one tensor is the whole gap. Re-exported with the input embedding in the compute dtype and the other 497 quantized modules left exactly as they are, this arm serves on vLLM 0.27.1 and answers correctly -- so nothing else in the allocation is in the way. That export is not what is published here. Until it is, use the transformers path above.

## How it was made

- **Base**: `Qwen/Qwen3.8-27B`, text tower only
- **Regime**: qlora, LoRA rank 32
- **Data**: 9,999 conversations from `spider`, `gretel`, `wikisql`, `create-context`, 350,799 supervised tokens
- **Steps**: 625 at effective batch 16, lr 0.0001, final train loss 0.0963
- **Signals**: collected from 546 modules during the fine-tune itself, with no extra forward or backward pass
- **DynQuant**: 0.5.2

- **Contamination**: the training split was matched against every evaluation item and 601 examples were dropped for colliding with one. No source overlaps an evaluation task after that.

## Citation

The method has no published paper yet, so these cite the code and these weights.

```bibtex
@software{dynquant,
  author  = {Pal, Vikram},
  title   = {{DynQuant}: dynamic-signal quantization for large language models},
  version = {0.5.2},
  url     = {https://github.com/kambojvikram/dynquant},
  year    = {2026}
}

@misc{qwen3_8_27b_text2sql_dynquant_3bit,
  author       = {Pal, Vikram},
  title        = {Qwen3.8-27B-DynQuant-3bit},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/VikramPal/Qwen3.8-27B-DynQuant-3bit}}
}
```

## Limitations

- Fine-tuned and evaluated on text-to-SQL. General-purpose ability was not measured, and quantization is not free elsewhere.
- Only the base model's text tower was trained and quantized. This checkpoint does not carry the vision path.
- The score is execution accuracy on the datasets named above, against their own schemas. Accuracy on your schemas is a different measurement.

- 310 modules sit below their role's floor; the table above says which.
