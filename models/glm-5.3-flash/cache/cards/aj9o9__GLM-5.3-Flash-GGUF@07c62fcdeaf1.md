---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
language:
  - en
  - zh
tags:
  - gguf
  - llama.cpp
  - moe
---

# GLM-5.3-Flash, hand-mixed GGUF

Two quants, 87 GB and 112 GB, built by assigning every tensor class by hand
instead of using a preset.

| build | size | bpw | shards | runs on |
|---|---:|---:|---:|---|
| **AJ-IQ3_XXS** | 112.40 GB | 2.80 | 3 | 128 GB unified memory, or a 96 GB card with a small spill |
| **AJ-IQ2_XXS** | 87.35 GB | 2.23 | 2 | a 24 GB card with 64 GB of system RAM |

## The model

GLM-5.3-Flash is 320.6B total parameters with 17.3B active. Three things about
its architecture drive every decision in these quants.

**It is 94.9% routed experts.** 304.4B of the 320.6B parameters sit in
`ffn_*_exps` tensors across 42 sparse layers, 288 experts each, 8 active per
token plus 1 shared. Everything else in the model, all the attention and
embeddings and norms, is 8.9B parameters. That means the file size is set almost
entirely by one decision: how many bits the experts get. Everything else can be
kept generous for almost nothing.

**Attention is hybrid, and mostly not attention.** Of 45 layers, 34 use Kimi
Delta Attention, a linear-attention variant that carries a fixed-size recurrent
state rather than a growing KV cache. The other 11 use Multi-head Latent
Attention with a DeepSeek sparse attention indexer, and MLA caches only a
512-wide latent. The practical result is that **context is unusually cheap**:
32K costs 0.44 GB, 131K costs 1.55 GB, 300K costs 3.45 GB at f16.

**The KDA layers name their projections like normal attention.** They use
`attn_q`, `attn_k`, `attn_v`, not `ssm_*` as the recurrent naming might suggest.
Getting this wrong is silent: unmatched tensors fall through to the base type
with no error. An early version of this mix missed them and 3.42B parameters
quietly dropped to 2 bits.

There is also one MTP/NextN block. Both builds drop it with `--prune-layers 45`,
so the parameter count in these files is 313.33B rather than 320.6B, and
speculative decoding through the MTP head is not available.

## The two quants and what actually differs

The mix files are **identical except for their routed-expert rules**. Verified by
md5 over everything else. So the entire difference in quality between these two
files is attributable to the experts alone, with no other variable moving.

| tensor class | AJ-IQ2_XXS | AJ-IQ3_XXS |
|---|---|---|
| norms, router, `hc_` | F32 | F32 |
| KDA gates and decay, DSA indexer | Q8_0 | Q8_0 |
| all attention, embeddings, lm_head, dense FFN, shared expert | Q6_K | Q6_K |
| **routed experts, gate and up** | **IQ2_XXS** | **IQ2_S** |
| **routed experts, down** | **IQ2_XXS** (IQ2_S on edge layers) | **IQ3_XXS** (IQ3_S on edge layers) |

That is the whole delta. 25 GB, spent entirely on 304.4B expert weights.

**Why down gets the higher tier.** Gate and up together are two thirds of every
expert layer's weights, so they set the price. Down is the remaining third and is
the more sensitive of the three. Raising down by one tier costs 12.7 GB; raising
gate and up costs 25.4 GB. Buying quality through down first is simply cheaper
per unit of improvement, and the measurements below bear that out.

## Quality

**How this was measured, in full.** Wikitext-2 `wiki.test.raw`, 30 chunks,
`n_ctx=512`, `llama-perplexity -fa on -b 512 -ub 512`, llama.cpp PR #27752 at
commit `24652c57a`, on an RTX PRO 6000 Blackwell. The KLD base is BF16 logits
generated in the same session, on the same build, with the same flags, from
`unsloth/GLM-5.3-Flash-GGUF` BF16. All three rows are one build, one corpus, one
chunk count. Only quantization differs.

| model | size | bpw | PPL | vs BF16 | KLD | RMS Δp | Same top-p |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 642 GB | 16 | 2.2474 ± 0.0452 | base | 0 (base) | n/a | n/a |
| **AJ-IQ3_XXS** | **112.40 GB** | **2.80** | **3.0147 ± 0.0687** | **1.341x** | **0.3557 ± 0.0089** | **23.80%** | **81.77%** |
| **AJ-IQ2_XXS** | **87.35 GB** | **2.23** | **4.2314 ± 0.1070** | **1.883x** | **0.7072 ± 0.0145** | **34.16%** | **73.20%** |

- **The extra 25 GB halves the divergence.** KLD 0.7072 to 0.3557, a factor of
  1.99. Same-top-p gains 8.56 points. The PPL penalty falls from 1.88x to 1.34x.
  Error bars do not overlap on any column.
- **The cost is 0.0140 KLD per extra GB.** That is the number to beat before
  baking a further tier.
- **30 chunks, not 580.** Wider error bars than a full run. BF16 costs 136 s per
  pass even on a 96 GB card, because 642 GB does not fit in a 367 GB container
  and streams from disk every pass. The quants run at about 1 s per pass, so the
  chunk count is a BF16 budget decision, not a methodology preference.

### Why every run here uses `-b 512`

Every number above uses `-b 512 -ub 512`. The `llama-perplexity` default of
`-b 2048` gives `n_seq=4`, and that path is not reliable on this architecture. It
returns `nan` from chunk 2 on quantized models. On BF16 it returns finite but
divergent values:

| chunk | BF16 at n_seq=4 | BF16 at n_seq=1 |
|---|---:|---:|
| 1 | 1.9119 | 1.9139 |
| 2 | 7.2195 | 2.8699 |
| 3 | 6.2120 | 2.2662 |
| 4 | 4.8322 | 1.9707 |

Chunk 1 agrees to four significant figures across two different builds. Chunk 2
and everything after diverges by up to 2.5x, at exactly the chunk where quantized
models produce `nan`. The multi-sequence path was corrupting results silently.

**Earlier versions of this model card carried numbers from that path. They were
wrong and have been replaced.** Any perplexity figure for this architecture
published without `-b 512` should be treated with suspicion.

## Against unsloth

`unsloth/GLM-5.3-Flash-GGUF` publishes top-1 accuracy and mean KLD. Those are
`Same top p` and `Mean KLD` from the same `llama-perplexity --kl-divergence`
tool, so they are the same quantities. **The corpus and the base logit file
differ, so this is indicative, not matched.** A properly matched comparison would
need their GGUFs scored against the same base, which has not been done.

| quant | size | top-1 | KLD |
|---|---:|---:|---:|
| UD-IQ1_S | 93.09 GB | 70.89% | 0.6697 |
| UD-IQ1_M | 97.58 GB | 73.06% | 0.5724 |
| **AJ-IQ2_XXS** | **87.35 GB** | **73.20%** | **0.7072** |
| UD-IQ2_XXS | 101.84 GB | 76.30% | 0.4501 |
| UD-Q2_K_XL | 108.72 GB | 78.34% | 0.3801 |
| **AJ-IQ3_XXS** | **112.40 GB** | **81.77%** | **0.3557** |
| UD-IQ3_XXS | 120.37 GB | 81.63% | 0.2838 |
| UD-Q3_K_XL | 147.54 GB | 86.25% | 0.1597 |

- **AJ-IQ3_XXS matches UD-IQ3_XXS on top-1 while being 6.6% smaller**, 81.77%
  against 81.63% at 112.40 GB against 120.37 GB. On a 128 GB machine that is
  15.6 GB free instead of 7.6 GB, which decides whether the file runs with real
  context or only just loads.
- **AJ-IQ2_XXS matches UD-IQ1_M on top-1 while being 10.5% smaller**, 73.20%
  against 73.06% at 87.35 GB against 97.58 GB.
- **UD wins on KLD at both sizes**, 0.2838 against 0.3557 and 0.5724 against
  0.7072. Same argmax agreement, closer full distribution.
- **What that trade means in practice.** Keeping routers at F32 and attention at
  Q6_K preserves which expert fires and which token wins. Pushing gate and up
  down to 2 bits coarsens the magnitudes behind that choice, so error lands in
  the tail rather than in the top-1 pick. These builds suit greedy and
  low-temperature decoding. For sampling at higher temperature, the unsloth
  builds at comparable size are the better choice.

## Speed

`llama-bench`, `-ctk q8_0 -ctv q8_0 -fa on`, nothing else running.

**AJ-IQ2_XXS, RTX PRO 6000 Blackwell 96 GB, fully resident**

| test | t/s |
|---|---:|
| tg256 | 69.89 ± 0.15 |
| tg256 @ d8192 | 52.74 ± 0.35 |
| tg256 @ d16384 | 44.30 ± 0.15 |
| tg256 @ d32768 | 31.70 ± 0.38 |
| tg256 @ d65536 | 20.91 ± 0.05 |
| tg256 @ d131072 | 12.24 ± 0.00 |

**AJ-IQ2_XXS, single RTX 3090 24 GB + 64 GB DDR5, `-ncmoe 40`**

| test | t/s |
|---|---:|
| pp2048 | 36.69 ± 6.23 |
| tg128 | 11.55 ± 0.68 |

- **Where the experts live is the only variable that matters.** The same 87 GB
  file gives 69.89 tok/s resident on a 96 GB card and 11.55 tok/s with 40 of 42
  expert layers on CPU. A 6x gap, entirely PCIe streaming.
- **Depth roughly halves the rate every 32K.** 69.89 at zero, 31.70 at 32K, 12.24
  at 131K. Budget for that if you plan long-context work.
- **AJ-IQ3_XXS is not benchmarked.** At 104.68 GiB against 95.6 GiB of card it
  cannot go fully resident and needs a small spill, so it would not be comparable
  to the rows above anyway. Expect it to land between the two tables.

## Run

Everything on GPU, 96 GB or more:

```bash
llama-server -m GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf \
  -ngl 99 -fa on -c 131072 -ctk q8_0 -ctv q8_0 --jinja
```

AJ-IQ3_XXS on a 96 GB card, small spill:

```bash
llama-server -m GLM-5.3-Flash-AJ-IQ3_XXS-00001-of-00003.gguf \
  -ngl 99 -ncmoe 14 -fa on -c 131072 -ctk q8_0 -ctv q8_0 --jinja
```

24 GB card with system RAM, all experts on CPU:

```bash
llama-server -m GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf \
  -ngl 99 -cmoe -fa on -c 32768 -ctk q8_0 -ctv q8_0 --jinja
```

Point at shard 1 and llama.cpp finds the rest. Prefer `-ncmoe N` over `-cmoe`
whenever there is VRAM to spare; every layer kept on the card is a large win.

**Use `-np 1` unless you genuinely have concurrent requests.** `-np N` divides
`-c` across N slots, so `-c 131072 -np 4` gives each request only 32,768 tokens.

### Build requirement

Needs llama.cpp from [PR #27752](https://github.com/ggml-org/llama.cpp/pull/27752),
which is unmerged. Specifically the commit `llama : skip foreign-sequence cells
when building the glm5next k-pools` or later. Earlier commits return `nan` on the
CPU expert path. The PR gets rebased, so pin by that commit message rather than
by hash.

## The mix, and why it is ordered

`--tensor-type-file` is **first regex match wins**, using substring search. A
general rule placed too early silently swallows the ones after it. Order:

| class | type | reason |
|---|---|---|
| norms, `hc_`, `ffn_gate_inp`, `exp_probs_b` | F32 | the router picks 8 of 288; a wrong pick costs far more than any weight error |
| KDA gates and decay | Q8_0 | recurrent state, error accumulates along the sequence |
| DSA indexer | Q8_0 | **must precede** the attention rules, or `indexer.attn_k.weight` is caught by `attn_k\.weight` |
| attention, embeddings, lm_head, dense FFN, shared expert | Q6_K | 8.9B params total, cheap to keep rich |
| routed experts | IQ2_XXS or IQ2_S / IQ3_XXS | the entire budget |

Everything above the experts is **2.84% of the model and 7.5 GB**. Raising all of
it from Q6_K to Q8_0 costs 2.09 GB, which spent on experts instead would buy only
0.055 bpw across 304B weights. That is why it stays where it is, and why both
builds share it unchanged.

## Files

- `AJ-IQ3_XXS/GLM-5.3-Flash-AJ-IQ3_XXS-0000{1,2,3}-of-00003.gguf`
- `AJ-IQ2_XXS/GLM-5.3-Flash-AJ-IQ2_XXS-0000{1,2}-of-00002.gguf`
- mixes: `glm53-flash-iq3xxs.txt`, `glm53-flash-iq2xxs.txt`
- base: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), MIT
- BF16 GGUF and imatrix: [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF).
  Using theirs saved a 328 GB FP8 download and an imatrix pass that is not
  practical at this scale.
