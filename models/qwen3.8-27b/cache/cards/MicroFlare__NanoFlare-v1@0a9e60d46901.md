---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---

NanoFlare is my newest model, based on the recently released Qwen 3.8 27B. It is designed to run completely on GPU if you have 8GB of VRAM, otherwise it can be run on CPU if you have at least 10GB or more of system memory (this will be slow). I have applied a non-uniform quantization across the weights, allocating more precision to the more sensitive parts of the model to preserve as much quality as possible.

# v1.c is the latest version (recommended)
_Revision C brings more improvements to the quality of output, and coherence in the model's reasoning process. It's been squeezed down to a reduced size so that a small kv cache can be fit onto the GPU as well._

If you are running NanoFlare on an 8GB GPU, you can have up to around an 8k context on GPU (tested with k:q5_0, v:iq4_nl), but if you want it higher you will have to offload the kv cache to CPU.

The MTP draft predictor is stripped out of regular NanoFlare to save space, but there is also a build that includes it, look for "--with-mtp" in the file name. Note, I can not get this version to fit fully onto my own 8GB card, so to use the MTP included version you may need a 10GB GPU or bigger.

Relative perplexity scores  (all tested with wikitext2, _lower is better_):
|Model |Size |2k ctx PPL |8k ctx PPL |
|--- |--- |--- |--- |
|**v1 revision c** |_7130.2 MB_ |_8.1826 ±0.05284_ | **7.8064 ±0.05058** |
|v1 revision b |7469.3 MB |8.2480 ±0.05573 |7.9631 ±0.05479 |
|unsloth/IQ2_XXS (UD 3.0) |6929.5 MB |**7.5441 ±0.04906** |8.0013 ±0.05689 |
|v1 revision a |7165.9 MB |8.6832 ±0.05605 |8.1774 ±0.05276 |
|release candidate 1 |7442.6 MB |8.7946 ±0.05764 |8.2998 ±0.05454 |
|beta 1 |7532.8 MB |8.8221 ±0.05797 |8.3352 ±0.05506 |

The model seems to run best at `temperature: 0.95` + `top-k: 20`.

Note this model has been hyper-compressed, so it's not perfect. It may flake out on you from time to time, but it is mostly fairly stable. Taking quantization to the extreme levels I'm using comes at a cost. The LLM will sometimes not complete its thought process. If you find yourself in this situation, try the prompt again. It often works correctly on the second try when this happens. It seems to be pretty good at most general tasks, and is good at coding. It has forgotten some facts and figures from its training data through the heavy quantization, so I'd recommend giving it web search access or double checking anything its telling you from recall.

From subjective, manual testing v1.c seems to be the most stable and usable version so far.

## MMLU Pro (limited) Results*
_*note only a partial set was ran, the first 20 questions in each of the 14 categories_

Tests were ran at a 5 shot with an 8k context, using NanoFlare v1.c. Top-k: 40, Temperature: 0.80 (llama.cpp defaults)

| Tasks |Version|    Filter    |n-shot|  Metric   |Value |Stderr|
|-------|------:|--------------|-----:|-----------|----:|-----:|
|biology|    3.1|custom-extract|     5|exact_match|  0.9|±0.0688|
|business|    3.1|custom-extract|     5|exact_match| 0.65|±0.1094|
|chemistry|    3.1|custom-extract|     5|exact_match| 0.75|±0.0993|
|computer_science|    3.1|custom-extract|     5|exact_match|  0.7|±0.1051|
|economics|    3.1|custom-extract|     5|exact_match|  0.6|±0.1124|
|engineering|    3.1|custom-extract|     5|exact_match| 0.45|±0.1141|
|health|    3.1|custom-extract|     5|exact_match|  0.5|±0.1147|
|history|    3.1|custom-extract|     5|exact_match|  0.5|±0.1147|
|law  |    3.1|custom-extract|     5|exact_match| 0.45|±0.1141|
|math |    3.1|custom-extract|     5|exact_match| 0.65|±0.1094|
|other|    3.1|custom-extract|     5|exact_match|  0.6|±0.1124|
|philosophy|    3.1|custom-extract|     5|exact_match| 0.55|±0.1141|
|physics|    3.1|custom-extract|     5|exact_match| 0.55|±0.1141|
|psychology|    3.1|custom-extract|     5|exact_match| 0.65|±0.1094|
|**Total** | | MMLU Pro ||average|**0.61** | |

So NanoFlare should give you fairly decent quality, but [microFlare](https://huggingface.co/MicroFlare/microFlare-v1) scored a 0.73 and will give you more stable output.

For more details, see my [announcment blog post](https://microflare.bearblog.dev/nanoflare-compresses-qwen-27b-down-to-fit-on-an-8gb-gpu/).