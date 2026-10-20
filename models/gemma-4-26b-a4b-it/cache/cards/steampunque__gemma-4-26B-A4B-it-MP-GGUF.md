---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
- gemma 4 26B A4B
- GGUF
- quantized
- 4-bit
- mixed precision
---

## Mixed Precision GGUF layer quantization of gemma-4-26B-A4B-it by Google

Original model: https://huggingface.co/google/gemma-4-26B-A4B-it

The hybrid quant employs different quantization levels on a per layer basis to enable
both high performance and small file size at the same time. The quants employed are all K to avoid
slow CPU or older GPU processing of IQ quants.  For this file the Q4_E_H layer quants using new extended
K layer definitions are as follows (refreshed 5/6/2026):

```
  LAYER_TYPES='[
   ["A","attn","Q","attn_q","K","attn_k","V","attn_v","O","attn_o","S","ssm","F","ffn","G","ffn_g","U","ffn_u","D","ffn_d"],
   ["MAP","VOD","0","QN_K","2","Q2_K","3","Q3_K","4","Q4_K","5","Q5_K","6","Q6_K","8","Q8_0","h","F16","f","F32"],
   [0 ,"Q5_K_656"],[1 ,"Q4_K_646"],[2 ,"Q4_K_554"],[3 ,"Q3_K_555"],[4 ,"Q4_K_554"],[5 ,"Q3_K_555"],[6 ,"Q4_K_554"],[7 ,"Q3_K_555"],
   [8 ,"Q4_K_554"],[9 ,"Q3_K_555"],[10,"Q4_K_554"],[11,"Q3_K_555"],[12,"Q4_K_554"],[13,"Q4_K_554"],[14,"Q4_K_554"],[15,"Q4_K_554"],
   [16,"Q4_K_665"],[17,"Q4_K_554"],[18,"Q4_K_665"],[19,"Q4_K_554"],[20,"Q4_K_665"],[21,"Q4_K_665"],[22,"Q4_K_665"],[23,"Q4_K_665"],
   [24,"Q4_K_665"],[25,"Q4_K_665"],[26,"Q5_K_665"],[27,"Q5_K_665"],[28,"Q5_K_666"],[29,"Q6_K_886"]
   ]'
   FLAGS="--token-embedding-type Q5_K --output-tensor-type Q6_K --layer-types-high"
```

The quant was tested for very strong performance over a small set of curated reasoning prompts and sized to
slightly smaller than Q4_K_M bit efficiency.  It solved almost the entire set of eval prompts extremely efficiently
and correctly without using think block.

A second Q4_P_H E quant is available which pads FFN tensor lengths to eliminate all legacy quants from the model:

```
   LAYER_TYPES='[
   ["A","attn","Q","attn_q","K","attn_k","V","attn_v","O","attn_o","S","ssm","F","ffn","G","ffn_g","U","ffn_u","D","ffn_d"],
   ["MAP","VOD","0","QN_K","2","Q2_K","3","Q3_K","4","Q4_K","5","Q5_K","6","Q6_K","8","Q8_0","h","F16","f","F32"],
   [0 ,"Q5_K_655"],[1 ,"Q5_K_555"],[2 ,"Q4_K_544"],[3 ,"Q4_K_544"],[4 ,"Q4_K_544"],[5 ,"Q4_K_544"],[6 ,"Q4_K_544"],[7 ,"Q4_K_544"],
   [8 ,"Q4_K_544"],[9 ,"Q4_K_544"],[10,"Q4_K_544"],[11,"Q4_K_544"],[12,"Q4_K_544"],[13,"Q4_K_544"],[14,"Q4_K_544"],[15,"Q4_K_544"],
   [16,"Q4_K_654"],[17,"Q4_K_654"],[18,"Q4_K_654"],[19,"Q4_K_654"],[20,"Q4_K_654"],[21,"Q4_K_654"],[22,"Q4_K_654"],[23,"Q4_K_655"],
   [24,"Q4_K_655"],[25,"Q4_K_665"],[26,"Q5_K_665"],[27,"Q5_K_666"],[28,"Q5_K_668"],[29,"Q6_K_866"]
   ]'
   FLAGS="--token-embedding-type Q6_K --output-tensor-type Q6_K --layer-types-high --tensor-pad [[704,768],[1408,1536,2],[2112,2304]] --override-kv gemma4.expert_feed_forward_length=int:768 --override-kv gemma4.feed_forward_length=int:2304"   
```

This quant upsizes to approx Q4_K_M size due to the tensor padding and elimination of all Q3_K tensors from the quant but noticeably outperforms Q4_E_H on some test prompts.

A slightly larger Q6_K_H quant is also available:
```
Q6_K_M : attn_v = q8_0 ffn_d = q8_0
Q6_K_L : attn_v = q8_0 attn_o = q8_0 ffn_d = q8_0

   LAYER_TYPES='[
   [0 ,"Q6_K_S"],[1 ,"Q5_K_M"],[2 ,"Q4_K_M"],[3 ,"Q4_K_S"],[4 ,"Q4_K_M"],[5 ,"Q4_K_S"],[6 ,"Q4_K_M"],[7 ,"Q4_K_S"],
   [8 ,"Q4_K_M"],[9 ,"Q4_K_S"],[10,"Q4_K_M"],[11,"Q4_K_S"],[12,"Q4_K_M"],[13,"Q4_K_S"],[14,"Q4_K_M"],[15,"Q4_K_S"],
   [16,"Q4_K_M"],[17,"Q4_K_M"],[18,"Q4_K_M"],[19,"Q4_K_M"],[20,"Q4_K_M"],[21,"Q4_K_L"],[22,"Q4_K_M"],[23,"Q4_K_L"],
   [24,"Q5_K_S"],[25,"Q5_K_M"],[26,"Q5_K_L"],[27,"Q6_K_S"],[28,"Q6_K_M"],[29,"Q6_K_L"]
   ]'
   FLAGS="--token-embedding-type Q6_K --output-tensor-type Q6_K --layer-types-high"
```

This quant uses a minumum Q4_K_S across layers, Q6_K embeddings, and and a strong Q6_K_L final layer. The quant efficiently aced the entire
set of eval prompts with no think block turned on (except for the trick test question which most models miss, but
it gets it with think enabled).  It also does a good job with code gen and is sized at a very signficant 5G smaller than Q6_K
to enable largest possible context size when fully offloaded into 24G VRAM.

Comparison:

Quant  |  size   |  PPL  |   Comment
-------|---------|-------|-----------
Q4_K_M | 16.8e9  | 15.1  | modified PPL, see discussion below.
Q4_E_H | 16.1e9  | 14.5  | modified PPL, 0.7B smaller than Q4_K_M
Q4_P_H | 16.7e9  | 14.5  | modified PPL, ~ Q4_K_M size, all K quants
Q6_K   | 22.6e9  | 13.7  | modified PPL
Q6_K_H | 17.5e9  | 14.2  | modified PPL, 5G smaller than Q6_K

Usage:

gemma 4 26B A4B it is a vision capable moe RL model. It can be used together with its multimedia projector layers to process images and text inputs
and generate text outputs. The mmproj file is made available in this repository.

Thinking:

By default the model will not create a RL reasoning block and just outputs

```
<|channel>thought
<channel|>
```

at the start of gen.  To get it to fill in the think block use a system prompt with:
```
<|think|>
```
as the first token.  This is a special token in the model vocab and must be tokenized as such to work.  No other text in the system prompt
besides the think token is needed to get it to fill in the RL block though other text can be added if desired.

The model was found to be highly capable on reasoning tasks when skipping think block.

Running:

Use of speculation with the model is not recommended since it is a moe with 4B activated experts meaning it can be efficiently run on CPU,
but CPU does not have enough parallel hardware to leverage processing many tokens in a batch at once.  Thus gen rate will go down if speculation
is used with experts processed on CPU.  If the model is run fully on GPU speculation can be used with the model.  A recommended low overhead
speculator is gemma-3-270m-it-256k. To use this speculator the inference platform must support dynamic vocab translation between draft and target.
Very low performance gains are found with speculation possibly due to some interaction with the moe experts that the dense draft cannot predict reliably.
Google has released a new model https://huggingface.co/google/gemma-4-26B-A4B-it-assistant specifically designed to draft gemma-4 but it is not yet
supported by llama.cpp.

The model can be run fully offloaded into 24G VRAM, or with CPU and expert layer offload via config OT="-ot exps=CPU -ngl 99".
Because the model is a 4B active moe the CPU expert offload still gives good gen rate with very large context available, however prompt
processing will be very slow for large contexts with experts offloaded to CPU.

On a 9900k/4070 or 2x 4070 setup (1 RPC) approx performance for the Q4_E_H quant is:

CONFIG  (no vision tower)   |   QKV    |  NKV   |  gen tps | pp tps (batch 128)
----------------------------|----------|--------|----------|--------
4070+ 9900k CPU exp offload | F16      |  256k  |  23      |  ~100
2x4070 (RPC)                | F16      |  128k+ |  77      |  ~1000
2x4070 (RPC) + spec , ND=2  | F16      |  128k+ |  84      |
2x4070 (RPC)                | Q8_0     |  256k  |  76      |
2x4070 (RPC) + spec , ND=2  | Q8_0     |  256k  |  79      |

The Q4_E_H model passed two long context tests with Q8_0 QKV running on 2x4070 showing usable prompt processing speed for
100k+ token prompts. It impressively handles a 106k token prompt  https://thireus.com/REDDIT/Qwen3_Runescape_Massive_Prompt.txt
very efficiently (greedy sampling used for gen):

```
lm Qwen3_Runescape_Massive_Prompt.txt 
<|channel>thought
<channel|>Based on the **Skills/Experience Table** provided in the text, the maximum experience level is **99**, which requires **13,034,431** experience.

Half of that maximum experience is **6,517,215.5**.

Looking at the experience requirements in the table:
*   Level **92** requires **6,517,253** experience.
*   Level **91** requires **5,902,831** experience.

Therefore, you reach half of the maximum experience at **level 92**.
```
Q4_P_H was also tested in F16 KV on the runescape prompt:
```
lm Qwen3_Runescape_Massive_Prompt.txt 
<|channel>thought
<channel|>Based on the "Skills/Experience Table" provided in the text, the maximum experience level is **99**, which requires **13,034,431** experience points.

Half of that maximum experience is **6,517,215.5**.

Looking at the table to find which level reaches or exceeds that number:
*   Level 91 requires 5,902,831 exp.
*   Level 92 requires 6,517,253 exp.

Therefore, you reach half of the maximum experience at **level 92**.
```
The structural similarity of the responds against Q4_E_H with signficant differences in the quant layers is notable, suggesting
the response are both just regurgitated out of the pretrained and /or finetuned latent space with the prompt and solution both
trained into the model.  However Q4_P_H was found to be noticeably stronger on other reasoning prompts.

Vision:

The model was tested in vision mode on a couple pretty tough bird ID image and found to exhibit poor performance in both think and nonthink
mode, but did at least consider the correct bird on one of the test images.  As a comparision gemma3 27B went 1 for 2 and Qwen3 27B completely aces these
tough (quite blurry images of a small bird) ID tests.  The model did a great job on some text based image prompts though.

Q4_P_H was also terrible at images and also tested against IDing various plants in a wetlands environment which it completely flopped
compared to Qwen3.5 / 3.6 models which completely ace the bird and plant ID prompts.  The source of this poor vision performance is unknown, it
may be either a badly trained model or a bad multimodal implemention or a combination of both.

Code:

The Q4_E_H and Q6_K_H models do a good job with code gen across a small set of eval promps using greedy sampling.
Both quants do not do a good job one of the test prompts.  Q4_P_H did a pretty good job on all the code test prompts
but does poorly in multiturn prompting to tweak generated code according to provided instructions, it is not in the same
league as the Qwen3 and 3.5 models for code gen.

Llama.cpp inference/isssues:

Llama.cpp minimum version to run gemma-4-31B-it should be b8648 and above due to correction of the Gemma 4 tokenizer.

The model cannot compute valid perplexity due to the instruct tune forcing it to generate

```
<|channel>thought
````

as assitant gen independent of previous prompt contents.  To work around this problem a modifed perplexity is
computed by overwriting the beginning of the perplexity chunk contents with the forced assistent gen as follows:

```
      # chunk is a string of text to eval perplexity on
      injects='model\n<|channel>thought\n<channel|>'
      chunk="${injects}${chunk:${#injects}}"
```

logprobs are skipped over the beginning part of the perplexity prompt using a modified llama.cpp downstream
server to compute perplexity.  Discussion at: https://github.com/ggml-org/llama.cpp/issues/21388#issuecomment-4185889930

Benchmarks:

A full set of both math and vision benchmarks for the model will eventually be given here: https://huggingface.co/spaces/steampunque/benchlm

## Download the file from below:
| Link | Type | Size/e9 B | Notes |
|------|------|-----------|-------|
| [gemma-4-26B-A4B-it.Q4_E_H.gguf](https://huggingface.co/steampunque/gemma-4-26B-A4B-it-MP-GGUF/resolve/main/gemma-4-26B-A4B-it.Q4_E_H.gguf) | Q4_E_H | 16.1e9 B | 0.7B smaller than Q4_K_M
| [gemma-4-26B-A4B-it.Q4_P_H.gguf](https://huggingface.co/steampunque/gemma-4-26B-A4B-it-MP-GGUF/resolve/main/gemma-4-26B-A4B-it.Q4_P_H.gguf) | Q4_P_H | 16.7e9 B | ~ Q4_K_M size, tensors padded to enable K quant use on all layers
| [gemma-4-26B-A4B-it.Q6_K_H.gguf](https://huggingface.co/steampunque/gemma-4-26B-A4B-it-MP-GGUF/resolve/main/gemma-4-26B-A4B-it.Q6_K_H.gguf) | Q6_K_H | 17.5e9 B | 5B smaller than Q6_K |
| [gemma-4-26B-A4B-it.mmproj.gguf](https://huggingface.co/steampunque/gemma-4-26B-A4B-it-MP-GGUF/resolve/main/gemma-4-26B-A4B-it.mmproj.gguf) | F16 | 1.2e9 B | multimedia projector |
 
A discussion thread about the hybrid layer quant approach can be found here on the llama.cpp git repository:

https://github.com/ggml-org/llama.cpp/discussions/13040