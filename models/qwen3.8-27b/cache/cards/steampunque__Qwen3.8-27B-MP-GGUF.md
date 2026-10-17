---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- Qwen3.8 27B
- GGUF
- quantized
- 4-bit
- mixed precision
---

## Mixed Precision GGUF layer quantization of Qwen3.8-27B by Qwen

Original model: https://huggingface.co/Qwen/Qwen3.8-27B

The hybrid quant employs different quantization levels on a per layer basis to enable
both high performance and small file size at the same time. The quants employed are all K to avoid
slow CPU or older GPU processing of IQ quants.  An extended layer definition E quant Q4_E_H for the model is defined as follows (updated 8/22/2026):

```
   LAYER_TYPES='[
   ["A","attn","Q","attn_q","K","attn_k","V","attn_v","O","attn_o","S","ssm","F","ffn","G","ffn_g","U","ffn_u","D","ffn_d"],
   ["MAP","VOSD"],
   [0 ,"Q5_K_6668"],[1 ,"Q5_K_6665"],[2 ,"Q4_K_6555"],[3 ,"Q4_K_6555"],
   [4 ,"Q4_K_5555"],[5 ,"Q4_K_5555"],[6 ,"Q4_K_4444"],[7 ,"Q4_K_5554"],
   [8 ,"Q4_K_4444"],[9 ,"Q4_K_4444"],[10,"Q4_K_4444"],[11,"Q4_K_5554"],
   [12,"Q4_K_4444"],[13,"Q4_K_4444"],[14,"Q4_K_4444"],[15,"Q4_K_5554"],
   [16,"Q4_K_4444"],[17,"Q4_K_4444"],[18,"Q4_K_4444"],[19,"Q4_K_5554"],
   [20,"Q4_K_4444"],[21,"Q4_K_4444"],[22,"Q4_K_4444"],[23,"Q4_K_5554"],
   [24,"Q4_K_4444"],[25,"Q4_K_4444"],[26,"Q4_K_4444"],[27,"Q4_K_5554"],
   [28,"Q4_K_5444"],[29,"Q4_K_5444"],[30,"Q4_K_5444"],[31,"Q4_K_5554"],
   [32,"Q4_K_5454"],[33,"Q4_K_5454"],[34,"Q4_K_5454"],[35,"Q4_K_5555"],
   [36,"Q4_K_5454"],[37,"Q4_K_5454"],[38,"Q4_K_5454"],[39,"Q4_K_5555"],
   [40,"Q4_K_5554"],[41,"Q4_K_5554"],[42,"Q4_K_5554"],[43,"Q4_K_5555"],
   [44,"Q4_K_5554"],[45,"Q4_K_5554"],[46,"Q4_K_5554"],[47,"Q4_K_5555"],
   [48,"Q4_K_5554"],[49,"Q4_K_5554"],[50,"Q4_K_5554"],[51,"Q4_K_5555"],
   [52,"Q4_K_5555"],[53,"Q4_K_5555"],[54,"Q4_K_5556"],[55,"Q4_K_6666"],
   [56,"Q4_K_5556"],[57,"Q4_K_5556"],[58,"Q4_K_5556"],[59,"Q5_K_5566"],
   [60,"Q5_K_6666"],[61,"Q5_K_6666"],[62,"Q5_K_6668"],[63,"Q6_K_8666"],
   [64,"Q4_K_6554"]
   ]'
   FLAGS="--token-embedding-type Q4_K --output-tensor-type Q6_K --layer-types-high"
```

The quant was optimized for approximately Q4_K_M bpw with strong performance across a curated set of reasoning test prompts, scoring solid
100% across the test set with greedy deterministic sampling, exhibiting good common sense, and not overthinking or infinite rep loops on any of the prompts.
The quant includes layer 64 nextn MTP layer.  If not using MTP the loader will give warning messages about
unused tensors on layer 64 but the model will run correctly.  The quant was updated 8/22/2026 for minium quant across layers at Q4_K
to reduce entropy loss and provide noticeable performance improvement compared to the first Q4_E_H quant.

Comparison:

Quant  |  size   |  PPL  |   Comment
-------|---------|-------|-----------
Q4_K_M | 16.8e9  | 7.2   | Q4_K_M with default embedding and output + MTP layer (~0.3B)
Q4_E_H | 16.8e9  | 7.3   | Hybrid quant with Q4_K embedding Q6_K output + MTP layer (~0.3B)

of note is a significant decrease in PPL compared to Qwen 3.5/3.6 series, which may imply this version of model was
trained heavier on english language processing or just stronger pre training on wikitext so it memorizes it better.

Usage:

Qwen3.8-27B is a vision capable dense RL model. It can be used together with its multimedia projector layers to process images and text inputs
and generate text outputs. The mmproj file is made available in this repository.

Due to the attention scheme used in the model, straightforward speculation approaches cannot be used. Later versions of llama.cpp as of 4/23/2026 attempt to
circumvent this problem however efficiency will not be good so use of speculation prior to Qwen3.5 MTP patches is not recommended.
As of llama.cpp b9180 MTP support for the model was added to upstream and may be experimented with by turning on the MTP functionality if using
upstream llama.cpp.  MTP functionality was not evaluated.

On a 2x 4070 setup (1 RPC) approx performance with no MTP and no mmproj loaded is:

Q      |   QKV    |  NKV   |  gen tps
-------|----------|--------|--------
Q4_E_H | F16      |  90k +|  24  
Q4_E_H | Q8_0     |  140k +|  23

The model appears to be trained to decide itself whether to do a think block or not.  If it thinks a prompt is easy it just prefixes its
response with an empty think block.  When it decided to do a think block it did not overthink or fall into rep loops (common RL model problem) on any of the test
prompts.  To for the model to skip thinking think start and think stop tokens first thing after assistant prompt:

```
THINK_START="<think>\n"
THINK_STOP="\n</think>\n\n"
```

If the model doesnt feel like doing thinking on a given prompt it will automatically do this.  To force the model into
a think block inject a THINK_START bootstrap following the assistant prompt:

```
"<think>\n"
```

All test prompts below use greedy deterministic sampling and 2x 4070 + RPC + cuda backend.

On one IQ test prompt the model decided to wing it without a think block and it failed, but by forcing a think block
it came up with the right answer.

The model has common sense on the recent dumb washcar prompt in both think and forced nothink mode:

washcar.txt:
```
I want to wash my car. The car wash is 50 meters away. Should I walk or drive?
```

```
lm washcar.txt 
<think>
The user is asking whether they should walk or drive to a car wash that is 50 meters away. Let me think about this practically.

The key point here is: they want to wash their car. The car wash is 50 meters away. 

If they walk, they leave the car behind. Then they'd need to go back to get the car, drive it to the car wash, and then come back. That's a lot of unnecessary travel.

If they drive, they just drive the car 50 meters to the car wash. Simple and direct.

The answer is obviously: drive. You need to get the car TO the car wash. Walking would mean you'd have to make multiple trips. 50 meters is short, but the whole point is to get the car there.

This is a pretty straightforward question, almost a trick question. Let me give a clear, helpful answer.
</think>

**Drive.**

The whole point is to get the car *to* the car wash. If you walk, you'd have to go back, get the car, drive it 50 meters, and then walk back again. Just drive the 50 meters, get it washed, and drive home. Simple.
```

```
NOTHINK=1 lm washcar.txt 
You should **drive**.

Here’s why:
*   **The goal is to wash the car.** If you walk to the car wash, your car will be left at home, and you won’t be able to wash it.
*   **Distance is irrelevant to the method.** Even though 50 meters is a short walk, you need the car to be present at the car wash to get it cleaned.

So, drive your car to the car wash! 🚗💦
```

These responses suggest the prompt and answer were trained into the fine tune of Qwen 3.8.

VISION:

The model went 2 for 2 on a couple tough bird ID images prompts and did very well summarizing a
test text image.

CODE:

The quant was tested across a small set of code gen prompts and found to be very solid in its ability to generate
working programs with think mode enabled.  Code gen was not evaluated with think block forced off.

LONG CONTEXT:

The model passes the long context Qwen3 runescape test prompt running Q8_0 KV: https://thireus.com/REDDIT/Qwen3_Runescape_Massive_Prompt.txt:

```
lm Qwen3_Runescape_Massive_Prompt.txt 
<think>
The user is asking what level you are at when you have half the XP of max level in RuneScape.

Max level is 99, which requires 13,034,431 XP.

Half of that would be 13,034,431 / 2 = 6,517,215.5 XP.

Looking at the experience table:
- Level 92: 6,517,253 XP

So 6,517,215.5 is just barely below level 92 (which requires 6,517,253 XP).

Let me check level 91: 5,902,831 XP

So half of max XP (6,517,215.5) falls between level 91 (5,902,831) and level 92 (6,517,253).

Since 6,517,215.5 < 6,517,253, you would still be at level 91 (you haven't quite reached level 92 yet).

Wait, let me re-read the question. "What level are you at when you get to half the xp of max level?" This could mean: at what level do you have accumulated half the total XP needed for level 99?

Half of 13,034,431 = 6,517,215.5

Looking at the table:
- Level 91: 5,902,831 XP
- Level 92: 6,517,253 XP

6,517,215.5 is between level 91 and 92, so you'd be at level 91 (you haven't quite reached 92 yet).

Actually, wait. Let me reconsider. The question might be asking "at what level do you have half the XP that level 99 requires?" In that case, you'd be at level 91 since you need 6,517,253 to reach level 92, and half of max is 6,517,215.5, which is just under that threshold.

So the answer is level 91.
</think>

Based on the experience table provided in the book, max level (99) requires **13,034,431 XP**. Half of that is **6,517,215.5 XP**.

Looking at the table:
- Level 91 requires 5,902,831 XP
- Level 92 requires 6,517,253 XP

Since 6,517,215.5 is just barely below the 6,517,253 needed for level 92, you would be at **level 91** when you reach half the XP of max level.
```

```
NOTHINK=1 lm Qwen3_Runescape_Massive_Prompt.txt 
Based on the **Skills/Experience Table** provided in the text, here is the breakdown:

1.  **Max Level XP:** The table shows that Level 99 requires **13,034,431** experience points.
2.  **Half of Max Level XP:** Half of 13,034,431 is **6,517,215.5**.
3.  **Finding the Level:** Looking at the table for the experience values closest to this number:
    *   Level 91 requires 5,902,831 XP.
    *   Level 92 requires **6,517,253** XP.

Level 92 is the first level where the required experience exceeds half of the maximum level experience (6,517,253 > 6,517,215.5).

Therefore, you are at **Level 92** when you reach (and slightly exceed) half the XP of the max level.
```

The second NOTHINK response most likely is just pulled straight out of (trained) latent space; i.e. the model has been
trained with both the massive prompt and answer (as opposed to original Qwen3 which did not have this prompt baked
into the mooel training).

Prompt processing slows from about 660 tps at start of prompt to 280 tps at end of ~108k token prompt.

The model passes a 75k token needle in haystack test also.

Benchmarks:

Math, vision, and code benchmarks for the model are given here: https://huggingface.co/spaces/steampunque/benchlm

## Download the file from below:
| Link | Type | Size/e9 B | Notes |
|------|------|-----------|-------|
| [Qwen3.8-27B.Q4_E_H.gguf](https://huggingface.co/steampunque/Qwen3.8-27B-MP-GGUF/resolve/main/Qwen3.8-27B.Q4_E_H.gguf) | Q4_E_H | 16.8e9 B | ~Q4_K_M size, includes ~0.3B MTP layer |
| [Qwen3.8-27B.mmproj.gguf](https://huggingface.co/steampunque/Qwen3.8-27B-MP-GGUF/resolve/main/Qwen3.8-27B.mmproj.gguf) | F16 | 0.93e9 B | multimedia projector |
 
A discussion thread about the hybrid layer quant approach can be found here on the llama.cpp git repository:

https://github.com/ggml-org/llama.cpp/discussions/13040