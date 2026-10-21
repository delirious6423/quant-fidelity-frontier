---
base_model:
- Qwen/Qwen3.8-27B
---

Feel free to test the two versions of the Qwen 3.8 heretication and share your thoughts about how you find them in discussions. Do let me know if you need other quants.

Standard testing so far:

```
LM-Eval Testing @ Q4_K_M

RC1 ([Trial 377] Refusals:  2/101, KL divergence: 0.0759):
|Tasks|Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----|------:|------|-----:|--------|---|-----:|---|-----:|
|piqa |      1|none  |     0|acc     |↑  |0.4951|±  |0.0117|
|     |       |none  |     0|acc_norm|↑  |0.4951|±  |0.0117|
|Tasks|Version|Filter|n-shot|Metric|   |Value|   |Stderr|
|-----|------:|------|-----:|------|---|----:|---|-----:|
|copa |      1|none  |     0|acc   |↑  | 0.55|±  |  0.05|
|  Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|      1|none  |     0|acc   |↑  |0.4957|±  |0.0141|

RC2 ([Trial 318] Refusals:  6/101, KL divergence: 0.0577):
|Tasks|Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----|------:|------|-----:|--------|---|-----:|---|-----:|
|piqa |      1|none  |     0|acc     |↑  |0.4951|±  |0.0117|
|     |       |none  |     0|acc_norm|↑  |0.4951|±  |0.0117|
|Tasks|Version|Filter|n-shot|Metric|   |Value|   |Stderr|
|-----|------:|------|-----:|------|---|----:|---|-----:|
|copa |      1|none  |     0|acc   |↑  | 0.55|±  |  0.05|
|  Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|      1|none  |     0|acc   |↑  |0.4957|±  |0.0141|
```

In my short refusal testing in both reasoning and non-reasoning modes, I noticed that RC2 sometimes referred to a "policy" but was quick to dismiss it, while I'm yet to see RC1 mention anything like that. Due to lower ablation weights and tied benchmarks, I'm considering RC1 to be the release despite slightly higher KLD. Let me know your thoughts and, should you wish to test, consider adding `--seed 1234` to your llama-server to ensure better comparability. (The seed number can be anything but needs to be consistent throughout testing.)