---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen3.8-27B
- harness-hub/Qwen3.8-27B-pi-adapted
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- qwen3_5
- coding
- pi
- splash
- community-quant
---

# Qwen3.8-27B Pi-adapted — Splash Q4_K_M

An unofficial community **merged Q4_K_M GGUF**, tested with **Splash 1.7.0 on Apple Silicon** and the **Pi coding agent 1.5.2**. The published harness-hub LoRA is merged into its declared Qwen BF16 base before quantization. No additional training was performed here. Credit for the base model and adapter belongs to their respective upstream authors.

This is useful when serving the adapter with an engine that does not load LoRA at runtime. It is a **text-only** release: no vision projector or draft weights are bundled. Other GGUF engines were not tested with this artifact.

| File | Size | SHA-256 |
|---|---:|---|
| `Qwen3.8-27B-pi-adapted-Q4_K_M.gguf` | 16,810,714,560 bytes / 15.66 GiB | `30fb1fbafc97e4a2ef1512fa98d1b85d09346ea58d79aca12e98dbe829dca28d` |

## Upstream sources and conversion

- Base: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`.
- Adapter: [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted), revision `e786f8491742268f9b2bc729697b13bfac26d4ad`.
- CPU PEFT safe merge, processed one base shard at a time; BF16 GGUF export; Q4_K_M quantization without an importance matrix.
- llama.cpp converter revision `52975342ff6b85127ff31930d39e36725fd08809`; llama.cpp quantizer package version `0.7.0`.
- **Splash-specific detail:** GDN QKV/gate and beta/alpha projections remain separate. The pinned converter normally concatenates them; our small wrapper disables that concatenation while retaining upstream head reordering. Splash 1.7 rejected the earlier fused layout. This final artifact has 866 tensors and passed its native shape check.

See [provenance.json](provenance.json), [LICENSE](LICENSE), [NOTICE](NOTICE) and the [conversion recipe](reproduction/README.md). This is a derivative conversion, not an official Qwen, harness-hub, or Splash release.

## Run with Splash 1.7

Use an Apple Silicon Mac supported by Splash 1.7. The measured machine was an M5 Max with 128 GiB unified memory and macOS 26.7.3. The example assumes the `hf` and **Splash 1.7** CLIs are already installed. It does not use an older global Splash installation.

```sh
hf download kesslerIO/Qwen3.8-27B-pi-adapted-Splash-Q4_K_M Qwen3.8-27B-pi-adapted-Q4_K_M.gguf --local-dir ./pi-q4
hf download incoai/Qwen3.8-27B-DFlash2 \
  --revision 015e795645c74b1a0eeef3b570031fb62e769bc5 \
  --include config.json model.safetensors --local-dir ./pi-draft

splash serve \
  --model local/qwen-pi-adapted-q4 \
  --model-dir ./pi-q4/Qwen3.8-27B-pi-adapted-Q4_K_M.gguf \
  --draft-model ./pi-draft --language-only \
  --host 127.0.0.1 --port 8111 \
  --max-memory 32G --max-context 256K --reasoning-effort none
```

Wait for Ready. `256K` means **262,144 tokens**. This is a configured ceiling; a full-window prompt was not stress-tested. The 32 GiB budget includes target weights, draft, GPU KV/state and buffers; macOS and Python memory are additional. It is not a claim that a Mac with 32 GiB total RAM can comfortably run this configuration.

### Pi configuration

Merge this provider into `~/.pi/agent/models.json`, preserving existing providers:

```json
{
  "providers": {
    "splash-pi": {
      "baseUrl": "http://127.0.0.1:8111/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [{
        "id": "local/qwen-pi-adapted-q4",
        "name": "Qwen3.8 27B Pi adapted Q4",
        "reasoning": false,
        "input": ["text"],
        "contextWindow": 262144,
        "maxTokens": 4096,
        "samplingParams": {"temperature": 0.8},
        "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}
      }]
    }
  }
}
```

```sh
pi --provider splash-pi --model local/qwen-pi-adapted-q4 --thinking off
```

Keep thinking off, as recommended by the adapter authors. Pi 1.5.2 omits chat-template kwargs for a non-reasoning model, so our tested server independently enforced `reasoning_effort: none`; do not rely on a client-only template setting.

## Our local coding benchmarks

**Observed improvement on the broader suite, but no statistically established reliability superiority.** These are our authored local Python repair tasks, not SWE-bench or Terminal-Bench results and not the adapter authors' release scores. Read the failures as well as the successes.

![Local coding results on an M5 Max: adapted Q4 completed 13 of 24 attempts versus base 7 of 24; functional checks passed 14 versus 7. All failures remain counted, and reliability superiority is inconclusive.](assets/coding-quality-modern.png)

[Download PNG](assets/coding-quality-modern.png) · [Download SVG](assets/coding-quality-modern.svg). The outlined attempt passed code checks but did not finish within the task deadline. All scored attempts are counted.

[Detailed timing and tool-call chart](assets/coding-benchmark.png) · [SVG](assets/coding-benchmark.svg) · [Plotting source](benchmarks/plot_broad.py). Failure-penalized time is an accounting measure, separate from actual elapsed time.

### Broader suite: 12 tasks × 2 repeats × 2 models = 48 runs

Measured 2026-11-18 on the same M5 Max/128 GiB Mac. Both arms used Q4_K_M, the same tokenizer/template, Splash/draft, Pi, thinking off, temperature 0.8, 4096 output tokens/request and a 262,144-token configured ceiling. Each attempt had 300 seconds and 60 attempted tool calls; shell commands were capped at 90 seconds. Graders were hidden from the agent, frozen before scoring, and replayed independently afterward. All 48 replay outcomes matched.

| Outcome | Base Q4 | Pi-adapted Q4 |
|---|---:|---:|
| Functional passes | 7/24 | 14/24 |
| Usable completions: correct and finished within limits | 7/24 | 13/24 |
| Actual agent time, all scored runs | 47m 34s | 42m 27s |
| Failure-penalized time: 300 seconds per unsuccessful run | 95m 45s | 74m 35s |
| Attempted tool calls | 1,141 | 425 |

Adapted used **62.8% fewer tool calls**, **10.8% less actual agent time**, and **22.1% less failure-penalized time**. These totals include unsuccessful attempts. The penalty prevents fast incorrect answers receiving a speed benefit; it is an accounting convention, not actual elapsed time.

Distinct tasks, not repeats, were the statistical comparison units. Adapted completed both repeats on 5/12 tasks; base on 1/12. Four discordant tasks favored adapted and none favored base; exact paired McNemar **p=0.125**. This does **not** meet the predeclared p<0.05 reliability criterion.

Mean paired task-level failure-penalized time difference (adapted minus base) was **−52.9 seconds**, with a 95% paired-task percentile bootstrap interval **[−102.4, −3.4] seconds**, 10,000 resamples and seed 20261118. This met our predeclared speed criterion (interval excludes zero and at least 20% improvement). It reflects more successful completions under the fixed budget, not proven faster token generation.

On the **six jointly successful pairs**, base took **8m 51s**, adapted **9m 00s**: adapted was **1.7% slower**, with 91 versus 189 tool calls. This selected subset is secondary and excludes failed pairs.

Both models failed file synchronization, unified patch application and layered configuration in both repeats. Base hit the tool-call limit 14 times. Adapted had ten incorrect normal exits and one task timeout; its timed-out Unicode submission was functionally correct but the agent did not finish. Full experiment elapsed was **95m 01s**, including startup, grading and infrastructure recovery.

**Interruption caveat:** an infrastructure restart interrupted one unfinished base archive attempt after 16 completed results. Completed outcomes were preserved; that unfinished EPIPE attempt was replaced once. Zero model failures were retried. The remainder of base repeat one used a fresh warmed server, changing cache history and limiting timing comparability.

[Full broader report](benchmarks/RESULTS-BROAD.md) · [per-run results](benchmarks/results-broad.json) · [frozen protocol](benchmarks/evaluation/broad-v1/PROTOCOL.md) · [benchmark files and reproduction notes](benchmarks/README.md).

### Earlier trials: mixed results, different limits

These were exploratory suites with different tasks/budgets; **do not pool them** with the 48-run suite.

| Suite | Base usable completions | Adapted usable completions | Important limitation |
|---|---:|---:|---|
| Three small tasks, two repeats/model; 12 runs | 4/6 | 3/6 | CSV function-return contract was ambiguous; one timing was a lower bound after a harness cleanup error. |
| Three harder tasks, three repeats/model; 18 runs | 8/9 | 8/9 | All 18 submitted modules passed functional checks. Base looped at the call cap once; adapted deadlocked in its own SQLite concurrency test once. |

In the 18-run suite, base actual agent time totaled **9m 08s**, adapted **25m 32s**. The adapted hang reached the old 20-minute limit and remains counted. On the seven jointly successful pairs it used about 40% fewer calls and 21% less time; that selected subset does not erase the failure. [Small trial](benchmarks/RESULTS.md) · [18-run report](benchmarks/RESULTS-EXTENDED.md).

### What this does and does not show

The adapted Q4 is a promising local coding choice on this setup. It still failed 11/24 usable attempts on the broader suite. Twelve authored Python tasks, unseeded model sampling, varying host load, cache reuse and the restart limit inference. There is no established universal coding-quality, maintainability, security, other-language or full-262K advantage. Q8 was not generated or compared. This quantization does not inherit the upstream BF16 evaluation scores by assumption.

## 32K throughput with the 262K ceiling configured

![Splash speed and memory on an M5 Max: cold prefill 796 tokens per second, decode 119 tokens per second, and 19.5 GiB GPU allocation after the 32K probe under a 32 GiB budget. Cached results are shown separately; the full 262K window was not tested.](assets/speed-memory-modern.png)

[Download PNG](assets/speed-memory-modern.png) · [Download SVG](assets/speed-memory-modern.svg). GPU allocation is not total process memory; macOS and Python use additional memory.

One synthetic-code cold/repeat pair, 32,768 actual input tokens and 512 generated output tokens, Q4_K_M/DFlash2, temperature 0.8 and thinking off. Existing development workloads remained active.

| Metric | Cold prompt | Exact repeat |
|---|---:|---:|
| Reused input tokens | 0 | 32,736 |
| Native time to first token | 41.175 s | 83.5 ms |
| Cold prefill | 795.8 tokens/s | cache hit; not cold prefill |
| Native decode | 119.3 tokens/s | 122.8 tokens/s |
| Client wall time | 45.463 s | 4.247 s |

A cached prompt processes only 32 new input tokens here; dividing all 32,768 tokens by its latency would misrepresent prefill throughput. This is a single throughput probe, not repository coding speed or a full-window stress test. [Details](benchmarks/PERFORMANCE.md) · [machine-readable timings](benchmarks/performance-32k.json).
