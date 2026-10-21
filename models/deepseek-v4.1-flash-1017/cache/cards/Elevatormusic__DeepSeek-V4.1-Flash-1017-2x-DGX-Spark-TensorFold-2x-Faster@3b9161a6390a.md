---
license: apache-2.0
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- recipe
- tensorfold
- exl3
- exllamav3
- deepseek
- deepseek_v41
- dgx-spark
- gb10
- tensor-parallel
- tool-calling
- structured-output
---

> **Recipe card only. This repo holds no weights.** The weights are in two other repos (see [Get the weights](#get-the-weights)). The recipe is on GitHub: [Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark).

# DeepSeek V4.1 Flash on two DGX Spark

**Run DeepSeek's V4.1 Flash model on two NVIDIA DGX Spark computers with about 2× the decode and prefill speed of the standard vLLM setup, ready for AI agents.**

Measured on 17 and 18 October 2026 on one pair of DGX Spark. Every number on this page comes from a saved run in the [GitHub recipe](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark). The engine is [my TensorFold fork](https://github.com/Elevatormusic/TensorFold) (branch `deepseek-v41-tp2`).

The speed results and the full 1M-token request test below apply to the base recipe. They do not apply to the optional uncensored BF16 overrides.

---

## At a glance

| | Standard setup (vLLM kit) | This recipe | |
|---|---:|---:|---|
| Decode, prose (1 request) | 27.0 tok/s | **56.8 tok/s** | **2.1× faster** |
| Decode, code (1 request) | 39.9 tok/s | **92.2 tok/s** | **2.3× faster** |
| Prefill, 32K-token prompt | 925 tok/s | **2,165 tok/s** | **2.3× faster** |
| Needle-in-a-haystack, 64K-token prompt (request time) | 80.4 s | **35.9 s** | **2.2× faster** |
| Max context length, as each setup ships | 262,144 tokens | **1,048,576 tokens** | **4× longer** |

*Prefill* is the speed at which the server reads the prompt; it sets the time to first token (TTFT). *Decode* is the speed at which it writes the reply. A *token* is a piece of a word: 100 tokens are about 75 English words.

Both setups ran on the same two computers, with the same model files. The prefill speed of this recipe is the median of 5 runs with cold prompts (new random prompts, no prefix-cache hit); the range is 2,065 to 2,201 tok/s. The vLLM prefill speed comes from the test suite's 32K prompt, which can include a prefix-cache hit; that would favor vLLM. In a needle test with a 1,035,020-token prompt, the model found the needle; the TTFT was 13.6 minutes.

Prefill with cold prompts (bertholomus's kit_bench client, median of 5 runs):

| Prompt length | 8K tokens | 32K | 64K | 128K | 1,035,020 |
|---|---:|---:|---:|---:|---:|
| Prefill (tok/s) | 2,179 | 2,165 | 2,037 | 1,961 | 1,266 (1 run) |

The kit_bench numbers use a small-vocabulary prompt (random words from a list of about 150 words). Random natural text gives 1,963 tok/s at 32K on release 1.1.0 (median of 3, cold prompt).

---

## What you get

- **About 2× the decode speed** of the standard vLLM setup: 56.8 vs 27.0 tok/s on prose, 92.2 vs 39.9 tok/s on code.
- **Fast prefill.** About 2,000 to 2,200 tok/s for prompts up to 64K tokens.
- **Ready for AI agents.** Forced tool calls (`tool_choice` set to `required` or to one tool) and structured output (a JSON schema, a regex or a list of choices).
- **4 concurrent requests.** Up to 4 requests share the two computers. Aggregate decode reaches 140 tok/s on code.
- **Prefix cache.** A second question about the same long document has a TTFT of about 1 s instead of 13 s. Cached prompts also go to disk, so after a restart the TTFT is about 3 s.
- **Fault isolation.** One failed request ends only that request. It does not stop the whole server.
- **1M-token context.** The full 1,048,576-token context window. A needle test with a 1,035,020-token prompt passed.

---

## Get the weights

This recipe uses two sets of files. Both are already public. The recipe downloads them for you at these exact versions.

| Files | Source | Version |
|---|---|---|
| Model weights (EXL3 2.9 bpw, 43 files) | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | `afc7aa0a` |
| Engram tables (files 47 and 48 only) | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `1bae0e10` |

The Engram tables of the 1017 release are new. Do not use the tables of the older V4.1 Flash release.

---

## Optional uncensored BF16 overrides (experimental)

This option adds 54 original dense BF16 matrices from the uncensored source to the base EXL3 model. It replaces `wo_b` and shared-expert `w2` in layers 10 through 36. It is not an all-BF16 checkpoint.

The source is [dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED/tree/89af9bf193575b7eeeb4f1dd85463ec24ea5bb4c), pinned to `89af9bf193575b7eeeb4f1dd85463ec24ea5bb4c`. See [recipe PR #14](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/pull/14) and its [optional recipe instructions](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/00cf8f325fae83ab5805bb1d0a364ae696bde247/docs/UNCENSORED.md). This option needs the [engine change](https://github.com/Elevatormusic/TensorFold/pull/8) at `dccc4a997b7c18fc684781ee333273f8d902c510`. These pull requests are not merged. The default engine image does not support these overrides. The quick start below still uses the base model.

Real replies passed with 8K-token and 64K-token prompts and with four concurrent requests. The server was configured for a 1M-token context, but no full 1M-token request was tested with these overrides. Both nodes logged `NV_ERR_NO_MEMORY` warnings while the API requests completed. No OOM kill was observed. This does not establish that memory failures are fixed.

The 8 bpw override test produced a repetition loop. No decode-speed benefit was established, so BF16 is the documented option. These checks do not establish model quality or an uncensored benchmark score. See the [saved test limits](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/00cf8f325fae83ab5805bb1d0a364ae696bde247/runs/2026-10-19/uncensored-bf16/service-check.json). That summary records an earlier local engine, not GPU qualification of the new engine PR head or preparation wrapper.

---

## What you need

- **Two NVIDIA DGX Spark**, joined by one QSFP cable between their ConnectX-7 network ports.
- **About 500 GB of free disk on each computer.** The model files use 385 GB. The first start uses 110 GB more.
- **About 100 GB of free memory on each computer.** Close other AI programs first.
- **A network setup** with fixed addresses on the cable link and ssh from the first computer to the second. The GitHub recipe has a network guide.
- **Time.** Most of the first setup is the 400 GB download. It takes a few hours on a fast internet link.

The *head* is the computer where you type the commands. The *worker* is the second computer.

---

## Quick start

Run each command on the head.

```bash
git clone https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark
cd DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark
```

1. **Make your settings file.** Then open `.env` and type your addresses and folder paths.

   ```bash
   cp .env.example .env
   ```

2. **Download the model.** This step gets 400 GB, checks every file, and copies it to the worker.

   ```bash
   scripts/download.sh
   ```

3. **Prepare the chat template.** This step gets the template and applies a small fix.

   ```bash
   scripts/patch_template.py --worker
   ```

4. **Build the engine.** This step builds the server software from [my TensorFold fork](https://github.com/Elevatormusic/TensorFold) and copies it to the worker.

   ```bash
   docker/build.sh
   ```

5. **Tune the memory settings (once).** This keeps the model out of swap and stops memory stalls during long prefills, on both computers. If memory runs out, it stops the model, not the computer (this needs the earlyoom package, which DGX OS has). It asks for your password on each computer.

   ```bash
   scripts/host_tuning.sh
   ```

6. **Start the server.** It checks your setup first. The first start takes about 15 minutes. Later starts take 2 to 4 minutes.

   ```bash
   ./start.sh
   ```

7. **Check that it works.** This sends five real requests and shows the result of each.

   ```bash
   ./verify.sh
   ```

To stop the server, run `./stop.sh`. To see if it runs, run `./status.sh`.

---

## Connect your app

The server speaks the OpenAI API. Any app that supports this API can use it.

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8888/v1` |
| Model name | `DeepSeek-V4.1-Flash-1017` |

Thinking is **off** by default. To turn it on, do one of these in your request:

- Set `reasoning_effort` to one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`.
- Set `chat_template_kwargs.enable_thinking` to `true`.

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4.1-Flash-1017",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "reasoning_effort": "medium"
  }'
```

---

## Good to know

- **The first long prompt after a start has a higher TTFT,** about 1 to 2 s more. Later prompts do not.
- **Prefill is a little slower than bertholomus reports.** At 32K tokens this setup prefills at 2,165 tok/s; they publish 2,421 on different computers and the original V4.1 weights. Prefill speed also changes from run to run (64K: 1,840 to 2,120 tok/s).
- **A 1M-token prompt has a TTFT of about 14 minutes** (prefill at 1,266 tok/s).
- **vLLM was tested with 1 request at a time only.** So the concurrency numbers have no vLLM comparison.
- **Living Weights is not available.** That TensorFold feature works only with a different model on one computer.

---

## Credits

This recipe stands on the work of other people. Thank you.

- **[bertholomus](https://github.com/bertholomus) (Bertholomus AI) wrote the fast V4.1 engine that this recipe runs.** It gives the DeepSeek V4.1 model code, the kernels, the DSpark drafting, the Engram support and the fast link between the two computers. All of the speed comes from their work. See their recipe: [bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10).
- **[LibertAIDAI](https://huggingface.co/LibertAIDAI)** made the compressed EXL3 2.9 bpw weights.
- **[DeepSeek](https://huggingface.co/deepseek-ai)** made the model, its Engram tables and the reference prompt format.
- **[turboderp](https://github.com/turboderp-org/exllamav3)** made ExLlamaV3, the EXL3 format.
- **[ashhart](https://github.com/ashhart/TensorFold)** and the TensorFold contributors made TensorFold, the engine framework.
- **willjriley** and **Capicua25x** made fixes to the engine.
- **[MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)** made the vLLM kit that gives the standard-setup numbers, and the chat template.
- **tonyd2wild** made the benchmark method.

What this recipe adds to the engine is five changes: forced tool calls, prompt encoding identical to DeepSeek's reference, structured output, fault isolation, and faster long prefills. The engine changes are in the fork [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold), in pull requests [#1](https://github.com/Elevatormusic/TensorFold/pull/1), [#2](https://github.com/Elevatormusic/TensorFold/pull/2), [#3](https://github.com/Elevatormusic/TensorFold/pull/3) and [#4](https://github.com/Elevatormusic/TensorFold/pull/4).

This recipe is not affiliated with or endorsed by DeepSeek, NVIDIA, LibertAIDAI, the TensorFold authors, bertholomus or MiaAI-Lab.

---

## Licence

The scripts, tests and documents in the recipe are Apache-2.0. The model, the chat template and the engine keep their own licences:

| Part | Licence |
|---|---|
| Recipe scripts, tests and documents | Apache-2.0 |
| DeepSeek-V4.1-Flash-1017 (the model) | MIT |
| LibertAIDAI EXL3 2.9 bpw weights | MIT |
| Optional dealignai BF16 overrides | [DeepSeek V4.1 Flash 1017 licence](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE) |
| TensorFold and the V4.1 engine | Apache-2.0 |
| MiaAI-Lab vLLM kit and chat template | AGPL-3.0 |
