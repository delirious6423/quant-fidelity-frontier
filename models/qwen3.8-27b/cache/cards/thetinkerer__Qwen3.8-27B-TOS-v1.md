---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen3.8-27B
---

## Qwen3.8 27B Instruct finetuned and modifed for HolyC/TempleOS API knowledge

This is an alpha release v1, your mileage will be limited.  The purpose is HolyC coding and code completion, but some general knowledge questions may also yield results.  Obviously the amount of TempleOS related training data is severly limited in scope so don't expect miracles.  This is trained on only public domain TempleOS code and not any 3rd party or TinkerOS/ZealOS/etc code.  This model does still support thinking and tool usage so it should be suitable to use inside your IDE to read/write/modify HolyC files.  Some non-English tokens and vision capabilities have been removed to help remove model size and some HolyC/TempleOS specific tokens have been added in their place to improve performance.

Not enough VRAM? see 4B and 9B variants: https://huggingface.co/collections/thetinkerer/templeos-v1-finetune-models

# Approximate TempleOS v1 Model Series Performance

![image](https://cdn-uploads.huggingface.co/production/uploads/678d51675e6af698f2d979ff/xVwGcN19a4Z94a0ES8a3l.png)

# LM Studio setup

 - Copy the GGUF files to a directory structure like:
   - Linux:  `/home/youruser/.lmstudio/models/templeos/qwen38_TOS_v1_Q3_K_M/qwen38-27b-instruct-v1-Q3_K_M.gguf`
   - Windows: `C:\Users\[Your Username]\LM Studio\models\templeos\qwen38_TOS_v1_Q3_K_M\qwen38-27b-instruct-v1-Q3_K_M.gguf`
 - Setup the System Prompt using the recommended prompt from below in the model Inference tab System Prompt:
![image](https://cdn-uploads.huggingface.co/production/uploads/678d51675e6af698f2d979ff/2EQ5mM6mcxdo7BPAOSbls.png)
 - Set a reasoning limit so it doesn't get stuck thinking forever about things like the differences between C and HolyC
 ![image](https://cdn-uploads.huggingface.co/production/uploads/678d51675e6af698f2d979ff/Fh_pB6lWsHBerabGseRm4.png)

# Recommended minimal coding system prompt:
```
You are answering questions about HolyC, the programming language of Terry Davis's TempleOS operating system. Use real TempleOS API names (MAlloc/Free, StrCpy/StrLen, FileRead/FileWrite, ...). Never use any of the following: enums, function macros, prefix-casting, C-standard-library names, typedef, ternary operators, or continue.
```

# Recommended minimal knowledge question system prompt:
```
You are answering questions about TempleOS (an operating system) and HolyC (its programming language), both created single-handedly by Terry A. Davis. Answer conceptual, historical, and "why does it work this way" questions in plain English -- this is about understanding the project, not writing code. TempleOS is a self-hosting, single-user, single-address-space OS that runs entirely in ring 0 with no memory protection between programs; Terry Davis described it as a temple to God, comparable to the Biblical instructions for building Solomon's Temple, and treated many of its constraints (640x480 16-color graphics, no networking, no third-party libraries, a 100,000-line-of-code ceiling) as fixed design principles rather than technical limitations to eventually lift. Answer only from what is actually documented about the project; don't speculate about details that aren't established.
```

# Say a prayer, try a question (and maybe profit?):

![image](https://cdn-uploads.huggingface.co/production/uploads/678d51675e6af698f2d979ff/yTjLSJX-pa4numM6xt1oV.png)

# Note: lower quantizations like Q3 will run faster and use less VRAM, however the output quality likely will be worse than Q4/Q5

See this chart to get an idea how quantization effects performance and VRAM requirements:

![image](https://cdn-uploads.huggingface.co/production/uploads/678d51675e6af698f2d979ff/u7tsFOR1PEzv5oGRb2obQ.png)

Original chart source:
https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/README.md
