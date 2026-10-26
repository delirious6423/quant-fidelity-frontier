---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - moe
  - bubo
---

# Halowin Bubo: Qwen3.8 Flash Next Huihui (lower memory)

A build of Qwen3.8 Flash Next for [Halowin Bubo](https://github.com/haloforge-labs/bubo) 0.1.1, for AMD Strix Halo laptops on Windows. At the default context of 200,000 tokens, Bubo estimates about 66 GiB of GPU memory for this model (standard build: about 91 GiB). That is about 25 GiB less.

It reads prompts at about 0.88 to 0.92 of the speed of the standard build. It writes code and agent text at about 0.75 to 0.9 of that speed, and prose at about the same speed (Bubo 0.1.1, about 70 W; ranges from one run).

**96 GB machines:** use this model. Set Variable Graphics Memory (AMD Software: Adrenalin Edition > Performance > Tuning) to the largest value offered (72 GB if available, at least 64 GB), then restart Windows. At 72 GB, Bubo uses the default 200,000-token context for one session. At 64 GB it does too when about 16 GB of system memory is free, and picks a smaller context when less is free. These values are calculated: Bubo has not yet run on a 96 GB machine.

## What is in it

This is Huihui's abliterated Qwen3.8 Flash Next with smaller expert weights. Its dense tensors come from Huihui, and its expert tensors come from ISTA-DASLab's compressed version of Qwen3.8 Flash Next. It is compatible with Bubo's decode twin and draft head: it uses the same decode twin, draft head and MTP predictor files as the standard Huihui build (the same files, with the same SHA-256 values), so drafting works the same way in Bubo.

On a refusal check (100 AdvBench requests, 20 October 2026) this model refused none, the same as the standard model. It uses 96 of the 101 tensors that Huihui's abliteration changed; the 5 changed expert layers come from ISTA's file instead.

- The dense tensors come from [Huihui-Qwen3.8-Flash-Next-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) (revision `73e9eb7c`).
- The expert tensors come from [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by ISTA-DASLab (revision `ed59f920`). They use mixed low-bit formats.
- HaloForge Labs composed the two sets into one set of files. The composer copied the tensors unchanged. `compose-manifest.json` lists every tensor, where it came from, and its checksum.
- The file metadata is not the same as in the Huihui files. The name, the quantizer (HaloForge Labs), the repository address and the file type are new, and keys that name the two sources are added.

| File | Size |
|---|---:|
| `Qwen3.8-Flash-Next-Huihui-LowMemory-00001-of-00004.gguf` | 10,947,264 B |
| `Qwen3.8-Flash-Next-Huihui-LowMemory-00002-of-00004.gguf` | 42,270,514,496 B |
| `Qwen3.8-Flash-Next-Huihui-LowMemory-00003-of-00004.gguf` | 33,568,858,464 B |
| `Qwen3.8-Flash-Next-Huihui-LowMemory-00004-of-00004.gguf` | 8,758,754,880 B |
| `compose-manifest.json` | tensor list |

Keep the four shards in one folder. The SHA-256 value of each file is on its page.

## Quality

On a 35-question reasoning check (GPQA Diamond, AIME 2026, HMMT 2026, IMO 2026; 25 October 2026), this model answered 29 correctly, the same as the standard Q4 model.

Its expert weights are the expert tensors of ISTA-DASLab's GSQ-RCO IQ3_S file (arXiv:2604.18556 and arXiv:2605.00649). For that file, ISTA's card reports a match with the base model on AIME25 and a result within 0.57 points on LiveCodeBench v6. This model uses Huihui's abliterated dense weights in place of ISTA's.

## Use

Use Bubo 0.1.1 or newer. Bubo can download this model. Bubo can also use files you already have.

To download it, open **Models**, then **Discover**, then **Download** in Bubo. Or use this command:

```powershell
bubo prepare model --catalogue-id huihui-qwen38-flash-next-lowmemory-v1
```

To use local files, put the eight model files in `ROOT\models\huihui-qwen38-flash-next-lowmemory-v1`. Use the four shard names in the table above. Keep the four auxiliary files in the `mtp`, `mmproj`, `twin` and `draft_head` folders. Run this command:

```powershell
bubo prepare model --catalogue-id huihui-qwen38-flash-next-lowmemory-v1 --output ROOT --offline --yes
```

Bubo reuses files that match its hashes and writes `bubo-package.json` in the model folder. To run the model, select it in the app, or use this command:

```powershell
bubo serve --manifest <folder>\bubo-package.json
```

Replace `<folder>` with the model folder.

The four shards need 84,609,075,104 bytes (79 GiB, rounded). The four auxiliary files need 5,282,481,216 bytes:

| File | Bytes |
|---|---:|
| `mtp\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2,786,568,256 |
| `mmproj\mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 907,542,944 |
| `twin\iq4-nofullattn-nohc.gguf` | 1,230,788,768 |
| `draft_head\draft-out-q4k.gguf` | 357,581,248 |

The eight model files need 89,891,556,320 bytes in total. Bubo also writes `bubo-package.json`, the licence, the notices and the preparation state.

Bubo estimates the memory this model needs before it starts.

## Licence

Qwen Community License 1.0, with the notice chain Qwen, Huihui, Unsloth, ISTA-DASLab, HaloForge Labs. See `LICENSE` and `NOTICE`. Keep both files with any copy. Not MIT.

The ISTA-DASLab model card says its weights inherit the licence of the base model. The base model is Qwen3.8 Flash Next.

The dense tensors come from an abliterated model. It can produce sensitive content. You are responsible for your use.

If you use the expert tensors, ISTA-DASLab asks you to cite the GSQ paper (arXiv:2604.18556) and the RCO paper (arXiv:2605.00649).
