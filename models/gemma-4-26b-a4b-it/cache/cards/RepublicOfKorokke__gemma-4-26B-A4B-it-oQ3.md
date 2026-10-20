---
library_name: mlx
tags:
- mlx
- oq
- quantized
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
---

# gemma-4-26B-A4B-it-oQ3

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.3.4) mixed-precision quantization.

## Quantization details

- **Model type**: gemma4
- **Bits**: 3
- **Group size**: 64
- **Format**: MLX safetensors

# Benchmark

| Model                                                          | File size | MMLU  | JMMLU | HELLASWAG | ARC_CHALLENGE | GSM8K |
| -------------------------------------------------------------- | --------- | ----- | ----- | --------- | ------------- | ----- |
| gemma-4-26B-A4B-it-oQ3                                         | 11.74 GB  | 79.0% | 75.7% | 86.0%     | 95.0%         | 93.3% |
| gemma-4-26B-A4B-it-oQ3.5                                       | 12.69 GB  | 57.0% | 68.7% | 82.7%     | 90.3%         | 43.0% |
| [gemma-4-26b-a4b-it-oQ4](https://huggingface.co/jdkohler/gemma-4-26b-a4b-it-oQ4)                                         | 14.73 GB  | 82.3% | 74.7% | 87.3%     | 95.7%         | 95.3% |
| gemma-4-26b-a4b-it-4bit                                        | 14.57 GB  | 82.3% | 75.0% | 88.0%     | 96.0%         | 95.3% |
| gemma-4-e4b-it-4bit                                            | 4.89 GB   | 64.3% | 60.3% | 73.7%     | 84.0%         | 85.3% |

## Detail


| Model                                                          | Benchmark     | Accuracy | Correct | Total | Time(s) |
| -------------------------------------------------------------- | ------------- | -------- | ------- | ----- | ------- |
| gemma-4-26B-A4B-it-oQ3                                         | MMLU          | 79.0%    | 237     | 300   | 612.7   |
| gemma-4-26B-A4B-it-oQ3                                         | JMMLU         | 75.7%    | 227     | 300   | 256.4   |
| gemma-4-26B-A4B-it-oQ3                                         | HELLASWAG     | 86.0%    | 258     | 300   | 332.4   |
| gemma-4-26B-A4B-it-oQ3                                         | ARC_CHALLENGE | 95.0%    | 285     | 300   | 244     |
| gemma-4-26B-A4B-it-oQ3                                         | GSM8K         | 93.3%    | 280     | 300   | 1582.7  |
| gemma-4-26B-A4B-it-oQ3                                         | MMLU          | 79.0%    | 237     | 300   | 612.7   |
| gemma-4-26B-A4B-it-oQ3                                         | JMMLU         | 75.7%    | 227     | 300   | 256.4   |
| gemma-4-26B-A4B-it-oQ3                                         | HELLASWAG     | 86.0%    | 258     | 300   | 332.4   |
| gemma-4-26B-A4B-it-oQ3                                         | ARC_CHALLENGE | 95.0%    | 285     | 300   | 244     |
| gemma-4-26B-A4B-it-oQ3                                         | GSM8K         | 93.3%    | 280     | 300   | 1582.7  |
| gemma-4-26B-A4B-it-oQ3.5                                       | MMLU          | 57.0%    | 171     | 300   | 939.2   |
| gemma-4-26B-A4B-it-oQ3.5                                       | JMMLU         | 68.7%    | 206     | 300   | 306.6   |
| gemma-4-26B-A4B-it-oQ3.5                                       | HELLASWAG     | 82.7%    | 248     | 300   | 422.6   |
| gemma-4-26B-A4B-it-oQ3.5                                       | ARC_CHALLENGE | 90.3%    | 271     | 300   | 278.3   |
| gemma-4-26B-A4B-it-oQ3.5                                       | GSM8K         | 43.0%    | 129     | 300   | 2116.4  |
| gemma-4-26b-a4b-it-oQ4                                         | MMLU          | 82.3%    | 247     | 300   | 625     |
| gemma-4-26b-a4b-it-oQ4                                         | JMMLU         | 74.7%    | 224     | 300   | 281.7   |
| gemma-4-26b-a4b-it-oQ4                                         | HELLASWAG     | 87.3%    | 262     | 300   | 350.8   |
| gemma-4-26b-a4b-it-oQ4                                         | ARC_CHALLENGE | 95.7%    | 287     | 300   | 254.7   |
| gemma-4-26b-a4b-it-oQ4                                         | GSM8K         | 95.3%    | 286     | 300   | 1475.4  |
| gemma-4-26b-a4b-it-4bit                                        | MMLU          | 82.3%    | 247     | 300   | 605     |
| gemma-4-26b-a4b-it-4bit                                        | JMMLU         | 75.0%    | 225     | 300   | 260.4   |
| gemma-4-26b-a4b-it-4bit                                        | HELLASWAG     | 88.0%    | 264     | 300   | 321.3   |
| gemma-4-26b-a4b-it-4bit                                        | ARC_CHALLENGE | 96.0%    | 288     | 300   | 239.9   |
| gemma-4-26b-a4b-it-4bit                                        | GSM8K         | 95.3%    | 286     | 300   | 1323.6  || gemma-4-e4b-it-4bit                                            | MMLU          | 64.3%    | 193     | 300   | 418.2   |
| gemma-4-e4b-it-4bit                                            | JMMLU         | 60.3%    | 181     | 300   | 192.3   |
| gemma-4-e4b-it-4bit                                            | HELLASWAG     | 73.7%    | 221     | 300   | 219.3   |
| gemma-4-e4b-it-4bit                                            | ARC_CHALLENGE | 84.0%    | 252     | 300   | 176.5   |
| gemma-4-e4b-it-4bit                                            | GSM8K         | 85.3%    | 256     | 300   | 1486.1  |

### Update (2026-04-07) more sample


| Model                    | Benchmark     | Accuracy | Correct | Total | Time(s) |
| ------------------------ | ------------- | -------- | ------- | ----- | ------- |
| gemma-4-26B-A4B-it-oQ3   | HELLASWAG     | 83.2%    | 416     | 500   | 515.7   |
| gemma-4-26B-A4B-it-oQ3   | ARC_CHALLENGE | 95.4%    | 1118    | 1172  | 898.9   |
| gemma-4-26B-A4B-it-oQ3   | GSM8K         | 92.7%    | 1223    | 1319  | 6127.5  |
| gemma-4-26B-A4B-it-oQ3.5 | HELLASWAG     | 78.6%    | 393     | 500   | 593.2   |
| gemma-4-26B-A4B-it-oQ3.5 | ARC_CHALLENGE | 90.4%    | 1060    | 1172  | 1001.4  |
| gemma-4-26B-A4B-it-oQ3.5 | GSM8K         | 42.4%    | 559     | 1319  | 9116.7  |
| gemma-4-26b-a4b-it-oQ4   | HELLASWAG     | 87.0%    | 435     | 500   | 516.2   |
| gemma-4-26b-a4b-it-oQ4   | ARC_CHALLENGE | 95.7%    | 1121    | 1172  | 884.2   |
| gemma-4-26b-a4b-it-oQ4   | GSM8K         | 93.3%    | 1231    | 1319  | 5774.7  |
| Qwen3.5-35B-A3B-oQ3e     | HELLASWAG     | 92.6%    | 463     | 500   | 458.9   |
| Qwen3.5-35B-A3B-oQ3e     | ARC_CHALLENGE | 96.3%    | 1128    | 1172  | 741.1   |
| Qwen3.5-35B-A3B-oQ3e     | GSM8K         | 93.8%    | 1237    | 1319  | 4632.9  |
