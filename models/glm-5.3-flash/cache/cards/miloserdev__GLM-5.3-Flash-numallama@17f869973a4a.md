---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
library_name: numallama
tags:
- numallama
- cpu
- arm64
- aarch64
- numa
- text-generation
---

# GLM-5.3-Flash-numallama

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), converted for [numallama](https://github.com/miloserdev/numallama): a C inference engine for CPU servers, with NUMA-aware execution across all sockets of multi-socket machines.

Mixture of experts, 320B total / 18B active parameters, hybrid attention (Kimi Delta Attention linear layers + MLA sparse attention with the DSA indexer) and manifold-constrained hyper-connections (mHC). **Text only**: the vision encoder of the original multimodal model is not converted. Needs about 210 GB of RAM. Add `-r 1` to the chat command to turn on reasoning (thinking).

> **These files are not GGUF.** They run only with numallama (they do not work with llama.cpp, Ollama, LM Studio or transformers).

## Files

| File | Weights | Size | Decode speed* |
|---|---|---:|---:|
| `glm-5.3-flash-q4.bin` | Q4 experts + Q8 | 199.8 GB | 17.4 tok/s |
| `tokenizer.bin` | tokenizer | 2.3 MB | |

Converted from the official FP8 checkpoint: the routed experts are asymmetric 4-bit (groups of 32, fp16 scale and min), all other weights (attention, shared experts, dense layers, router, embeddings, output head) are int8 with per-group scales. No importance matrix and no other changes to the weights.
Checked against the transformers reference implementation: logits of the first 4 layers correlate at > 0.999 with the same top-1 token.

\* tokens/s for a single chat, measured on 2 × Ampere Altra Max M128-30 (256 Neoverse N1 cores), both sockets. Other machines will differ.

## Run

Linux. Optimized for ARM64 with NEON dot product (Ampere Altra, AWS Graviton, other Neoverse cores); other CPUs, including x86-64, run portable C code that is not optimized.

```bash
git clone https://github.com/miloserdev/numallama && cd numallama && make
pip install huggingface_hub
hf download miloserdev/GLM-5.3-Flash-numallama --local-dir models/GLM-5.3-Flash-numallama

# chat in the terminal
./runq models/GLM-5.3-Flash-numallama/glm-5.3-flash-q4.bin -z models/GLM-5.3-Flash-numallama/tokenizer.bin -m chat

# or an OpenAI-compatible server on http://127.0.0.1:8080/v1
./runq models/GLM-5.3-Flash-numallama/glm-5.3-flash-q4.bin -z models/GLM-5.3-Flash-numallama/tokenizer.bin -m server -P 8080
```

See the [numallama README](https://github.com/miloserdev/numallama#readme) for all options (context length, sampling, tools, speculative decoding, API details).

## Citation

Original model by Z.AI. If you use it in research, please cite the GLM-5 technical report:

```bibtex
@misc{glm5team2026glm5vibecodingagentic,
      title={GLM-5: from Vibe Coding to Agentic Engineering},
      author={GLM-5-Team and : and Aohan Zeng and Xin Lv and Zhenyu Hou and Zhengxiao Du and Qinkai Zheng and Bin Chen and Da Yin and Chendi Ge and Chenghua Huang and Chengxing Xie and Chenzheng Zhu and Congfeng Yin and Cunxiang Wang and Gengzheng Pan and Hao Zeng and Haoke Zhang and Haoran Wang and Huilong Chen and Jiajie Zhang and Jian Jiao and Jiaqi Guo and Jingsen Wang and Jingzhao Du and Jinzhu Wu and Kedong Wang and Lei Li and Lin Fan and Lucen Zhong and Mingdao Liu and Mingming Zhao and Pengfan Du and Qian Dong and Rui Lu and Shuang-Li and Shulin Cao and Song Liu and Ting Jiang and Xiaodong Chen and Xiaohan Zhang and Xuancheng Huang and Xuezhen Dong and Yabo Xu and Yao Wei and Yifan An and Yilin Niu and Yitong Zhu and Yuanhao Wen and Yukuo Cen and Yushi Bai and Zhongpei Qiao and Zihan Wang and Zikang Wang and Zilin Zhu and Ziqiang Liu and Zixuan Li and Bojie Wang and Bosi Wen and Can Huang and Changpeng Cai and Chao Yu and Chen Li and Chengwei Hu and Chenhui Zhang and Dan Zhang and Daoyan Lin and Dayong Yang and Di Wang and Ding Ai and Erle Zhu and Fangzhou Yi and Feiyu Chen and Guohong Wen and Hailong Sun and Haisha Zhao and Haiyi Hu and Hanchen Zhang and Hanrui Liu and Hanyu Zhang and Hao Peng and Hao Tai and Haobo Zhang and He Liu and Hongwei Wang and Hongxi Yan and Hongyu Ge and Huan Liu and Huanpeng Chu and Jia'ni Zhao and Jiachen Wang and Jiajing Zhao and Jiamin Ren and Jiapeng Wang and Jiaxin Zhang and Jiayi Gui and Jiayue Zhao and Jijie Li and Jing An and Jing Li and Jingwei Yuan and Jinhua Du and Jinxin Liu and Junkai Zhi and Junwen Duan and Kaiyue Zhou and Kangjian Wei and Ke Wang and Keyun Luo and Laiqiang Zhang and Leigang Sha and Liang Xu and Lindong Wu and Lintao Ding and Lu Chen and Minghao Li and Nianyi Lin and Pan Ta and Qiang Zou and Rongjun Song and Ruiqi Yang and Shangqing Tu and Shangtong Yang and Shaoxiang Wu and Shengyan Zhang and Shijie Li and Shuang Li and Shuyi Fan and Wei Qin and Wei Tian and Weining Zhang and Wenbo Yu and Wenjie Liang and Xiang Kuang and Xiangmeng Cheng and Xiangyang Li and Xiaoquan Yan and Xiaowei Hu and Xiaoying Ling and Xing Fan and Xingye Xia and Xinyuan Zhang and Xinze Zhang and Xirui Pan and Xu Zou and Xunkai Zhang and Yadi Liu and Yandong Wu and Yanfu Li and Yidong Wang and Yifan Zhu and Yijun Tan and Yilin Zhou and Yiming Pan and Ying Zhang and Yinpei Su and Yipeng Geng and Yong Yan and Yonglin Tan and Yuean Bi and Yuhan Shen and Yuhao Yang and Yujiang Li and Yunan Liu and Yunqing Wang and Yuntao Li and Yurong Wu and Yutao Zhang and Yuxi Duan and Yuxuan Zhang and Zezhen Liu and Zhengtao Jiang and Zhenhe Yan and Zheyu Zhang and Zhixiang Wei and Zhuo Chen and Zhuoer Feng and Zijun Yao and Ziwei Chai and Ziyuan Wang and Zuzhou Zhang and Bin Xu and Minlie Huang and Hongning Wang and Juanzi Li and Yuxiao Dong and Jie Tang},
      year={2026},
      eprint={2602.15763},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.15763},
}
```

## License

Licensed under the [MIT License](LICENSE), like the original model. Copyright (c) 2026 Z.AI Co., Ltd. The license text and copyright notice are included in this repository as required. The weights are provided "as is", without warranty of any kind.

Original model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). Conversion: numallama.
