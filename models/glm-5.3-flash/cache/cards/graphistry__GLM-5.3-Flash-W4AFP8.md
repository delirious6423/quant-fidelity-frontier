---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- glm
- glm5_next
- w4afp8
- quantized
- moe
pipeline_tag: text-generation
---


# GLM-5.3-Flash-W4AFP8

W4AFP8 quantization of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash):
routed MoE expert weights in INT4 (group-128, symmetric, AWQ-calibrated) with static
per-tensor FP8 activation scales; attention, dense MLPs and shared experts retain the
source block-FP8 weights. Served by SGLang's native CUTLASS W4A8 grouped GEMM on Hopper
(SM90).

**177.4 GB**, down from 328.3 GB (FP8 source) and 642.6 GB (BF16 master).


## Requirements

SGLang built from source. The `glm5_next` architecture landed after v0.5.19, and W4AFP8
checkpoints with a skip-list need a small loader fix that is still in review
([sgl-project/sglang#39250](https://github.com/sgl-project/sglang/pull/39250)).

Until it merges, install the branch `main` plus that fix:

```bash
git clone -b fix-w4afp8-ignored-layers https://github.com/tanmoyio/sglang
cd sglang && pip install -e "python[all]"
```

Already on `main` or using `lmsysorg/sglang:dev`:

```bash
git remote add tanmoyio https://github.com/tanmoyio/sglang
git fetch tanmoyio fix-w4afp8-ignored-layers
git ch
```

## Serving
Two flags are load-bearing. `--disable-shared-experts-fusion` is **required** -- without it
the shared-expert weights are fused into a layout the W4A8 kernel cannot consume. And
`--kv-cache-dtype` must be left unset.

### Without EAGLE

```bash
python -m sglang.launch_server \
  --model-path graphistry/GLM-5.3-Flash-W4AFP8 \
  --quantization w4afp8 \
  --disable-shared-experts-fusion \
  --tp-size 2 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --context-length 1048576 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --enable-cache-report
```

### With EAGLE
```bash
python -m sglang.launch_server \
  --model-path graphistry/GLM-5.3-Flash-W4AFP8 \
  --quantization w4afp8 \
  --disable-shared-experts-fusion \
  --tp-size 2 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --context-length 1048576 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --enable-cache-report \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

Quantization is auto-detected from `config.json`; `--quantization w4afp8` is optional.

The flag set above was verified together on 2xH200: server loads, generation is clean, both parsers work.


### Measured throughput

2xH200, TP=2, short prompts, 256-token generations.
| concurrency | no EAGLE | EAGLE | notes |
|---|---|---|---|
| 1 | 105.5 tok/s | 152.3 tok/s | single-stream, median of 3 |
| 4 | 73.6 /req | 99.5 /req | |
| 16 | 68.6 /req | 68.6 /req | compute-bound, speculation no longer helps |


## Evaluation

| Benchmark | **GLM-5.3-Flash w4afp8** | GLM-5.3-Flash unquantized |
|---|---|---|
| GPQA-Diamond | **90.9** | 91.2 |
| Graphistry Internal CyBT59 on Louie | **59.3%** | **61.01** |

More eval results will be published soon. 



## Citation

If you find GLM-5.3-Flash useful in your research, please cite our technical report:

```
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