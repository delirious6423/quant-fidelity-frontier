---
language:
- zh
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
tags:
- rewrite3
- novel-rewrite
- chinese
- web-novel
- jev
- mtp
- q4km
- gguf
- compressed-tensors
- text-generation
---

## 本README已大量使用rewrite3优化文案

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', 'Microsoft YaHei', sans-serif; border: 1px solid #cbd5e1; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.05); overflow: hidden; background: #ffffff; margin-bottom: 30px;">

  <div style="background: linear-gradient(135deg, #7c3aed 0%, #4c1d95 100%); padding: 24px; color: white;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
      <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: white; border: none;">🖊️ rewrite3-27B · Q4_K_M</h1>
      <span style="background: #10b981; color: white; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;">Rewrite · MTP Release</span>
    </div>
    <p style="margin: 8px 0 0 0; font-size: 14px; color: #ddd6fe; font-weight: 500;">面向中文网文的「JEV 打分驱动」改写模型：把 AIGC 劣化稿重写成通顺、有作者个人风格、不照抄的正文，并自带 MTP 投机解码</p>
  </div>

  <div style="display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 24px; background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
    <span style="background: #f3e8ff; color: #6b21a8; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #e9d5ff;">🖊️ 特化改写思维链提高泛化</span>
    <span style="background: #dbeafe; color: #1e40af; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #bfdbfe;">🎯 外置判别器提高改文效果</span>
    <span style="background: #d1fae5; color: #065f46; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #a7f3d0;">⚡ 微调了MTP提高TPS</span>
  </div>
</div>

## 量化档位的 JEV 表现

本仓库是 Q4_K_M GGUF 版本，权重 17.0 GB（15.8 GiB）。下面的表是同一批样本下各格式的横评。

「JEV 分」是句子层面的劣质概率，由 jevnovel3 0.8B 打分器给出，能分辨 AI 八股和人类笔触。这套打分器是在大量「AI 味重／人味足」的对照样本上训练出来的，越接近八股套话、注水凑字数，分数就越往上涨。

分数越低越好：0.08 以下算干净，0.08 到 0.12 是正常改写的区间，0.2 往上基本都是眼熟的 AI 味。

再看表格，表格里的每一格都是同一组六条提示跑出来的平均结果，同样的机器，KV cache 也都统一用 bf16，就是为了排除 KV 量化带来的干扰，速度里连思考过程和正文都算了进去。

| 格式 | JEV 均值 ↓ | 脱稿度 ↑ | 最长照抄（字）↓ | 照抄总量 ↓ | 正文长度 | 超长样本 | 平均耗时 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **FP8**（W8A8） | **0.0966** | **0.844** | 31.2 | **54.2** | 1036 | 3/6 | 71s |
| **IQ4_XS**（GGUF） | 0.0982 | 0.767 | **25.2** | 86.0 | 1007 | 0/6 | **41s** |
| **NVFP4**（W4A4） | 0.1016 | 0.804 | 25.5 | 70.2 | 944 | 0/6 | 43s |

表格里还有两个辅助指标：脱稿度是正文里 5 字窗口「原文没出现过」的比例，越高说明改写得越彻底，加料加得越多；最长照抄和照抄总量统计与原文逐字重合的字数，越少越好（本项目的硬性约束是禁止连续照抄超过 15 字）。

超长样本指的是改写后比原文更长的情况：偶尔会「加料换长度」，介意的话，在提示词里把长度要求写死即可。

**我们认为：FP8 > IQ4_XS > NVFP4。** FP8 在照抄与脱稿度上双双拉满，也是我们首选的主力权重；IQ4_XS 速度几乎快一倍、JEV 只差一点，适合显存紧张或本地部署；NVFP4 在 Blackwell 上的显存占用占优，但 JEV 稍高一些。

BF16 是质量基准，但没有单独跑这组 6 条评测；Q3_K_M、Q4_K_M、IQ3_XS 属于 3/4bit 档，暂时还没测试，后续会补。

## 快速开始

本仓库是 GGUF 格式，MTP 头已经内嵌（`blk.32.nextn.*`），用新版 llama.cpp（nightly b12374 及以上）加载即可：

```bash
llama-server -m rewrite3-27b-Q4_K_M.gguf \
  -c 16384 --n-gpu-layers 99 \
  --spec-type draft-mtp --spec-draft-n-max 3
```

想用 vLLM 的话，换对应的 BF16 / FP8 / NVFP4 仓库即可。

## 全部版本

| 格式 | 定位 | 权重 | Hugging Face | ModelScope |
|:--|:--|--:|:--|:--|
| **BF16** | 质量基准 | 54.7 GB | [rewrite3-27b-bf16](https://huggingface.co/aikexue170/rewrite3-27b-bf16) | [rewrite3-27b-bf16](https://modelscope.cn/models/alkaid55555/rewrite3-27b-bf16) |
| **FP8** W8A8 | 主力部署（48G 级） | 35.9 GB | [rewrite3-27b-fp8](https://huggingface.co/aikexue170/rewrite3-27b-fp8) | [rewrite3-27b-fp8](https://modelscope.cn/models/alkaid55555/rewrite3-27b-fp8) |
| **NVFP4** W4A4 | 这个量化炸了，用 IQ4_XS 吧 | 19.7 GB | [rewrite3-27b-nvfp4](https://huggingface.co/aikexue170/rewrite3-27b-nvfp4) | [rewrite3-27b-nvfp4](https://modelscope.cn/models/alkaid55555/rewrite3-27b-nvfp4) |
| **IQ4_XS** GGUF | 本地优先质量 | 15.5 GB | [rewrite3-27b-iq4xs](https://huggingface.co/aikexue170/rewrite3-27b-iq4xs) | [rewrite3-27b-iq4xs](https://modelscope.cn/models/alkaid55555/rewrite3-27b-iq4xs) |
| **Q4_K_M** GGUF（本仓库） | 16G 塞不下，建议 IQ4_XS | 17.0 GB | [rewrite3-27b-q4km](https://huggingface.co/aikexue170/rewrite3-27b-q4km) | [rewrite3-27b-q4km](https://modelscope.cn/models/alkaid55555/rewrite3-27b-q4km) |
| **Q3_K_M** GGUF | 16G 卡余量 | 13.7 GB | [rewrite3-27b-q3km](https://huggingface.co/aikexue170/rewrite3-27b-q3km) | [rewrite3-27b-q3km](https://modelscope.cn/models/alkaid55555/rewrite3-27b-q3km) |
| **IQ3_XS** GGUF | 极限小显存 | 12.3 GB | [rewrite3-27b-iq3xs](https://huggingface.co/aikexue170/rewrite3-27b-iq3xs) | [rewrite3-27b-iq3xs](https://modelscope.cn/models/alkaid55555/rewrite3-27b-iq3xs) |

## 训练与评估

我上一代的rewrite2数据集是AIGC劣化输入->人类网文原作输出，一共两万条。后来我们发现了诸多问题：1、由于劣化文普遍长于人类原作，而劣化到原文的映射几乎都是删改为主，不会补充内容，所以模型倾向于缩短字数，导致一些正常长度的文会被改的太短太碎。2、无cot导致模型注意力低下、泛化性差。改文容易出现逻辑错误，文风较差，对长度敏感等等诸多问题。3、模型并不知道哪些是真正的八股，只能凭借数据先验判断句子好坏，造成了识别上的巨大误差，误伤了极多好句子，而对八股杀不全。

针对这三个问题，我全面升级了数据集。首先我引入了我曾经的模型jevnovel3 0.8B作为八股识别器，在整句的末尾以【概率】的形式附加了jevnovel3对本句话“烂”的概率(取观感曲线的末尾作为整句的p_bad)，随后使用claude opus 5.5作为第一次扩写，对人类原作进行细节补充，集中加料描写部分，避开引入文段无法推断出来的信息。随后由于claude本身也有八股，所以我们使用jevnovel3对全文打分标记，然后使用spark x 2.6（讯飞星火家的模型，中文写作能力优秀）进行定点修改。再一次进行jevnovel3打分，最终使用kimi k3（强大的中文模型，同样写作优秀）进行最后一轮次修改。经过一轮扩写两轮修改，最终的加料文在jevnovel平均分上已经持平或优于人类原作。随后我们使用gemini 4 argon构建了思维链（因为文学能力极为出众且cot活人感强）。argon接收到人类总结的八股库（告知八股类型）、劣化文原文及其jevnovel分数、最终的加料文，将jevnovel分数、八股类型和改文思路、加料缘由一一推断出来，看答案写过程构建出完整的cot映射。这就是最新的数据集的构建方法。

## 使用案例

以下是 DeepSeek V4.1 flash 1017 无预设直出效果，及其 rewrite3 改写后效果。

<div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0 8px 0;">
  <div style="flex: 1 1 360px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; padding: 16px;">
    <p style="margin: 0 0 10px 0; font-size: 12px; font-weight: 700; color: #b91c1c; letter-spacing: 0.3px;">AIGC 原文（全文 1418 字）</p>
    <p style="margin: 0; font-size: 12.5px; color: #7f1d1d; line-height: 1.9;">我端着餐盘在食堂里扫视了一圈，最后在靠窗的位置找到了她。<br><br>校花林晚坐在那里，一个人。她面前放着一碗白粥，一只剥了一半的茶叶蛋，还有一小碟萝卜干。她剥蛋的动作很慢，指甲掐进蛋壳里，一片一片往下撕，像是在做什么精密实验。晨光从窗户斜进来，把她的侧脸照得有点透明，连耳朵尖上那层细细的绒毛都看得清。<br><br>我手里端着两个肉包、一碗豆浆，在“假装路过”和“找个不尴尬的位置坐下”之间反复横跳了大概十五秒。最后我选了离她三张桌子远的地方坐下——这个距离比较体面，不会显得刻意，又能让她余光扫到我的时候不至于完全没印象。<br><br>其实我连她认不认识我都不确定。我们同一个年级，她在三班，我在七班，隔着一整个走廊。上学期校园文化节的时候，我和她搭过同一部电梯。她当时抱着一摞书法社的展板，最上面那块歪了，我帮她扶了一下。她说了声“谢谢”，声音很轻，像往水里丢了一颗小石子。<br><br>就这三个字，我记到现在。<br><br>我咬了一口包子，烫得直吸气。豆沙馅流出来沾在手指上，我赶紧拿纸巾擦，擦到一半下意识抬头——她正好看过来。<br><br>四目相对，零点几秒。<br><br>我瞬间低头，假装在专心致志地研究手里的纸巾。心跳得跟跑完八百米似的，耳朵根子一阵发烫。我在心里骂自己：你躲什么啊？你又不是做了什么亏心事！<br><br>但我就是不敢。她太……亮了。那种亮不是刺眼，是像冬天教室里突然开了暖光灯，让人想凑近，又怕手伸过去显得冒犯。据说隔壁班有个男生连着给她带了一个月的早饭，每天一杯热美式一个三明治，放在她桌洞里，她一口都没动过，全分给同桌了。这事传开之后，那个男生被笑了整整一个学期。<br><br>我可不想成为下一个笑话。<br><br>我闷头喝豆浆，喝得太急，嘴角沾了一圈白沫。我正拿纸巾擦，余光里她站起来了——端着她的粥碗和半个茶叶蛋，朝我这个方向走过来。<br><br>我整个人僵住了。手还捏着纸巾按在嘴角，动作定得像被人按了暂停键。<br><br>她走到我旁边那张桌子，坐下。原来她是要去对面拿醋瓶子，顺路换个位置。她把碗放下的时候冲我笑了一下，说：“你那个豆沙包，是二楼新出的吗？我昨天想买没抢到。”<br><br>我大脑空白了大概两秒。然后我听见自己说：“啊……对，二楼，挺好吃的，就是有点甜。”<br><br>说完我就想抽自己。什么叫有点甜？人家就随口一问，你搁这做美食测评呢？<br><br>但她没走，还歪了歪头说：“那你觉得红糖馒头跟豆沙包哪个甜？”<br><br>我认真想了想，说：“红糖馒头。豆沙包是那种……温柔的甜。”<br><br>她愣了一下，然后笑了。眼睛弯起来的时候，有点像食堂门口那只总晒太阳的橘猫。<br><br>“温柔的甜，”她重复了一遍，“行，那我下次试试。”<br><br>然后她就端着碗回自己位置了。我坐在那儿，豆浆凉了，包子还剩一个，但我突然觉得食堂的广播音乐都变得顺耳了。<br><br>我掏出手机给同桌发了条消息：“她刚才跟我说话了。”<br><br>三秒后回：“谁？”<br><br>“校花。”<br><br>“？？？说什么了？”<br><br>我想了想，打字：“讨论了一下红糖馒头和豆沙包的甜度分级。”<br><br>对面沉默了十秒，然后回：“你赢了。全校没人用这种方式跟校花搭讪的。”<br><br>我把手机揣回兜里，把最后那个包子吃完。她还在旁边那张桌子低头喝粥，耳边的碎发垂下来。我没再过去说话，但走的时候故意绕了半圈，从她斜后方经过，步子放得很稳。<br><br>她好像没抬头。但我走到食堂门口的时候，鬼使神差回头看了一眼——她正举起那个剥了一半的茶叶蛋，朝我轻轻晃了晃，像在说“下次见”。<br><br>我差点被食堂门槛绊一跤。<br><br>但那天的早饭，是我上高中以来吃得最香的一顿。</p>
  </div>
  <div style="flex: 1 1 360px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 16px;">
    <p style="margin: 0 0 10px 0; font-size: 12px; font-weight: 700; color: #15803d; letter-spacing: 0.3px;">rewrite3-27B 改写（全文 1551 字）</p>
    <p style="margin: 0; font-size: 12.5px; color: #14532d; line-height: 1.9;">我端着早餐在食堂里转悠，早上人多，打饭窗口前排着一溜学生，嘈杂声嗡嗡地响，最后才在靠窗的位置看到了她。靠窗的位置上，坐着校花林晚。面前是一碗冒热气的白粥，一只剥了一半的茶叶蛋，还有一碟萝卜干。她剥蛋剥得不紧不慢，剥下来的蛋壳一片一片码在碟子边上。我端着自己的早餐，两个肉包，一碗豆浆，犹豫了一会儿，挑了个离她三张桌子远的位子坐下。这个距离不远也不近，合适得很。我放下托盘，偷偷瞄了一眼，她正低着头，好像在跟那个茶叶蛋较劲。<br><br>其实我俩也算半个熟人。同一年的，不过她在三班，我在七班，中间隔了一个走廊，平时见面基本全靠缘分。再说我们学校那么大，偏偏校园文化节那天，我跟她搭了同一部电梯。她抱着一摞书法社的展板，最上面一块歪了，我伸手帮她扶了一下。她说了声谢谢。<br><br>我咬了一口包子，烫嘴，豆沙馅顺着指缝流出来，我慌忙抽了张纸巾去擦，抬头一看，正好对上她的目光。<br><br>她朝我这边看。<br><br>我赶紧低下头假装若无其事，心一下子跳到了嗓子眼，把纸巾揉成一团攥在手心里。我心想，你躲什么啊？又不是做了什么亏心事！但我就是不敢跟她打招呼。她太耀眼了。听说隔壁班有个男生，连着给她带了一个月的早餐，每天一杯热美式配一个三明治，放在她桌洞里，她一口没动，全都转手给了同桌。这件事后来传开了，那个男生被嘲笑了整整一个学期。这种事最好别往自己身上引。我可不想成为下一个笑话。我低头喝着豆浆，喝得腮帮子鼓鼓的，生怕被人看出我心不在焉。正拿纸巾擦嘴角，余光里却看见她起身了。她端着碗，手里还捏着半个茶叶蛋，正朝我这边走过来。<br><br>我整个人僵住了，连呼吸都不敢大声，脑子里甚至已经开始盘算待会儿该怎么解释，为什么我会在她面前犯这种低级错误。林晚走到我旁边的桌子坐下。原来她是去对面拿醋瓶子，拿完就顺势在我旁边坐下了。路过我桌边时，裙摆带起一阵风。<br><br>她把碗往桌上一放，冲我笑了一下：“你那豆沙包是二楼新出的吗？我昨天去买没买到。”<br><br>我一时没反应过来，愣了好一会儿，脑子空空的，等回过神，嘴巴已经先一步把话说了出来：“啊……是二楼，好吃，就是有点甜了。”<br><br>说完我就想抽自己。什么叫有点甜了？人家随口一问，你搁这儿做美食测评呢？<br><br>她却像是真感兴趣，还追着问了一句：“那你说红糖馒头和豆沙包，哪个甜？”<br><br>我张了张嘴，只觉得手心全是汗，脑子里那点准备好的搭讪词全飞了，只好老老实实思考这个问题。“红糖馒头。豆沙包是那种……温和的甜。”<br><br>林晚愣了愣，随即笑出声来，眼睛弯成了两道月牙，看着我，好像第一次知道还有人在乎馒头和包子的区别。<br><br>“温和的甜，”她点点头，“诶，好，那我下次买来尝尝。”<br><br>她拿着醋瓶回到自己的桌子旁，往白粥里滴了两下，继续低头喝粥。我坐在那里，豆浆已经凉了，包子还剩一个没动，可奇怪的是，我突然觉得食堂背景音里放的流行歌，都变得好听起来了。我掏出手机，给同桌发了条消息：“她刚才跟我说话了。”<br><br>过了没一会儿，屏幕上跳出三个字：“谁？”<br><br>“校花。”我回复得很快，一字一顿地打了过去。<br><br>“？？？说什么了？”<br><br>我想了想，觉得自己说得还挺专业，又回复：“讨论了一下红糖馒头和豆沙包的甜度分类。”<br><br>对面沉默片刻，回了句：“你赢了。全校应该没人用这种方式跟校花搭讪。”<br><br>我把手机揣回兜里，把最后一个包子吃了，又喝了一口凉掉的豆浆。她还坐在旁边那张桌子低头喝粥，一缕头发垂在耳边，被她随手拨到耳后。我偷偷看了两眼，便起身准备去上早读。走出食堂时，我故意绕了个圈子，从她斜后方的位置经过，没敢停步，也刻意放轻了脚步。<br><br>她好像没注意到，也没回头。但走出门口时我鬼使神差地往回看了一眼，她正站在窗边，举着手里的半个茶叶蛋冲我挥了挥，像是在打招呼。我几乎被食堂的门槛绊了一跤。<br><br>但那天的早餐，是我入校以来吃得最香的一次。</p>
  </div>
</div>

<p style="font-size: 12px; color: #64748b; margin: 0 0 8px 0;">输入按生产流程处理：jevnovel3 0.8B 逐句打分 → 拼上 <code>INSTRUCTION.txt</code> → 送进 rewrite3-27B。示例里省掉了【分数】标记，方便阅读。</p>

## 使用建议与限制

不要拿它去改写其他文体，比如广告语、公文、法律文书，它只认小说。

虽然训练语料的平均长度在一千五百字上下，但有了 CoT，泛化能力明显变强，实测从八百字到三千字都能稳着出活。超出这个范围，我没有系统地测过，就像没走过的夜路，不好打包票，欢迎大家自己来试试。

它只负责改写中文小说，改别的也行，不过毕竟OOD了出bug也是很正常的。不过我实测下来泛化性还真挺强的，即使喂进来文书它cot也大概率惊讶一下吐槽几句，然后老老实实的改，改完后效果竟然意外挺好。jevnovel3 分数只是重点提示，不是硬性限制，有打分器的时候，会盯着高分句重点杀；没有打分器的情况下也能直接用纯文本喂，【】填NA就行，就是效果会差一些，可能漏杀八股。

另外，jevnovel3 0.8B 是另一个独立模型，不包含在这个仓库里，需要的话请去我jevnovel相关的仓库获取，欢迎取用。低比特格式，尤其是 3bit GGUF，可能会放大复读、提前收尾、直接照抄原文的毛病，建议你在正式使用前，先拿自己的文本抽几条样本测一测。

它继承了基座模型的一些知识限制，同时也会继承输入文本中的某些倾向性，请勿用它产生任何违法内容，也不要把它的输出当成通用的内容审查工具。本次发布的权重采用 Apache-2.0 协议，具体许可条件以 Qwen 官方规定为准。

## 文件清单

本仓库（Q4_K_M GGUF）只有一个权重文件 `rewrite3-27b-Q4_K_M.gguf`（MTP 已内嵌），外加 `INSTRUCTION.txt` 和 `SHA256SUMS`。用新版 llama.cpp（nightly b12374 及以上）加载即可；加载后建议先跑一条短改写，确认显存与实际输出，再接入生产。
