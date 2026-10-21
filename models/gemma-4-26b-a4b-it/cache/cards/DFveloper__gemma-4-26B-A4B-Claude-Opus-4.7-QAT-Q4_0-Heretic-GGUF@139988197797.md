---
license: apache-2.0
datasets:
- DFveloper/claude-opus-4.6-4.7-korean-8.7k
language:
- ko
- en
tags:
- fine-tuned
- gemma-4
- gguf
- llama.cpp
- korean-reinforced
- abliterated
- MoE
- Claude
- reasoning
- chain-of-thought
- Distill
base_model:
- google/gemma-4-26B-A4B-it
model-index:
- name: Gemma-4-Opus
  results:
  - task:
      type: text-generation
    dataset:
      type: HLE
      name: HLE(Ko)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.067
      verified: false
  - task:
      type: text-generation
    dataset:
      type: MMLU-Pro
      name: KMMLU-Pro
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.606
      verified: false
  - task:
      type: text-generation
    dataset:
      type: CLIcK
      name: CLIcK
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.709
      verified: false
  - task:
      type: text-generation
    dataset:
      type: MuSR
      name: MuSR(Ko)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.638
      verified: false
---
# gemma-4-26B-A4B-Opus

`gemma-4-26B-A4B-Opus-4.7-QAT-Q4_0-Heretic` is a high-performance conversational model from the Gemma-4-Opus family. It has been fine-tuned to provide concise, useful, and accurate information across a wide range of topics.

## Model Card

### Model details

- **Model Name:** gemma-4-26B-A4B-Opus-4.7
- **Architecture:** Fine-tuned Gemma 4 26B A4B.
- **Training Data:** The model was fine-tuned using the `DFveloper/claude-opus-4.6-4.7-korean-8.7k`, which includes instructions, reasoning, and high-quality conversational examples.
- **License:** Apache-2.0.

### Usage

```bash
llama-server -m Gemma-4-26B-A4B-Opus-4.8-Q4_0.gguf --host 0.0.0.0 --port 11435 -c 262114 -np 1 -fa on --jinja --cache-ram 256 --cache-reuse 256 --cache-prompt --temp 1.3 --mmproj Gemma-4-26B-A4B-Opus-4.8-Q4_0-mmproj.gguf
```

<details><summary>Korean Example</summary>

> LLM의 원리를 전문지식없는 중2가 알아듣게 설명해봐.

쉽게 말해서, LLM(거대언어모델)은 **'엄청나게 똑똑한 자동 완성 기능'**이라고 생각하면 돼.

스마트폰으로 카톡할 때, "오늘"이라고 치면 "오늘 뭐해?"처럼 다음에 올 단어를 추천해주잖아? 그 원리랑 똑같은 건데, LLM은 그 스케일을 엄청 키운 거야.

---

### 1. 어떻게 공부했냐면...

이 친구는 세상에 있는 거의 모든 글(인터넷 게시글, 책, 뉴스, 논문, 위키피디아 등)을 다 읽었어. 그냥 읽은 게 아니라, 아주 꼼꼼하게 '어떤 단어가 어떤 순서로 나오는지' 그 **통계적인 패턴**을 학습한 거야.

예를 들어, **"철수는 배가..."**라는 문장이 있으면,
* 다음에 **"고프다"**가 올 확률이 80%
* **"아프다"**가 올 확률이 15%
* **"사과를 먹는다"**가 올 확률이 5%

이런 식으로 수많은 데이터로 확률을 계산하는 훈련을 수없이 반복한 거지.

### 2. 왜 똑똑하게 느껴지는 걸까? (핵심 비밀)

"어? 그건 단순한 확률 계산인데, 어떻게 질문에 답변을 하고 코딩까지 해?"라고 의문이 들 수도 있어.

그 이유는 **문맥(Context)**을 읽는 능력이 엄청나기 때문이야. LLM은 문장에서 모든 단어가 서로 어떻게 연관되어 있는지 계산해. 이걸 '어텐션(Attention)'이라는 기술이라고 불러.

* 문장에 "사과"라는 단어가 나왔을 때, 그 앞에 "맛있는 과일"이 있으면 '과일'에 집중하고, 그 앞에 "스티브 잡스는"이 있으면 '컴퓨터 회사'에 집중해.
* 이렇게 주변 단어들 사이의 관계를 계산하기 때문에, 마치 사람이 진짜로 이해하는 것처럼 자연스러운 답변을 내놓을 수 있는 거야.

### 3. AI는 진짜로 '생각'을 할까?

**엄밀히 말하면 "아니야"**라고 답해야 해.

AI는 "사과는 달콤하고 빨갛다"라는 문장을 내놓을 때, 사과를 직접 먹어본 느낌을 갖는 게 아니야. 단지 세상에 있는 수십억 개의 문장을 학습한 결과, "사과"라는 단어 다음에 "달콤하다", "빨갛다"라는 단어가 나타날 확률이 매우 높다는 걸 수학적으로 계산한 거야.

즉, AI는 **세상에서 가장 거대한 '수학적 패턴 매칭 머신'**이라고 할 수 있어.

### 요약

1.  **학습:** 세상의 거의 모든 텍스트를 읽는다.
2.  **원리:** 다음에 어떤 단어가 오는 게 가장 자연스러울지 확률을 계산한다.
3.  **왜 똑똑한지:** 문맥의 관계를 계산하는 능력이 상상을 초월할 정도로 뛰어나기 때문이다.
4.  **주의점:** 실제로 무언가를 이해하고 생각하는 '지능'이 아니라, 매우 정교한 '통계적 추론'에 가깝다.
</details>
<details><summary>English Example</summary>

> Explain the principles of LLM in a way that a 8th grader without specialized knowledge can understand.

Imagine you’re texting a friend and your phone suggests the next word as you’re typing. You’ve probably seen that before. An LLM (Large Language Model) is basically a much bigger, much smarter version of that autocomplete.

Here’s how it works in three simple steps:

**1. Reading almost everything ever written**
Imagine a robot that reads every book, every Wikipedia article, and every blog post ever written. While it reads, it’s not really "learning" things like a human does. Instead, it's looking for patterns. It sees that whenever the word "the" appears, the next word is often an "a," "an," or "the" again—never a comma. It notices that when someone talks about "Paris," words like "France," "Eiffel Tower," and "city" show up frequently. It builds a massive statistical map of how all human language fits together.

**2. The Great Guessing Game**
When you ask an LLM a question, it doesn't "know" facts in the way you do. Instead, it calculates the probability of the next word. If you ask, "What color is the sky?", it doesn't search a database for a fact; it calculates that the word "blue" is the most likely word to follow after that question, based on everything it ever read. It does this word-by-word, over and over, until it builds a full sentence.

**3. Paying "Attention" to what matters**
This is the secret sauce. In the past, computers had trouble with words that have multiple meanings. For example, in the sentence "The bank of the river," the word "bank" means something different than in "I put my money in the bank." An LLM uses something called "attention"—it looks at all the other words in the sentence simultaneously to figure out which words give context to the others. This helps it understand which meaning of "bank" is correct.

**The bottom line**
An LLM isn't "thinking"—it's just an incredibly advanced pattern-matching engine. It's like having a student who has memorized every sentence ever written and can instantly guess what the next sentence should be, based on how people usually write.
</details>

### Prompting Guidelines

`gemma-4-26B-A4B-Opus-4.7` performs optimally with the following strategies:
- **Be explicit and detailed:** The more specific instructions you give, the better the model performs.
- **Use examples:** Providing both positive and negative examples (few-shot prompting) is highly effective.
- **Chain of Thought:** Explicitly request step-by-step reasoning.
- **Specifying format:** If you need a specific length or format (like JSON or Markdown), clearly state this.

### Licensing and Deployment

This model is released under the Apache-2.0 license.

\+ Since I'm an high school student, I'll might be very busy some day. before the day comes, I'll do my best in this present.

---
Finetuned by DFveloper. Released under Apache-2.0.

This model is a derivative of Gemma-4 and is distributed under the
[**Gemma Terms of Use**](https://ai.google.dev/gemma/terms). By using this model
you agree to those terms and Google's Prohibited Use Policy.