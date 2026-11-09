---
license: apache-2.0
pipeline_tag: text-generation
language:
- ja
- en
base_model:
- Qwen/Qwen3.8-27B
tags:
- Qwen3.8
- Qwen3.8 27B
---

![qwen3.8-13.5b](https://cdn-uploads.huggingface.co/production/uploads/639e974f7145123e0d5b98ea/-kKJSN4vTsC_3WysG3kxh.png)

# Qwen3.8 13.5B

このモデルは[Qwen3.8 27b](https://huggingface.co/Qwen/Qwen3.8-27B)の半分のサイズの13.5Bを作ったら面白そうじゃね？という実験で生まれたモデルです。**実用性はありません。**

Qwen3.8 27bの枝刈りを行い、64層ある層を32層へ半分に削りました。何パターンか削り方を試しperplexityを測って良かったものを採用。<br>
embeddingは触らなかったので、**正確なパラメータ数は13.5Bではなく14.7B**です。

枝刈り直後の出力は完全に崩壊していましたが、Qwen3.8 27B, Qwen3.8 Flash Nextを教師として学習させることで大幅に回復(とはいえ日本語と英語が喋れるだけで入力に対しての出力の内容にズレがある)<br>
正直に使い物になるようなモデルではないが、さらに大規模な追加学習ができる場合可能性がありそうだと思ったので公開します。**計算リソースが有り余っている方はぜひこのモデルを最強にするチャレンジをしてみてください。**


## English

This model was born from a silly question: "wouldn't it be fun to build a 13.5B model at half the size of Qwen3.8 27b?" It is not meant to be practical.

I pruned Qwen3.8 27b, halving its 64 layers down to 32. I tried a few pruning patterns, measured perplexity, and adopted the one that scored best.<br>
Embedding was left untouched, so the exact parameter count is 14.7B, not 13.5B.

Output right after pruning was completely broken. Training with Qwen3.8 27B and Qwen3.8 Flash Next as teachers recovered it substantially — though it only speaks Japanese and English, and its outputs tend to drift from the inputs.<br>
Honestly, it is not a model you can rely on yet. But I thought it could have potential with a much larger run of additional training, so I'm publishing it. If you have spare compute, you are welcome to try the challenge of making this model the strongest one.


## 謝辞

この学習のために計算リソースを貸してくれた[ちゃれんげる
](https://x.com/challengervrllm)に感謝します。彼がいなければ今頃私はcolab破産していたでしょう。

## license

- Apache-2.0

## その他

問い合わせ等あれば[くすりちゃん](https://x.com/kusurichan_)に連絡ください。