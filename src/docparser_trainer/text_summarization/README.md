# 文本摘要

- Sequence to Sequence

## 评价指标

Rouge（Recall-Oriented Understudy for Gisting Evaluation）是一套用于评估摘要质量的指标，通过将生成的摘要与参考摘要进行比较（主要关注召回率）的方式来评估。

- ROUGE
  - ROUGE-1, ROUGE-2, ROUGE-L
  - 分别基于1-gram, 2-gram, LCS
  - 示例：

    ```txt
            预测           标签
    原始文本：今天不错       今天太阳不错
    1-gram：今 天 不 错     今 天 太 阳 不 错 
    2-gram：今天 天不 不错   今天 天太 太阳 阳不 不错

    Rouge-1: Precision=4/4, Recall=4/6, F1=2*1*0.666/(1+0.666)=0.8 衡量生成摘要和参考摘要之间的单字词重叠。
    Rouge-2: Precision=4/6, Recall=2/5, F1=2*0.666*0.4/(0.666+0.4)=0.5 衡量生成摘要和参考摘要之间的双字词重叠。
    Rouge-L: Precision=4/4, Recall=4/6, F1=0.8 衡量生成摘要和参考摘要之间的最长公共子序列。
    ```

## 基于 Transformer 的解决方案

### 方案一：编码器 + 解码器 (T5)

- 数据处理
  - input 和 labels 分开处理, labels 的最后一定是 eos_token
  - labels 不仅是标签，还是解码器的输入

  ```txt
  1 2 3 4 5 6 7 eos | bos 10 11 12 13 14 eos
  编码器输入          \\\
                     | bos 10 11 12 13 14 eos
                        解码器输出

  计算 loss 时，只计算 decoder 的 loss
  ```

- 模型结构
  - XXForConditionalGeneration

### 方案二：只使用解码器 (GLM)

- 模型原理
  - 只使用编码器，借助注意力掩码实现编码器解码器的效果，只计算目标部分损失(掩码、自回归、自注意力)

  ```txt
  a. sample spans from the input sequence
          __    _____ 
    x1 x2 x3 x4 x5 x6
            
  b. divide the input into PartA/PartB
    PartA: x1 x2 [M] x4 [M]
    PartB:        x3     x5 x6

  c. generate the PartB spans autoregressively

                x1 x2 [M] x4 [M] | [S] x5 x6 | [S] x3
    Position 1: 1  2   3  4   5     5  5  5     3  3   (全局位置)
    Position 2: 0  0   0  0   0     1  2  3     1  2   (局部位置)
                                 ||
            GLM (Transformer w / masked self-attention)
                                 \/
                                    x5 x6 [E] | x3 [E]

  d. self-attention mask
        Key
        x1 x2 [M] x4 [M] [S] x5 x6 [S] x3
  Q x1  .  .   .  .   .   *  *  *   *  *
  e x2  .  .   .  .   .   *  *  *   *  *
  r [M] .  .   .  .   .   *  *  *   *  *
  y x4  .  .   .  .   .   *  *  *   *  *
    [M] .  .   .  .   .   *  *  *   *  *
    [S] .  .   .  .   .   .  *  *   *  *
    x5  .  .   .  .   .   .  .  *   *  *
    x6  .  .   .  .   .   .  .  .   *  *
    [S] .  .   .  .   .   .  .  .   .  *
    x3  .  .   .  .   .   .  .  .   .  .
  ```

- 数据处理
  - input 和 labels 合并在一起处理, labels 的最后一定是 eos_token

  ```txt
  1 2 3 4 5 6 7 eos | bos  10 11 12 13 14 eos
  input              \\\
  -100...            |-100 10 11 12 13 14 eos
  label
  ```
