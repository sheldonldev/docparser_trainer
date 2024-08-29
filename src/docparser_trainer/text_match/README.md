# 文本匹配与文本相似度任务介绍

- 只要涉及到两段文本之间的关系，都可以视作文本匹配任务。
  - 相似度计算
  - 问答匹配
  - 对话匹配
  - 文本推理
  - 片段推理和多项选择本质上也是文本匹配
- 本质上是分类任务

## 文本相似度

### 示例

```txt
Sentence A: 明天多少度
Sentence B: 明天气温多少
Label: 1

Sentence A: 可怕的事情终于发生了
Sentence B: 你到底想说什么
Label: 0
```

### 基于 Transformer 的解决方案

#### 方案一：交互匹配策略/单塔

- 数据预处理

  ```txt
  [CLS]1 sentence A [SEP] sentence B [SEP]
  [CLS]2 sentence A [SEP] sentence C [SEP]
  ...

  Sentence A 和 Sentence B 同时过一个模型 (单塔)
  ```

- 模型训练

  ```txt
  # 交互策略
  [CLS]1 -> Similarity(0/1) -> score
  [CLS]2 -> Similarity(0/1) -> score
  ...
    ||
    \/ 
    argmax

  # 全量数据进行推理，数据量大时效率很低
  ```

- 交互策略，输入句子对，对是否相似进行学习
- Model Structure: sentence A + sentence B -> Bert -> FFNN -> similarity

#### 方案二：基于向量匹配的解决方案（

```txt
候选文本 ->          -> 候选向量集合 
            向量模型                -> 向量匹配 -> 匹配结果
用户问题 ->          -> 问题向量
```

- 分别对句子进行编码，目标是让两个相似的句子相似度分数尽可能接近 1

- 数据预处理

  ```txt
  # >>> 一个batch
  [CLS]1 sentence A [SEP]
  [CLS]2 sentence B [SEP]

  [CLS]1 sentence A [SEP]
  [CLS]2 sentence B [SEP]

  [CLS]1 sentence A [SEP]
  [CLS]2 sentence B [SEP]

  [CLS]1 sentence A [SEP]
  [CLS]2 sentence B [SEP]

  ...
  # <<<
  Sentence A 和 Sentence B 分别过一个模型 (双塔)

  数据维度：[batch_size, 2, sequence_length]
  ```

- 模型结构：自定义 CosineEmbeddingLoss

  ```txt
  Sentence A -> Bert -> sentence A Embedding
  [batch_size, sequence_length] -> [batch_size, hidden_dim]
                                                            -> CosineEmbeddingLoss
  Sentence B -> Bert -> sentence B Embedding
  [batch_size, sequence_length] -> [batch_size, hidden_dim]
  ```

  - CosineEmbeddingLoss 是一个特殊的 loss，其计算公式如下：
    - `loss = 1 - cos(x1, x2) if y=1`
    - `loss = max(0, cos(x1, x2) - margin) if y=-1` (margin, 模糊边界的阈值， 默认为 0)

#### 更高效的工具

- sentence-transformers <https://www.sbert.net>
- text2vec <https://github.com/shibing624/text2vec>
- uniem <https://github.com/wangyuxinwhy/uniem>
