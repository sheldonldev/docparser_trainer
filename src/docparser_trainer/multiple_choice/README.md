# MRC 任务介绍

## MRC 任务2: 片段抽取

- 本质上是分类任务

### 数据集格式

```json
{
    "id": "0",
    "context": "...",
    "question": "...",
    "candidates/choices": ["...", "..."],
    "answer": "...",
}
```

### 基于 Transformer 的解决方案

- 数据预处理

```txt
[CLS]1 context [SEP] question choice1 [SEP]
[CLS]2 context [SEP] question choice2 [SEP]
[CLS]3 context [SEP] question choice3 [SEP]
[CLS]4 context [SEP] question choice4 [SEP]
||
\/
[CLS]1 [CLS]2 [CLS]3 [CLS]4 => softmax
```

- Model Structure: BertForMultipleChoice
- Evaluation:
