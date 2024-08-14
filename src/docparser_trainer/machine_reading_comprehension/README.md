# MRC 任务介绍

## 常见类型

- 完形填空
- 选择题
- 片段抽取
- 自由生成

## 片段抽取

### 数据集格式

```json
{
    "id": "0",
    "context": "...",
    "question": "...",
    "answer": {
        "answer_start": [11, 22],
        "text": ["...", "..."]
    }
}
```

### 评估指标

- 精准匹配度（Exact Match）
- 模糊匹配度（F1）

```txt
label: 北京
pred: 北京天安门

EM = 0
F1 = (2 * 2/2 * 2/5) / (2/2 + 2/5) = 0.8 / 1.4 = 0.57 
```

### 基于 Transformer 的解决方案

- 数据预处理
  - 数据预处理格式：```[CLS] Question [SEP] Context [SEP]```
  - 定位答案位置：start_position, end_position (token's)
  - offset_mapping (类似 token classification 里的 word_ids)

- Context 过长如何解决：
  - 截断(信息会丢失)
  - 滑动窗口(实现复杂，但是损失较小)

- Model Structure: ModelForQuestionAnswering
- Evaluation:
