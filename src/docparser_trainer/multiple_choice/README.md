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
