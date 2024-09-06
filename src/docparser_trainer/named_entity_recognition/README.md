# NER 任务介绍

- 本质上是多分类问题（这个 token 属于什么实体）

## 标注体系

- IOB1
- IOB2
  - I: 实体内部
  - O: 实体外部
  - B: 实体开始
  - B/I-XXX 的 XXX: 表示实体类别
- IOE1
- IOE2
- IOBES
  - I: 实体内部（有时候也会用M）
  - O: 实体外部
  - B: 实体开始
  - E: 实体结束
  - S: 表示一个词单独形成一个命名实体
- BILOU

## 评估指标

- F1
- Precision
- Recall

## 示例

```txt
# IOBES
Sentence: The     Hospital said  it would probably know  by Tuesday whether  its      patients had      Congo    Fever    . 
Label:    b-AGENT e-AGENT  o     o  b-DSE m-DSE    e-DSE o  o       b-TARGET m-TARGET m-TARGET m-TARGET m-TARGET e-TARGET o
Pred:     o       e-AGENT  b-DSE o  b-DSE m-DSE    e-DSE o  o       b-AGENT  b-DSE    b-TARGET m-TARGET m-TARGET e-TARGET o

label_num = 3 (The Hospital, would probably know, whether its patients had Congo Fever)
pred_num = 2 (would probably know, patients had Congo Fever)
correct_num = 1 (would probably know)

Precision = correct_num / pred_num = 1 / 2 = 0.5
Recall = correct_num / label_num = 1 / 3 = 0.333
F1 = 2 * Precision * Recall / (Precision + Recall) = 2 * 0.5 * 0.333 / (0.5 + 0.333) = 0.4
```

## 基于 Transformer 的解决方案

- Model Structure: BertForTokenClassification
- Evaluation: seqeval

## Transformers 自动调参

- 安装一个自动调参的后台 optuna
- TrainingArgs model_init 传参
- 用 trainer.hyperparameters_search() 训练
