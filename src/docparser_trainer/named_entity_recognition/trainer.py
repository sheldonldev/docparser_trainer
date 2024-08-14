from pathlib import Path

import evaluate  # type: ignore
import numpy as np
from docparser_models._model_interface.model_manager import (
    ensure_local_model,
    load_local_model,
    load_tokenizer,
)
from transformers import (  # type: ignore
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from util_common.decorator import proxy

from docparser_trainer._cfg import MODEL_ROOT, PROJECT_ROOT, setup_env
from docparser_trainer._interface.get_datasets import get_datasets

setup_env()
CHECKPOINTS_DIR = PROJECT_ROOT.joinpath("checkpoints/named_entity_recognition")


@proxy(http_proxy='127.0.0.1:17890', https_proxy='127.0.0.1:17890')
def get_evaluator():
    seqeval = evaluate.load("seqeval")
    return seqeval


def get_model(ckpt_dir: Path | None = None):
    if ckpt_dir is None:
        model_dir = ensure_local_model(
            model_id,
            model_cls=AutoModelForTokenClassification,
            local_directory=MODEL_ROOT.joinpath(f'{model_id}-mrc-fragment-extraction'),
            num_lables=len(label_list),
        )
    else:
        model_dir = ckpt_dir
    model = load_local_model(
        model_dir,
        model_cls=AutoModelForTokenClassification,
        num_lables=len(label_list),
    )
    return model


def preprocess_datasets(tokenizer, ner_datasets):
    def process_token(examples):
        tokenized = tokenizer(
            examples["tokens"],
            max_length=128,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,  # datasets 的数据集已经分词过了
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)  # 在交叉熵损失函数里，默认 -100 表示忽略计算
                else:
                    label_ids.append(label[word_id])
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    tokenized_datasets = ner_datasets.map(
        process_token,
        batched=True,
    )
    return tokenized_datasets


def train(ckpt_dir=None):

    seqeval = get_evaluator()

    def eval_metric(pred):
        predictions, labels = pred.predictions, pred.label_ids
        predictions = np.argmax(predictions, axis=-1)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        result = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
            mode="strict",
            scheme="IOB2",
        )
        return {
            "f1": result["overall_f1"],  # type: ignore
        }

    tokenized_datasets = preprocess_datasets(tokenizer, ner_datasets)

    model = get_model(ckpt_dir)

    args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        logging_steps=50,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=eval_metric,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )
    trainer.train()

    trainer.evaluate(eval_dataset=tokenized_datasets["test"])


def infer(ckpt_dir=None):
    from collections import defaultdict

    from transformers import pipeline

    model = get_model(ckpt_dir)
    model.config.id2label = {i: label for i, label in enumerate(label_list)}  # type: ignore

    ner_pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
    )
    input = "小明在北京上班"
    res = ner_pipe(input, aggregation_strategy="simple")

    ner_res = defaultdict(list)
    for r in res:  # type: ignore
        ner_res[r["entity_group"]].append(input[r["start"] : r["end"]])  # type: ignore
    print(ner_res)


if __name__ == '__main__':

    datasets_name = 'peoples-daily-ner/peoples_daily_ner'  # 人民日报
    ner_datasets = get_datasets(datasets_name)
    label_list = ner_datasets["train"].features["ner_tags"].feature.names  # type: ignore

    model_id = 'hfl/chinese-macbert-base'
    tokenizer = load_tokenizer(model_id)

    ckpt_dir = None  # 从头训练
    ckpt_dir = CHECKPOINTS_DIR.joinpath('checkpoint-981')

    # train(ckpt_dir=ckpt_dir)
    infer(ckpt_dir=ckpt_dir)
