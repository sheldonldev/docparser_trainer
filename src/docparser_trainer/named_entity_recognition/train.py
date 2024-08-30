from pathlib import Path

import numpy as np
from transformers import (  # type: ignore
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_evaluator, load_model, load_tokenizer

setup_env()


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


def train(model, tokenizer, datasets, label_list, checkpoints_dir):
    seqeval = load_evaluator("seqeval")

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

    tokenized_datasets = preprocess_datasets(tokenizer, datasets)

    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
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


def main(
    datasets,
    label_list,
    model_id,
    pretrained_dir,
    checkpoints_dir,
    ckpt_version=None,
):

    tokenizer = load_tokenizer(model_id)
    model = load_model(
        model_id,
        ckpt_dir=Path(checkpoints_dir) / ckpt_version if ckpt_version else None,
        pretrained_dir=pretrained_dir,
        model_cls=AutoModelForTokenClassification,
        num_lables=len(label_list),
    )

    train(model, tokenizer, datasets, label_list, checkpoints_dir)


if __name__ == '__main__':
    datasets_id = 'peoples-daily-ner/peoples_daily_ner'  # 人民日报
    datasets = get_datasets(datasets_id)
    label_list = datasets["train"].features["ner_tags"].feature.names  # type: ignore

    model_id = 'hfl/chinese-macbert-base'
    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-mrc-fragment-extraction')
    checkpoints_dir = CKPT_ROOT.joinpath("named_entity_recognition")

    main(
        datasets,
        label_list,
        model_id,
        pretrained_dir,
        checkpoints_dir,
    )
