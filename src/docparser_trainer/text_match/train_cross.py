from pathlib import Path

import evaluate  # type:ignore
from datasets import load_dataset  # type:ignore
from transformers import (  # type:ignore
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from util_common.decorator import proxy

from docparser_trainer._cfg import CKPT_ROOT, DATA_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


@proxy(http_proxy='127.0.0.1:17890', https_proxy='127.0.0.1:17890')
def get_evaluator():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    return accuracy, f1


def preprocess_datasets(tokenizer, datasets):
    def process_token(examples):
        tokenized = tokenizer(
            examples['sentence1'],
            examples['sentence2'],
            padding='max_length',
            truncation=True,
            max_length=128,
        )
        tokenized['labels'] = [float(label) for label in examples['label']]
        return tokenized

    tokenized_datasets = datasets.map(
        process_token,
        batched=True,
        remove_columns=datasets['train'].column_names,  # type: ignore
    )

    return tokenized_datasets


def train(model):
    acc_metric, f1_metric = get_evaluator()

    def eval_metric(pred):
        predictions, labels = pred
        predictions = [int(p > 0.5) for p in predictions]
        labels = [int(l) for l in labels]
        acc = acc_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels)
        acc.update(f1)  # type: ignore
        return acc

    tokenized_datasets = preprocess_datasets(tokenizer, datasets)

    train_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=eval_metric,  # type: ignore
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()


def infer(model):
    from transformers import pipeline

    # model.config.id2label = {0: 'not_match', 1: 'match'}
    pipe = pipeline(
        'text-classification',
        model=model,
        tokenizer=tokenizer,
        device=model.device,
    )
    pipe({"text": "我喜欢北京", "text_pair": "北京是我喜欢的城市"}, function_to_apply="none")


if __name__ == '__main__':
    dataset = load_dataset(
        'json',
        data_files=str(DATA_ROOT.joinpath('CLUEbenchmark/simclue_public/train_pair.json')),
        split='train',
    )
    dataset = dataset.select(range(10000))  # type: ignore
    datasets = dataset.train_test_split(test_size=0.1)  # type: ignore

    model_id = 'hfl/chinese-macbert-base'
    tokenizer = load_tokenizer(model_id)

    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-text-match')
    checkpoints_dir = CKPT_ROOT.joinpath('text_match/text_similarity_cross')
    ckpt_version: str | None = None
    ckpt_dir: Path | None = checkpoints_dir / ckpt_version if ckpt_version else None
    model = load_model(
        model_id,
        ckpt_dir=ckpt_dir,
        model_cls=AutoModelForSequenceClassification,
        pretrained_dir=pretrained_dir,
        num_labels=1,
    )

    train(model)
    infer(model)
