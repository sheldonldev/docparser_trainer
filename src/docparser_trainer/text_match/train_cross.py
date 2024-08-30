from pathlib import Path

from transformers import (  # type:ignore
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from docparser_trainer._cfg import CKPT_ROOT, DATA_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_evaluator, load_model, load_tokenizer

setup_env()


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


def train(model, tokenizer, datasets, checkpoints_dir):
    acc_metric = load_evaluator("accuracy")
    f1_metric = load_evaluator("f1")

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


def main(
    datasets,
    model_id,
    pretrained_dir,
    checkpoints_dir,
    ckpt_version=None,
):
    tokenizer = load_tokenizer(model_id)
    model = load_model(
        model_id,
        ckpt_dir=Path(checkpoints_dir) / ckpt_version if ckpt_version else None,
        model_cls=AutoModelForSequenceClassification,
        pretrained_dir=pretrained_dir,
        num_labels=1,
    )

    train(model, tokenizer, datasets, checkpoints_dir)


if __name__ == '__main__':
    dataset = get_datasets(
        'json',
        data_files=str(DATA_ROOT.joinpath('CLUEbenchmark/simclue_public/train_pair.json')),
        split='train',
    )
    dataset = dataset.select(range(10000))  # type: ignore
    datasets = dataset.train_test_split(test_size=0.1)  # type: ignore

    model_id = 'hfl/chinese-macbert-base'
    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-text-match')
    checkpoints_dir = CKPT_ROOT.joinpath('text_match/text_similarity_cross')

    main(
        datasets,
        model_id,
        pretrained_dir,
        checkpoints_dir,
    )
