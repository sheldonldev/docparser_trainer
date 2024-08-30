from pathlib import Path

from datasets import load_dataset  # type:ignore
from transformers import Trainer, TrainingArguments  # type:ignore

from docparser_trainer._cfg import CKPT_ROOT, DATA_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_evaluator, load_model, load_tokenizer
from docparser_trainer.text_match.models import DualModel

setup_env()


def preprocess_datasets(tokenizer, datasets):
    def process_token(examples):
        sentences = []
        labels = []
        for sen1, sen2, label in zip(
            examples['sentence1'], examples['sentence2'], examples['label']
        ):
            sentences.append(sen1)
            sentences.append(sen2)
            labels.append(1 if int(label) == 1 else -1)
        tokenized = tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=128,
        )
        tokenized = {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in tokenized.items()}
        tokenized['labels'] = labels
        return tokenized

    tokenized_datasets = datasets.map(
        process_token,
        batched=True,
        remove_columns=datasets['train'].column_names,  # type: ignore
    )

    return tokenized_datasets


def train(model, tokenizer, datasets):
    acc_metric = load_evaluator('accuracy')
    f1_metric = load_evaluator('f1')

    def eval_metric(pred):
        predictions, labels = pred
        predictions = [int(p > 0.7) for p in predictions]
        labels = [int(l > 0) for l in labels]
        acc = acc_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels)
        acc.update(f1)  # type: ignore
        return acc

    tokenized_datasets = preprocess_datasets(tokenizer, datasets)

    train_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=1,
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
        model_cls=DualModel,
        pretrained_dir=pretrained_dir,
    )
    train(model, tokenizer, datasets)


if __name__ == '__main__':
    dataset = load_dataset(
        'json',
        data_files=str(DATA_ROOT.joinpath('CLUEbenchmark/simclue_public/train_pair.json')),
        split='train',
    )
    dataset = dataset.select(range(10000))  # type: ignore
    datasets = dataset.train_test_split(test_size=0.1)  # type: ignore

    model_id = 'hfl/chinese-macbert-base'
    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-text-match-dual')
    checkpoints_dir = CKPT_ROOT.joinpath('text_match/text_similarity_dual')
    ckpt_version = 'checkpoint-282'

    main(
        datasets,
        model_id,
        pretrained_dir,
        checkpoints_dir,
        ckpt_version=ckpt_version,
    )
