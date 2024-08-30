from pathlib import Path

import numpy as np
from transformers import AutoModelForMultipleChoice, Trainer, TrainingArguments  # type:ignore

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_evaluator, load_model, load_tokenizer

setup_env()


def preprocess_datasets(tokenizer, datasets):
    def process_token(examples):
        context = []
        question_choice = []
        labels = []
        for idx in range(len(examples['question'])):
            ctx = "\n".join(examples['context'][idx])
            question = examples['question'][idx]
            choices = examples['choice'][idx]
            answer = examples['answer'][idx]
            for choice in choices:
                context.append(ctx)
                question_choice.append(question + ' ' + choice)

            # 填充选项都为 4 项, tokenize 的时候，再每 4 个截断为一组完整的问答
            if len(choices) < 4:
                for _ in range(4 - len(choices)):
                    context.append(ctx)
                    question_choice.append(question + ' ' + '不知道')

            labels.append(choices.index(answer))

        tokenized = tokenizer(
            context,
            question_choice,
            max_length=256,
            truncation="only_first",
            padding='max_length',
        )

        # input_ids: 4000 * 256 -> 1000 * 4 * 256
        tokenized = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized.items()}
        tokenized['labels'] = labels
        return tokenized

    tokenized_datasets = datasets.map(
        process_token,
        batched=True,
    )

    return tokenized_datasets


def train(model, tokenizer, datasets, checkpoints_dir):
    accuracy = load_evaluator("accuracy")

    def eval_metric(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    tokenized_datasets = preprocess_datasets(tokenizer, datasets)

    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,  # 可能会提速
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=eval_metric,
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
        model_cls=AutoModelForMultipleChoice,
        ckpt_dir=Path(checkpoints_dir) / ckpt_version if ckpt_version else None,
        pretrained_dir=pretrained_dir,
    )
    train(model, tokenizer, datasets, checkpoints_dir)


if __name__ == '__main__':
    datasets_id = 'clue/clue'
    sub_datasets_name = 'c3'
    datasets = get_datasets(datasets_id, sub_name=sub_datasets_name)
    datasets.pop('test')

    model_id = 'hfl/chinese-macbert-base'
    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-mrc-multiple-choices')
    checkpoints_dir = CKPT_ROOT.joinpath('machine_reading_comprehension/multi_choice')

    main(
        datasets,
        model_id,
        pretrained_dir,
        checkpoints_dir,
    )
