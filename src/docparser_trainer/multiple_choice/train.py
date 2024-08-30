from pathlib import Path

import evaluate  # type:ignore
import numpy as np
import torch
from transformers import AutoModelForMultipleChoice, Trainer, TrainingArguments  # type:ignore
from util_common.decorator import proxy

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


@proxy(http_proxy='127.0.0.1:17890', https_proxy='127.0.0.1:17890')
def get_evaluator():
    accuracy = evaluate.load("accuracy")
    return accuracy


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


def train(model):

    accuracy = get_evaluator()

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


def infer(model):
    class MultipleChoicePipeline:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.device = model.device

        def preprocess(self, context, question, choices):
            cs, qcs = [], []
            for choice in choices:
                cs.append(context)
                qcs.append(question + ' ' + choice)
            return tokenizer(
                cs,
                qcs,
                max_length=256,
                truncation="only_first",
                return_tensors='pt',  # 返回 pytorch 的张量
            )

        def predict(self, inputs):
            inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}
            return self.model(**inputs).logits

        def postprocess(self, logits, choices):
            pred = torch.argmax(logits, dim=-1).cpu().item()
            return choices[pred]

        def __call__(self, context, question, choices):
            inputs = self.preprocess(context, question, choices)
            logits = self.predict(inputs)
            result = self.postprocess(logits, choices)
            return result

    pipe = MultipleChoicePipeline(model, tokenizer)
    print(
        pipe('小明在北京上班', '小明在哪里上班', ['北京', '上海'])
    )  # 预测的时候没有 batch 的概念，不需要 padding 也不限制于选项数量


if __name__ == '__main__':
    datasets_name = 'clue/clue'
    datasets = get_datasets(datasets_name, 'c3')
    datasets.pop('test')

    model_id = 'hfl/chinese-macbert-base'
    tokenizer = load_tokenizer(model_id)

    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-mrc-multiple-choices')
    checkpoints_dir = CKPT_ROOT.joinpath('machine_reading_comprehension/multi_choice')
    ckpt_version: str | None = None
    ckpt_dir: Path | None = checkpoints_dir / ckpt_version if ckpt_version else None

    model = load_model(
        model_id,
        ckpt_dir=ckpt_dir,
        model_cls=AutoModelForMultipleChoice,
        pretrained_dir=pretrained_dir,
    )

    train(model)
    infer(model)
