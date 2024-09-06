import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.optim as optim
from transformers import (  # type: ignore
    BertTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

from docparser_trainer._augment.augment_text import func_list
from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_model, load_tokenizer
from docparser_trainer._utils.seed import seed_everything
from docparser_trainer.text_multilabel_classification.data import (
    label_settings,
    preprocess_datasets,
)
from docparser_trainer.text_multilabel_classification.metrics import compute_metrics
from docparser_trainer.text_multilabel_classification.models import MultiLabelClassifier

setup_env()
seed_everything(42)


class GenerativeDataCollatorWithPadding(DataCollatorWithPadding):
    def __init__(self, generative_prob=0.3, **kwargs):
        super().__init__(**kwargs)
        self.generative_prob = generative_prob

    def _augment_text(self, text):
        for f in func_list:
            text = f(text, prob=self.generative_prob)
        return text

    def __call__(self, features: List[Dict[str, Any]], model: PreTrainedModel) -> Dict[str, Any]:
        augmented_features: List[Dict[str, Any]] = []

        for feature in features:
            if model.training is True:
                text = self._augment_text(feature['text'])
            else:
                text = feature['text']

            encoding = self.tokenizer.encode_plus(
                text,
                padding=self.padding,
                max_length=self.max_length,
                truncation=True,
                return_token_type_ids=False,
                return_attention_mask=True,
                return_tensors=self.return_tensors,
            )
            input_ids = encoding['input_ids'].flatten()  # type: ignore
            attention_mask = encoding['attention_mask'].flatten()  # type: ignore
            augmented_features.append(
                dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=torch.tensor(feature['labels']),
                ),
            )

        return super().__call__(augmented_features)


def train_and_save_result(trainer: Trainer, resume_from_checkpoint: bool):

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = (
        len(
            trainer.train_dataset,  # type: ignore
        )
        if trainer.train_dataset
        else 0
    )
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    metrics["eval_samples"] = (
        len(
            trainer.eval_dataset,  # type: ignore
        )
        if trainer.eval_dataset
        else 0
    )
    try:
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity
    except OverflowError:
        perplexity = float("inf")
        metrics["perplexity"] = perplexity
    except KeyError:
        pass

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def train(
    model_id: str,
    output_dir: Path,
    ckpt_dir: Path | None = None,
    resume_from_checkpoint: bool = False,
):
    '''
    model_id: 模型对应的 huggingface 的 model_id, 用于加载 tokenizer 和 model,
        model_id 会对应一个默认的本地存储，第一次会从 huggingface 下载，之后会从本地存储加载，
        本地存储需要配置环境变量 PRETRAINED_ROOT, 可以用 docparser_trainer._cfg.setup_env() 配上.
    output_dir: 用于模型训练后保存检查点.
    ckpt_dir: 如果是 None，加载基座模型，否则从指定检查点加载模型.
    resume_from_checkpoint: 在指定了检查点的情况下，且 resume_from_checkpoint 为 True，
        则训练轮次会在原有检查点基础上继续.
    '''
    train_dataset, eval_dataset = preprocess_datasets()
    tokenizer = load_tokenizer(model_id, tokenizer_cls=BertTokenizer)

    if ckpt_dir is None:
        resume_from_checkpoint = False

    model = load_model(
        model_id,
        ckpt_dir=ckpt_dir,
        model_cls=MultiLabelClassifier,
        num_labels=len(label_settings),
        classifier_dropout=0.3,
    )
    model.config.id2label = {i: label_settings[i][0] for i in range(len(label_settings))}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    model.config.label_weights = [0.2, 0.2, 0.2, 0.4, 0.2]

    # 带权重的 adam, 权重见 model.config.label_weights
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_dir=str(output_dir / 'log'),
        logging_steps=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        remove_unused_columns=False,
        num_train_epochs=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    num_training_steps = int(
        len(train_dataset)
        // training_args.per_device_train_batch_size
        * training_args.num_train_epochs
    )
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    data_collator = GenerativeDataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='max_length',
        max_length=2048,
        generative_prob=0.5,  # 文本增强器里每个 func 被调用的概率
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=lambda x: data_collator(x, model=model),
        optimizers=(optimizer, lr_scheduler),
    )

    train_and_save_result(trainer, resume_from_checkpoint=resume_from_checkpoint)


if __name__ == '__main__':

    model_id = 'schen/longformer-chinese-base-4096'
    output_dir = (
        CKPT_ROOT / 'text_multi_classification' / 'customs_declaration_5_labels' / 'trial_1'
    )
    ckpt_dir = output_dir.joinpath('checkpoint-7600')

    train(
        model_id,
        output_dir,
        ckpt_dir=ckpt_dir,
        resume_from_checkpoint=True,
    )
