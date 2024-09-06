"""
参考：https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/sft/finetune_clm_lora.py
"""

import math
from pathlib import Path
from typing import Literal

import torch
import transformers  # type: ignore
from peft.tuners.lora.config import LoraConfig
from peft.utils.other import prepare_model_for_kbit_training
from peft.utils.peft_types import TaskType
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.testing_utils import CaptureLogger  # type:ignore
from transformers.training_args import OptimizerNames  # type:ignore
from transformers.utils.quantization_config import BitsAndBytesConfig  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_evaluator, load_model, load_tokenizer
from docparser_trainer._optimization.peft import lora_tuning
from docparser_trainer._utils.model_util import print_model_info, print_params_dtype
from docparser_trainer._utils.seed import seed_everything
from docparser_trainer.llm_training.chat_single_round import chat_llama3

setup_env()
seed_everything(42)


def tokenize_datasets(
    tokenizer,
    datasets,
    max_train_samples=None,
    max_eval_samples=None,
):
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_llama3(examples):
        with CaptureLogger(tok_logger):
            output = tokenizer(
                [x for x in examples['text']],
                truncation=True,
                max_length=4096,
                padding=False,
                return_tensors=None,
            )
            output['labels'] = output['input_ids'].copy()
        return output

    tokenized_datasets = datasets.map(
        tokenize_llama3,
        batched=True,
        remove_columns=datasets['train'].column_names,
    )

    print(">>> Example Decoded:")
    print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
    print(tokenizer.decode(tokenized_datasets["train"][0]["labels"]))

    if datasets.get("train"):
        train_dataset = tokenized_datasets["train"]
        if max_train_samples:
            train_dataset = train_dataset.select(
                range(min(max_train_samples, len(train_dataset))),
            )
        print(">>> Train Dataset Size:", len(train_dataset))
    else:
        train_dataset = None

    if datasets.get("validation"):
        eval_dataset = tokenized_datasets["validation"]
        if max_eval_samples:
            eval_dataset = eval_dataset.select(
                range(min(max_eval_samples, len(eval_dataset))),
            )
        print(">>> Eval Dataset Size:", len(eval_dataset))
    else:
        eval_dataset = None

    return train_dataset, eval_dataset


def model_init(
    model_id,
    checkpoints_dir,
    ckpt_version,
    tokenizer,
    torch_dtype,
    load_in_bits,
    gradient_checkpointing,
    lora_config: LoraConfig,
):
    model = load_model(
        model_id,
        ckpt_dir=Path(checkpoints_dir) / ckpt_version if ckpt_version else None,
        model_cls=AutoModelForCausalLM,  # 官方用的CausalLM
        torch_dtype=torch_dtype,
        load_in_8bit=True if load_in_bits == 8 else False,  # 一般不会用, 会用 bitsandbytes 4bitQ
        quantization_config=(
            BitsAndBytesConfig(  # 节约加载模型的显存
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if load_in_bits == 4
            else None
        ),
    )

    # Test
    print(chat_llama3(model, tokenizer, '考试有什么技巧'))

    if load_in_bits is not None:
        gradient_checkpointing = True
        prepare_model_for_kbit_training(model)

    model = lora_tuning(model, lora_config)

    # 不保存梯度，显存优化
    if gradient_checkpointing is True:
        model.enable_input_require_grads()
        model.config.use_cache = False  # type:ignore

    return model


def train_and_save_result(trainer: Trainer):

    train_result = trainer.train()
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
    datasets,
    model_id,
    checkpoints_dir,
    ckpt_version=None,
    torch_dtype=torch.float32,
    load_in_bits: Literal[4, 8] | None = None,
    gradient_checkpointing=True,
    max_train_samples=None,
    max_eval_samples=None,
):
    tokenizer = load_tokenizer(
        model_id,
        padding_side='right',
    )
    tokenizer.pad_token = tokenizer.eos_token  # llama3 官方写法

    model = model_init(
        model_id=model_id,
        checkpoints_dir=checkpoints_dir,
        ckpt_version=ckpt_version,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        load_in_bits=load_in_bits,
        gradient_checkpointing=gradient_checkpointing,
        lora_config=LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=64,  # 权重缩放控制 lora_alpha / r
            lora_dropout=0.1,
        ),
    )

    train_dataset, eval_dataset = tokenize_datasets(
        tokenizer,
        datasets,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = load_evaluator('accuracy')

    def compute_metrics(eval_preds):
        import numpy as np

        preds, labels = eval_preds
        true_labels = [
            [l for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        true_preds = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        print(">>> Evaluate Example:")
        print(tokenizer.decode(true_labels[0]))
        print(tokenizer.decode(true_preds[0]))

        labels = np.array(labels)[:, 1:].reshape(-1)
        preds = np.array(preds)[:, 1:].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    # 半精度训练溢出优化
    if model.dtype == torch.float16:
        adam_epsilon = 1e-4
        model.half()
    else:
        adam_epsilon = 1e-8

    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=2,
        do_train=True if train_dataset else False,
        do_eval=True if eval_dataset else False,
        eval_strategy='steps',
        eval_steps=50,
        warmup_steps=400,
        save_strategy='steps',
        save_steps=50,
        num_train_epochs=1,
        learning_rate=1e-4,
        optim=OptimizerNames.PAGED_ADAMW,  # bitsandbytes installed
        logging_strategy='steps',
        logging_steps=10,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=gradient_checkpointing,
        adam_epsilon=adam_epsilon,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print_params_dtype(model)
    print_model_info(model)

    train_and_save_result(trainer)

    # Test
    print(chat_llama3(model, tokenizer, '考试有什么技巧'))


if __name__ == '__main__':
    from docparser_trainer.llm_training.data import preprocess_alpaca_datasets_for_llama3

    datasets = preprocess_alpaca_datasets_for_llama3()
    model_id = 'FlagAlpha/Llama3-Chinese-8B-Instruct'
    checkpoints_dir = CKPT_ROOT / f"llm/{model_id}-lora"
    ckpt_version: str | None = None

    train(
        datasets,
        model_id,
        checkpoints_dir,
        ckpt_version=ckpt_version,
        torch_dtype=torch.bfloat16,
        max_train_samples=5000,
        max_eval_samples=100,
    )
