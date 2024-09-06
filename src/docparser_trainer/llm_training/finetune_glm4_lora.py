"""
参考：https://github.com/THUDM/GLM-4/blob/main/finetune_demo/finetune.py
"""

import functools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Optional

import jieba  # type: ignore
import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
from peft.tuners.lora.config import LoraConfig
from peft.utils.other import prepare_model_for_kbit_training
from peft.utils.peft_types import TaskType
from rouge_chinese import Rouge  # type: ignore
from torch import nn
from transformers import AutoModelForCausalLM  # type: ignore
from transformers import Seq2SeqTrainingArguments  # type: ignore
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import EvalPrediction
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
from transformers.training_args import OptimizerNames  # type: ignore
from transformers.utils.quantization_config import BitsAndBytesConfig  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_model, load_tokenizer
from docparser_trainer._optimization.peft import lora_tuning
from docparser_trainer._utils.model_util import print_model_info, print_params_dtype
from docparser_trainer._utils.seed import seed_everything
from docparser_trainer.llm_training.chat_single_round import chat_glm4

setup_env()
seed_everything(42)


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = (
            [feature['output_ids'] for feature in features]
            if 'output_ids' in features[0].keys()
            else None
        )
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                    (max_output_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        return super().__call__(features, return_tensors)


class Seq2SeqTrainer(_Seq2SeqTrainer):
    # Not Support for apex
    def training_step(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        detached_loss = loss.detach() / self.args.gradient_accumulation_steps
        del inputs
        torch.cuda.empty_cache()
        return detached_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys=None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        with torch.no_grad():  # Ensure no gradient computation
            if self.args.predict_with_generate:
                output_ids = inputs.pop('output_ids')
            input_ids = inputs['input_ids']

            loss, generated_tokens, labels = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
            )

            generated_tokens = generated_tokens[:, input_ids.size()[1] :]
            labels = output_ids

            del inputs, input_ids, output_ids
            torch.cuda.empty_cache()

        return loss, generated_tokens, labels


def tokenize_datasets(
    tokenizer,
    datasets,
    max_train_samples=None,
    max_eval_samples=None,
):

    def process_message(message):
        if 'tools' in message and message['role'] == 'system':
            for tool in message['tools']:
                parameters = tool['function']['parameters']['properties']
                tool['function']['parameters']['properties'] = {
                    k: v for k, v in parameters.items() if v is not None
                }
        elif 'tools' in message:
            del message['tools']
        return message

    def process_batch(
        batch: Mapping[str, Sequence],
        max_input_length: int = 1024,
        max_output_length: int = 1024,
        combine: bool = True,
    ) -> dict[str, list]:
        batched_conv = batch['messages']
        batched_input_ids = []
        batched_labels = []
        for conv in batched_conv:
            input_ids = [151331, 151333]
            loss_masks = [False, False]
            if combine:
                new_input_ids = tokenizer.apply_chat_template(
                    conv, tokenize=True, return_dict=False
                )
                input_ids = new_input_ids
                loss_masks = [False] * len(input_ids)
                last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1
                for j in range(last_assistant_index + 1, len(input_ids)):
                    loss_masks[j] = True
            else:
                for message in conv:
                    message = process_message(message)
                    loss_mask_val = (
                        False if message['role'] in ('system', 'user', 'observation') else True
                    )
                    new_input_ids = tokenizer.apply_chat_template(
                        [message], tokenize=True, return_dict=False
                    )[2:]
                    input_ids += new_input_ids
                    loss_masks += [loss_mask_val] * len(new_input_ids)

            input_ids.append(151336)  # EOS for chat
            loss_masks = [False, *loss_masks]
            labels = []
            for input_id, mask in zip(input_ids, loss_masks):
                if mask:
                    labels.append(input_id)
                else:
                    labels.append(-100)
            max_length = max_input_length + max_output_length + 1
            batched_input_ids.append(input_ids[:max_length])
            batched_labels.append(labels[:max_length])

        del batched_conv, conv, input_ids, loss_masks, new_input_ids, labels
        torch.cuda.empty_cache()

        return {'input_ids': batched_input_ids, 'labels': batched_labels}

    def process_batch_eval(
        batch: Mapping[str, Sequence],
        max_input_length: int = 1024,
        max_output_length: int = 1024,
        combine: bool = True,
    ) -> dict[str, list]:
        batched_conv = batch['messages']
        batched_input_ids = []
        batched_output_ids = []

        for conv in batched_conv:
            if combine:
                new_input_ids = tokenizer.apply_chat_template(
                    conv, tokenize=True, return_dict=False
                )
                input_ids = new_input_ids
                last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1
                output_prompt, output_ids = (
                    input_ids[:1],
                    input_ids[last_assistant_index:],
                )
                output_ids.append(151336)
                batched_input_ids.append(input_ids[:max_input_length] + output_prompt[:1])
                batched_output_ids.append(output_ids[:max_output_length])
            else:
                input_ids = [151331, 151333]
                for message in conv:
                    if len(input_ids) >= max_input_length:
                        break
                    else:
                        message = process_message(message)
                        new_input_ids = tokenizer.apply_chat_template(
                            [message], tokenize=True, return_dict=False
                        )[2:]
                        if message['role'] == 'assistant':
                            output_prompt, output_ids = (
                                new_input_ids[:1],
                                new_input_ids[1:],
                            )
                            output_ids.append(151336)
                            batched_input_ids.append(
                                input_ids[:max_input_length] + output_prompt[:1]
                            )
                            batched_output_ids.append(output_ids[:max_output_length])
                        input_ids += new_input_ids

        del batched_conv, conv, input_ids, new_input_ids, output_prompt, output_ids
        torch.cuda.empty_cache()

        return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}

    train_dataset, eval_dataset = datasets
    _train_dataset = train_dataset.map(
        process_batch,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    _eval_dataset = eval_dataset.map(
        process_batch_eval,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    print(">>> Example Test Decoded:")
    print(tokenizer.decode(_train_dataset[0]["input_ids"]))
    print(tokenizer.decode(_train_dataset[0]["labels"]))

    print(">>> Example Eval Decoded:")
    print(tokenizer.decode(_eval_dataset[0]["input_ids"]))
    print(tokenizer.decode(_eval_dataset[0]["output_ids"]))

    if max_train_samples:
        _train_dataset = _train_dataset.select(
            range(min(max_train_samples, len(_train_dataset))),
        )
    print(">>> Train Dataset Size:", len(_train_dataset))

    if max_eval_samples:
        _eval_dataset = _eval_dataset.select(
            range(min(max_eval_samples, len(_eval_dataset))),
        )
    print(">>> Eval Dataset Size:", len(_eval_dataset))

    return _train_dataset, _eval_dataset


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
    print(chat_glm4(model, tokenizer, '考试有什么技巧'))

    if load_in_bits is not None:
        gradient_checkpointing = True
        prepare_model_for_kbit_training(model)

    model = lora_tuning(model, lora_config)

    # 不保存梯度，显存优化
    if gradient_checkpointing is True:
        model.enable_input_require_grads()
        model.config.use_cache = False  # type:ignore

    return model


def train_and_save_result(trainer: Seq2SeqTrainer):

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
        use_fast=False,
        padding_side='left',
    )

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
            target_modules=["query_key_value"],  # 查看 model.named_parameters, 支持正则
            modules_to_save=None,  # 用法同上
            r=8,
            lora_alpha=32,  # 权重缩放控制 lora_alpha / r
            lora_dropout=0.1,
        ),
    )

    train_dataset, eval_dataset = tokenize_datasets(
        tokenizer,
        datasets=datasets,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

    rouge = Rouge()

    def compute_metrics(eval_preds: EvalPrediction, tokenizer):
        batched_pred_ids, batched_label_ids = eval_preds
        metrics_dct: dict = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
        for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
            pred_ids = [x for x in pred_ids if x not in [-100, tokenizer.eos_token_id]]
            label_ids = [x for x in label_ids if x not in [-100, tokenizer.eos_token_id]]
            pred_txt = (
                tokenizer.decode(pred_ids)
                .replace('<|user|>', '')
                .replace('<|assistant|>', '')
                .strip()
            )
            label_txt = (
                tokenizer.decode(label_ids)
                .replace('<|user|>', '')
                .replace('<|assistant|>', '')
                .strip()
            )
            pred_tokens = list(jieba.cut(pred_txt))
            label_tokens = list(jieba.cut(label_txt))

            pred = ' '.join(pred_tokens)
            label = ' '.join(label_tokens)
            if len(pred) == 0:
                for k, v in metrics_dct.items():
                    v.append(0)
                continue

            scores = rouge.get_scores(pred, label)
            for k, v in scores[0].items():
                metrics_dct[k].append(round(v['f'] * 100, 4))
            metrics_dct['bleu-4'].append(
                sentence_bleu(
                    [label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method3
                )
            )
        return {k: np.mean(v) for k, v in metrics_dct.items()}

    # 半精度训练溢出优化
    if model.dtype == torch.float16:
        adam_epsilon = 1e-4
        model.half()
    else:
        adam_epsilon = 1e-8

    args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=2,
        eval_strategy='steps',
        eval_steps=50,
        warmup_steps=400,
        save_strategy='steps',
        save_steps=50,
        num_train_epochs=1,
        learning_rate=5e-4,
        optim=OptimizerNames.PAGED_ADAMW,  # bitsandbytes installed
        logging_strategy='steps',
        logging_steps=10,
        predict_with_generate=True,
        bf16=True,  # 训练
        bf16_full_eval=True,  # 评估
        gradient_checkpointing=gradient_checkpointing,
        adam_epsilon=adam_epsilon,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            padding='longest',
            return_tensors="pt",
        ),
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    print_params_dtype(model)
    print_model_info(model)

    train_and_save_result(trainer)

    # Test
    print(chat_glm4(model, tokenizer, '考试有什么技巧'))


if __name__ == '__main__':
    from docparser_trainer.llm_training.data import preprocess_alpaca_datasets_for_glm4

    datasets = preprocess_alpaca_datasets_for_glm4()
    model_id = 'THUDM/glm-4-9b-chat'
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
