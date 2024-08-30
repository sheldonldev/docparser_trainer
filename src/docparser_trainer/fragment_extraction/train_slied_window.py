from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
from transformers import (  # type: ignore
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_model, load_tokenizer
from docparser_trainer.fragment_extraction.cmrc_eval import evaluate_cmrc

setup_env()


def preprocess_datasets(tokenizer, datasets):
    def get_context_start(sequence_ids):
        """
        获取 context 的起始 token 位置
        """
        start = sequence_ids.index(1)
        end = sequence_ids.index(None, start + 1) - 1
        return start, end

    def position_mapping_to_token(answer_start, answer_end, offset, context_start, context_end):
        """
        将 char 位置映射到 token 位置
        """
        start_token = 0
        end_token = 0

        if offset[context_start][0] > answer_end or offset[context_end][1] < answer_start:
            return start_token, end_token

        for i in range(context_start, context_end + 1):
            if offset[i][0] <= answer_start < offset[i][1]:
                start_token = i
                break

        for i in range(start_token, context_end + 1):
            if offset[i][0] < answer_end <= offset[i][1]:
                end_token = i
                break

        return start_token, end_token

    def process_token(examples):
        def print_answer():
            print(">>>")
            print("question:", examples['question'][example_idx])
            print("answer:", examples['context'][example_idx][start_char:end_char])
            print(
                "check answer:",
                examples['context'][example_idx][offset[start_token][0] : offset[end_token][1]],
            )
            print(
                "check decode:",
                tokenizer.decode(
                    tokenized['input_ids'][truncate_idx][start_token : end_token + 1],
                ),
            )
            print("<<<")

        tokenized = tokenizer(
            text=examples['question'],
            text_pair=examples['context'],
            max_length=512,
            truncation='only_second',  # 只截断 text_pair
            padding='max_length',
            return_offsets_mapping=True,  # 返回每个 token 对应的 char 位置
            return_overflowing_tokens=True,  # 截断后，将原始数据复制一份，并添加到 batch 中
            stride=128,
        )

        overflow_mapping = tokenized.pop('overflow_to_sample_mapping')

        start_positions = []
        end_positions = []
        example_ids = []
        for truncate_idx, example_idx in enumerate(overflow_mapping):
            answer = examples['answers'][example_idx]
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])

            offset = tokenized['offset_mapping'][truncate_idx]
            context_start, content_end = get_context_start(tokenized.sequence_ids(truncate_idx))
            start_token, end_token = position_mapping_to_token(
                start_char, end_char, offset, context_start, content_end
            )

            if truncate_idx < 3:
                print_answer()

            start_positions.append(start_token)
            end_positions.append(end_token)
            example_ids.append(examples['id'][example_idx])

            tokenized['offset_mapping'][truncate_idx] = [
                (o if tokenized.sequence_ids(truncate_idx)[k] == 1 else None)
                for k, o in enumerate(offset)
            ]

        tokenized['start_positions'] = start_positions
        tokenized['end_positions'] = end_positions
        tokenized['example_ids'] = example_ids
        return tokenized

    tokenized_datasets = datasets.map(
        process_token,
        batched=True,
        remove_columns=datasets['train'].column_names,
    )
    return tokenized_datasets


def get_result(start_logits, end_logits, examples, features):

    predictions: Dict[str, str] = {}
    references: Dict[str, str] = {}

    # example 和 feature 的映射
    example_to_feature = defaultdict(list)
    for idx, example_id in enumerate(features['example_ids']):
        example_to_feature[example_id].append(idx)

    # 最多个最优答案候选
    n_best = 20
    # 答案的最长 token
    max_answer_length = 32

    for example in examples:
        example_id = example['id']
        context = example['context']
        answers = []
        for feature_id in example_to_feature[example_id]:
            start_logit = start_logits[feature_id]
            end_logit = end_logits[feature_id]
            offset = features[feature_id]['offset_mapping']
            start_ids = np.argsort(start_logit)[::-1][:n_best].tolist()
            end_ids = np.argsort(end_logit)[::-1][:n_best].tolist()
            for start_id in start_ids:
                for end_id in end_ids:
                    if offset[start_id] is None or offset[end_id] is None:
                        continue
                    if end_id < start_id or end_id - start_id + 1 > max_answer_length:
                        continue
                    answers.append(
                        {
                            "text": context[offset[start_id][0] : offset[end_id][1]],
                            "score": start_logit[start_id] + end_logit[end_id],
                        }
                    )
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x['score'])
            predictions[example_id] = best_answer['text']
        else:
            predictions[example_id] = ""
        references[example_id] = example['answers']['text']
    return predictions, references


def train(model, tokenizer, datasets, checkpoints_dir):

    def eval_metric(pred):
        start_logits, end_logits = pred[0]
        if start_logits.shape[0] == len(tokenized_datasets['validation']):
            p, r = get_result(
                start_logits,
                end_logits,
                datasets['validation'],
                tokenized_datasets['validation'],
            )
        else:
            p, r = get_result(
                start_logits,
                end_logits,
                datasets['test'],
                tokenized_datasets['test'],
            )
        return evaluate_cmrc(p, r)

    tokenized_datasets = preprocess_datasets(tokenizer, datasets)

    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
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
        data_collator=DefaultDataCollator(),
        compute_metrics=eval_metric,
    )
    trainer.train()


def main(datasets, model_id, pretrained_dir, checkpoints_dir, ckpt_version=None):
    tokenizer = load_tokenizer(model_id)
    model = load_model(
        model_id,
        ckpt_dir=Path(checkpoints_dir) / ckpt_version if ckpt_version else None,
        model_cls=AutoModelForQuestionAnswering,
        pretrained_dir=pretrained_dir,
    )
    train(model, tokenizer, datasets, checkpoints_dir)


if __name__ == '__main__':
    datasets_name = 'hfl/cmrc2018'
    datasets = get_datasets(datasets_name)

    model_id = 'hfl/chinese-macbert-base'
    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-mrc-fragment-extraction')
    checkpoints_dir = CKPT_ROOT.joinpath("machine_reading_comprehension/fragment_extraction_slide")
    ckpt_version: str | None = None

    main(
        datasets,
        model_id,
        pretrained_dir,
        checkpoints_dir,
        ckpt_version=ckpt_version,
    )
