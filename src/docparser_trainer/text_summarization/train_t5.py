import numpy as np
from rouge_chinese import Rouge  # type: ignore
from transformers import (  # type: ignore
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


def preprocess_datasets(tokenizer: T5Tokenizer, datasets):
    def preprocess_function(examples):
        contents = ["摘要生成:\n" + e["content"] for e in examples["data"]]  # 原始 t5 的预处理
        inputs = tokenizer(contents, max_length=384, truncation=True)
        labels = tokenizer(
            text_target=[e["title"] for e in examples["data"]], max_length=64, truncation=True
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = datasets.map(preprocess_function, batched=True)
    return tokenized


def train(model):
    rouge = Rouge()

    def eval_metrics(pred):
        predictions, labels = pred
        decode_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)  # -100 的位置变为 0
        decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decode_preds = [
            " ".join(x) for x in decode_preds
        ]  # 基于字拆空格（rouge 本来是以英文单词为单位的）
        decode_labels = [" ".join(x) for x in decode_labels]
        scores = rouge.get_scores(decode_preds, decode_labels, avg=True)
        return {
            'rouge-1': scores['rouge-1']['f'],  # type: ignore
            'rouge-2': scores['rouge-2']['f'],  # type: ignore
            'rouge-L': scores['rouge-l']['f'],  # type: ignore
        }

    args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=64,
        logging_steps=8,
        evaluation_strategy='steps',
        eval_steps=200,
        save_strategy='steps',
        save_steps=200,
        metric_for_best_model='rouge-1',
        predict_with_generate=True,  # seq2seq 的特点
        learning_rate=2e-5,
        num_train_epochs=1,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['test'],
        tokenizer=tokenizer,
        compute_metrics=eval_metrics,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    )
    trainer.train()


def infer(model):
    from transformers import pipeline

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
    pipe("摘要生成:\n" + ds['test'][-1]['data']['content'], max_len=64, do_sample=True)


if __name__ == '__main__':
    datasets_name = 'supremezxc/nlpcc_2017'
    ds = get_datasets(datasets_name)

    ds = ds['train'].select(range(10000)).select_columns(['data'])  # type: ignore
    datasets = ds.train_test_split(test_size=0.1, seed=42)  # type: ignore

    model_id = 'Langboat/mengzi-t5-base'
    tokenizer = load_tokenizer(model_id, tokenizer_cls=T5Tokenizer)

    tokenized_ds = preprocess_datasets(tokenizer, datasets)

    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-text-summarization')
    checkpoints_dir = CKPT_ROOT.joinpath('text_summarization/t5')
    ckpt_version: str | None = None
    ckpt_version = 'checkpoint-843'
    ckpt_dir = checkpoints_dir / ckpt_version if ckpt_version else None
    model = load_model(
        model_id,
        ckpt_dir=ckpt_dir,
        model_cls=T5ForConditionalGeneration,
        pretrained_dir=pretrained_dir,
    )

    train(model)
