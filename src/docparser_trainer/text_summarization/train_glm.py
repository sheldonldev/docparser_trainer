from pathlib import Path

from transformers import (  # type: ignore
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


def preprocess_datasets(tokenizer, datasets):
    def preprocess_function(examples):
        contents = ["摘要生成:\n" + e["content"] + tokenizer.mask_token for e in examples['data']]
        inputs = tokenizer(
            contents, max_length=384, truncation=True, padding=True, return_tensors="pt"
        )
        inputs = tokenizer.build_inputs_for_generation(  # type: ignore
            inputs,
            targets=[e["title"] for e in examples["data"]],
            padding=True,
            max_gen_length=64,
        )
        return inputs

    tokenized = datasets.map(preprocess_function, batched=True, remove_columns=["data"])
    # tokenizer.decode(tokenized['train'][0]['input_ids'])
    # tokenized['train'][0]['position_ids']
    return tokenized


def train(model, tokenizer, datasets, checkpoints_dir):

    # 用 glm 的官方评估函数 训练完了评估一次

    tokenized_datasets = preprocess_datasets(tokenizer, datasets)
    args = Seq2SeqTrainingArguments(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=64,
        logging_steps=8,
        save_strategy='epoch',
        num_train_epochs=1,
        metric_for_best_model='rouge-1',
        predict_with_generate=True,  # seq2seq 的特点
        learning_rate=2e-5,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets['train'],
        tokenizer=tokenizer,
    )
    trainer.train()


def main(
    datasets,
    model_id,
    pretrained_dir,
    checkpoints_dir,
    ckpt_version=None,
):

    tokenizer = load_tokenizer(model_id, tokenizer_cls=AutoTokenizer)
    model = load_model(
        model_id,
        ckpt_dir=Path(checkpoints_dir) / ckpt_version if ckpt_version else None,
        model_cls=AutoModelForSeq2SeqLM,
        pretrained_dir=pretrained_dir,
        num_labels=1,
    )
    train(model, tokenizer, datasets, checkpoints_dir)


if __name__ == '__main__':
    datasets_id = 'supremezxc/nlpcc_2017'
    ds = get_datasets(datasets_id)
    ds = ds['train'].select(range(10000)).select_columns(['data'])  # type: ignore
    datasets = ds.train_test_split(test_size=0.1, seed=42)  # type: ignore

    model_id = 'THUDM/glm-large-chinese'
    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-text-summarization')
    checkpoints_dir = CKPT_ROOT.joinpath('text_summarization/glm')

    main(
        datasets,
        model_id,
        pretrained_dir,
        checkpoints_dir,
    )
