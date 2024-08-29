from pathlib import Path

import torch
from rouge_chinese import Rouge  # type: ignore
from tqdm import tqdm
from transformers import (  # type: ignore
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import (
    ensure_local_model,
    load_local_model,
    load_tokenizer,
)

setup_env()
CHECKPOINTS_DIR = CKPT_ROOT.joinpath('text_summarization/glm')


def get_model(ckpt_dir: Path | None = None):
    model_cls = AutoModelForSeq2SeqLM
    if ckpt_dir is None:
        model_dir = ensure_local_model(
            model_id,
            model_cls=model_cls,
            local_directory=MODEL_ROOT.joinpath(f'{model_id}-text-summarization'),
        )
    else:
        model_dir = ckpt_dir
    model = load_local_model(
        model_dir,
        model_cls=model_cls,
    )
    return model


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


def train(model):

    # 用 glm 的官方评估函数 训练完了评估一次

    args = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
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
        train_dataset=tokenized_ds['train'],
        tokenizer=tokenizer,
    )
    trainer.train()


def predict(model, texts):
    predicts = []
    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(
                "摘要生成:\n" + text + tokenizer.mask_token,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
            inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=64)
            inputs = inputs.to(model.device)
            output = model.generate(
                **inputs, max_new_tokens=64, eos_token_id=tokenizer.eop_token_id, do_sample=True
            )

            predicts.append(
                tokenizer.decode(output[0].tolist())
                .split("<|startofpiece|>")[1]
                .replace("<|endofpiece|>", "")
                .strip()
            )
    return predicts


def evaluate(model):

    rouge = Rouge()
    model.cuda()
    model.eval()

    inputs = [x['content'] for x in datasets['test']['data']]
    predicts = predict(model, inputs)
    decoded_preds = [' '.join(x) for x in predicts]
    decoded_labels = [' '.join(x['title']) for x in datasets['test']['data']]
    scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)

    return {
        'rouge-1': scores['rouge-1']['f'],  # type: ignore
        'rouge-2': scores['rouge-2']['f'],  # type: ignore
        'rouge-L': scores['rouge-l']['f'],  # type: ignore
    }


if __name__ == '__main__':
    datasets_name = 'supremezxc/nlpcc_2017'
    ds = get_datasets(datasets_name)

    ds = ds['train'].select(range(10000)).select_columns(['data'])  # type: ignore
    datasets = ds.train_test_split(test_size=0.1, seed=42)  # type: ignore

    model_id = 'THUDM/glm-large-chinese'
    tokenizer = load_tokenizer(model_id, tokenizer_cls=AutoTokenizer)

    tokenized_ds = preprocess_datasets(tokenizer, datasets)

    ckpt_dir: Path | None = None
    ckpt_dir = CHECKPOINTS_DIR.joinpath('checkpoint-35')
    model = get_model(ckpt_dir)

    # train(model)
    print(evaluate(model))
