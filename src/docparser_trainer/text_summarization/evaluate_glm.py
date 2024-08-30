import torch
from rouge_chinese import Rouge  # type: ignore
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


def predict(model, tokenizer, texts):
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


def evaluate(model, tokenizer, inputs):
    rouge = Rouge()
    model.cuda()
    model.eval()

    inputs = [x['content'] for x in datasets['test']['data']]
    predicts = predict(model, tokenizer, inputs)
    decoded_preds = [' '.join(x) for x in predicts]
    decoded_labels = [' '.join(x['title']) for x in datasets['test']['data']]
    scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)

    return {
        'rouge-1': scores['rouge-1']['f'],  # type: ignore
        'rouge-2': scores['rouge-2']['f'],  # type: ignore
        'rouge-L': scores['rouge-l']['f'],  # type: ignore
    }


def main(
    datasets,
    model_id,
    ckpt_dir,
):

    tokenizer = load_tokenizer(model_id, tokenizer_cls=AutoTokenizer)
    model = load_model(
        ckpt_dir=ckpt_dir,
        model_cls=AutoModelForSeq2SeqLM,
        num_labels=1,
    )
    evaluate(model, tokenizer, datasets)


if __name__ == '__main__':
    datasets_id = 'supremezxc/nlpcc_2017'
    ds = get_datasets(datasets_id)
    ds = ds['train'].select(range(10000)).select_columns(['data'])  # type: ignore
    datasets = ds.train_test_split(test_size=0.1, seed=42)  # type: ignore

    model_id = 'THUDM/glm-large-chinese'
    ckpt_dir = CKPT_ROOT.joinpath(
        'text_summarization/glm',
    ).joinpath('checkpoint-35')

    print(main(datasets, model_id, ckpt_dir))
