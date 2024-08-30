from transformers import AutoModelForTokenClassification  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


def infer(model, tokenizer, label_list, input):
    from collections import defaultdict

    from transformers import pipeline

    model.config.id2label = {i: label for i, label in enumerate(label_list)}  # type: ignore

    ner_pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
    )
    res = ner_pipe(input, aggregation_strategy="simple")

    ner_res = defaultdict(list)
    for r in res:  # type: ignore
        ner_res[r["entity_group"]].append(input[r["start"] : r["end"]])  # type: ignore
    print(ner_res)


def main(model_id, ckpt_dir, label_list, input):
    tokenizer = load_tokenizer(model_id)
    model = load_model(
        ckpt_dir=ckpt_dir,
        model_cls=AutoModelForTokenClassification,
        num_lables=len(label_list),
    )

    infer(model, tokenizer, label_list, input)


if __name__ == '__main__':
    datasets_id = 'peoples-daily-ner/peoples_daily_ner'  # 人民日报
    datasets = get_datasets(datasets_id)
    label_list = datasets["train"].features["ner_tags"].feature.names  # type: ignore

    model_id = 'hfl/chinese-macbert-base'
    ckpt_dir = CKPT_ROOT.joinpath(
        "named_entity_recognition",
    ).joinpath('checkpoint-981')

    main(model_id, ckpt_dir, label_list, input)
