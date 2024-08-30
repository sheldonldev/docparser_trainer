from transformers import AutoModelForSequenceClassification  # type:ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


def infer(model, tokenizer, text, text_pair):
    from transformers import pipeline

    # model.config.id2label = {0: 'not_match', 1: 'match'}
    pipe = pipeline(
        'text-classification',
        model=model,
        tokenizer=tokenizer,
        device=model.device,
    )
    return pipe({"text": text, "text_pair": text_pair}, function_to_apply="none")


def main(model_id, ckpt_dir, texts, text_pairs):
    tokenizer = load_tokenizer(model_id)
    model = load_model(
        ckpt_dir=ckpt_dir,
        model_cls=AutoModelForSequenceClassification,
        num_labels=1,
    )
    results = []
    for text, pair in zip(texts, text_pairs):
        results.append(infer(model, tokenizer, text, pair))
    return results


if __name__ == '__main__':
    model_id = 'hfl/chinese-macbert-base'
    ckpt_dir = CKPT_ROOT.joinpath(
        'text_match/text_similarity_cross',
    ).joinpath('checkpoints-846')

    texts = ['小明喜欢北京']
    text_pairs = ['北京是小明喜欢的城市']
    main(model_id, ckpt_dir, texts, text_pairs)
