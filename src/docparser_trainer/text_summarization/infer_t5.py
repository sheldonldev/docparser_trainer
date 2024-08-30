from transformers import T5ForConditionalGeneration, T5Tokenizer  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


def infer(model, tokenizer, content):
    from transformers import pipeline

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
    pipe(f"摘要生成:\n{content}", max_len=64, do_sample=True)


def main(model_id, ckpt_dir, contents):
    tokenizer = load_tokenizer(model_id, tokenizer_cls=T5Tokenizer)
    model = load_model(
        ckpt_dir=ckpt_dir,
        model_cls=T5ForConditionalGeneration,
    )
    results = []
    for content in contents:
        results.append(infer(model, tokenizer, content))
    return results


if __name__ == '__main__':
    model_id = 'Langboat/mengzi-t5-base'

    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-text-summarization')
    checkpoints_dir = CKPT_ROOT.joinpath('text_summarization/t5')
    ckpt_version: str | None = None
    ckpt_version = 'checkpoint-843'
    ckpt_dir = checkpoints_dir / ckpt_version if ckpt_version else None

    datasets_name = 'supremezxc/nlpcc_2017'
    ds = get_datasets(datasets_name)
    ds = ds['train'].select(range(10000)).select_columns(['data'])  # type: ignore
    datasets = ds.train_test_split(test_size=0.1, seed=42)  # type: ignore
    contents = [x['content'] for x in datasets['test'][-10:]]
    main(model_id, ckpt_dir, contents)
