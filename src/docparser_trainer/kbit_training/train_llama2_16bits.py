from functools import lru_cache

from transformers import AutoModelForCausalLM  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


@lru_cache
def preprocess_datasets(tokenizer, datasets):
    def preprocess_function(examples):
        pass

    tokenized = datasets.map(preprocess_function, batched=True)
    return tokenized


def train(model):
    tokenized_datasets = preprocess_datasets(tokenizer, datasets)


def main(datasets_id, model_id=None, pretrained_dir=None):
    datasets = get_datasets(datasets_id)
    tokenizer = load_tokenizer(model_id)

    model = load_model(
        model_id,
        model_cls=AutoModelForCausalLM,  # 官方用的CausalLM
        pretrained_dir=pretrained_dir,
    )


if __name__ == '__main__':
    # https://huggingface.co/datasets/shibing624/alpaca-zh
    datasets_id = 'shibing624/alpaca-zh'

    # https://huggingface.co/FlagAlpha/Llama3-Chinese-8B-Instruct
    model_id = 'FlagAlpha/Llama3-Chinese-8B-Instruct'

    checkpoints_dir = CKPT_ROOT / f"llm/{model_id}-lora"
    ckpt_version: str | None = None
    ckpt_dir = checkpoints_dir / ckpt_version if ckpt_version else None

    main(datasets_id, model_id)
