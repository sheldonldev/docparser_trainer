from transformers import AutoModelForCausalLM  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import load_dataset
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


def preprocess_datasets(tokenizer, datasets):
    def preprocess_function(examples):
        pass

    tokenized = datasets.map(preprocess_function, batched=True)
    return tokenized


def train(model):
    tokenized_datasets = preprocess_datasets(tokenizer, datasets)


if __name__ == '__main__':
    # https://huggingface.co/datasets/shibing624/alpaca-zh
    datasets_id = 'shibing624/alpaca-zh'
    datasets = load_dataset(datasets_id)

    # https://huggingface.co/FlagAlpha/Llama3-Chinese-8B-Instruct
    model_id = 'FlagAlpha/Llama3-Chinese-8B-Instruct'
    tokenizer = load_tokenizer(model_id)

    checkpoints_dir = CKPT_ROOT / f"llm/{model_id}-lora"
    ckpt_version: str | None = None
    ckpt_dir = checkpoints_dir / ckpt_version if ckpt_version else None

    model = load_model(
        model_id,
        model_cls=AutoModelForCausalLM,  # 官方用的CausalLM
        ckpt_dir=ckpt_dir,
    )

    train(model)
