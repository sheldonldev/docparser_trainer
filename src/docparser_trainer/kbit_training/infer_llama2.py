from transformers import AutoModelForCausalLM  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


def infer(model):
    pass


def main(model_id=None, model_dir=None):
    model = load_model(
        model_id,
        model_cls=AutoModelForCausalLM,  # 官方用的CausalLM
        pretrained_dir=model_dir,
    )
    tokenizer = load_tokenizer(model_id)


if __name__ == '__main__':
    # https://huggingface.co/FlagAlpha/Llama3-Chinese-8B-Instruct
    model_id = 'FlagAlpha/Llama3-Chinese-8B-Instruct'

    checkpoints_dir = CKPT_ROOT / f"llm/{model_id}-lora"
    ckpt_version: str | None = None
    ckpt_dir = checkpoints_dir / ckpt_version if ckpt_version else None

    main(model_id, model_dir=ckpt_dir)
