from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_model, load_tokenizer
from docparser_trainer._optimization.peft import merge_lora
from docparser_trainer._utils.model_util import print_params
from docparser_trainer._utils.seed import seed_everything

setup_env()
seed_everything(42)


def infer(text, model):
    input = tokenizer(f"Human: {text}\n\nAssistant:", return_tensors="pt").to(model.device)
    return tokenizer.decode(
        model.generate(**input, max_length=256, do_sample=False)[0],
        skip_special_tokens=True,
    )


def load_lora_model_using_peft():
    model = load_model(
        model_id,
        model_cls=AutoModelForCausalLM,
        pretrained_dir=MODEL_ROOT.joinpath(f'{model_id}-generative-chatbot'),
    )  # pretrained

    # 模型大小和占用显存
    model_size = sum(param.numel() for param in model.parameters())
    print(f"model size: {model_size:,} params".capitalize())
    param_gpu_usage = model_size * 4
    print(f"parameters gpu usage: {param_gpu_usage:,} bytes".capitalize())

    # 加入 lora
    model = PeftModel.from_pretrained(model, model_id=str(ckpt_dir))
    model.eval().cuda()

    print(">>> infer with model loaded by peft:")
    print_params(model)
    print(infer(query, model))


def load_lora_model_using_transformers():
    model = load_model(
        model_id,
        model_cls=AutoModelForCausalLM,
        ckpt_dir=ckpt_dir,
        pretrained_dir=MODEL_ROOT.joinpath(f'{model_id}-generative-chatbot'),
    )  # checkpoint
    model.eval().cuda()

    # 虽然从 checkpoints 只保存 adapter_model，但是会根据 adapter_config 里记录的 name_or_path
    # 合并回原来的模型
    # lora 微调仅保存 lora 权重，节约空间
    print(f"ckpt_dir: {ckpt_dir}")
    print(f"name_or_path: {model.name_or_path}")

    print(">>> infer with model loaded by transformers:")
    print_params(model)
    print(infer(query, model))


def save_merged_lora_model():
    model = load_model(
        model_id,
        model_cls=AutoModelForCausalLM,
        pretrained_dir=MODEL_ROOT.joinpath(f'{model_id}-generative-chatbot'),
    )  # pretrained
    model = merge_lora(model, ckpt_dir)
    model.save_pretrained(str(checkpoints_dir / 'merged'))

    model.eval().cuda()

    print(">>> infer with merged model:")
    print_params(model)
    print(infer(query, model))


if __name__ == '__main__':

    model_id = 'Langboat/bloom-1b4-zh'
    tokenizer = load_tokenizer(model_id)

    checkpoints_dir = CKPT_ROOT / f"generative_chatbot/{model_id}-lora"
    ckpt_version: str | None = None
    ckpt_version = 'checkpoint-1525'
    ckpt_dir = checkpoints_dir / ckpt_version if ckpt_version else None

    query = '考试有哪些技巧'

    load_lora_model_using_peft()
    load_lora_model_using_transformers()
    save_merged_lora_model()
