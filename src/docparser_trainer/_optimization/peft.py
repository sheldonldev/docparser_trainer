from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.ia3 import IA3Config
from peft.tuners.lora.config import LoraConfig
from peft.tuners.p_tuning import PromptEncoderConfig, PromptEncoderReparameterizationType
from peft.tuners.prefix_tuning import PrefixTuningConfig
from peft.tuners.prompt_tuning import PromptTuningConfig, PromptTuningInit
from peft.utils.peft_types import TaskType


def bit_fit(model):
    # BitFit: 只更新 bias
    num_param = 0
    for name, param in model.named_parameters():
        if "bias" not in name:
            param.requires_grad = False
        else:
            num_param += param.numel()
    print(f"bit fit: {num_param} bias params")


def soft_prompt_tuning(model):
    config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)
    print(">>> Soft prompt tuning")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def hard_prompt_tuning(model, tokenizer, prompt=None):
    if prompt is None:
        prompt = "下面是一段与机器人的对话"
    config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        tokenizer_name_or_path=tokenizer.name_or_path,
        prompt_tuning_init_text=prompt,
        num_virtual_tokens=len(tokenizer(prompt)),
    )
    print(">>> Hard prompt tuning")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def p_tuning(model):
    config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=10,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
        encoder_num_layers=5,
        encoder_dropout=0.1,
        encoder_hidden_size=1024,
    )
    print(">>> P tuning")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def prefix_tuning(model):
    config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=10,
        prefix_projection=False,  # True 表示加个全连接层重参数化, 效果会好些，也更吃显存
        encoder_hidden_size=1024,
    )
    print(">>> Prefix tuning")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    print(model.prompt_encoder)
    return model


def lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["query_key_value"],  # 查看 model.named_parameters, 支持正则
        modules_to_save=["word_embeddings"],  # 用法同上
        inference_mode=False,
        r=16,
        lora_alpha=32,  # 权重缩放控制 lora_alpha / r
        lora_dropout=0.1,
    )
    _model = get_peft_model(model, config)
    _model.print_trainable_parameters()
    return _model


def merge_lora(model, ckpt_dir):
    peft_model = PeftModel.from_pretrained(model=model, model_id=str(ckpt_dir))
    # return peft_model
    merged_model = peft_model.merge_and_unload()
    return merged_model


def ia3(model):
    config = IA3Config(task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
