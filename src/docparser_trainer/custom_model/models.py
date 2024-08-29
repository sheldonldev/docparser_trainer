from pathlib import Path

import torch
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora import LoraConfig
from torch import nn

from docparser_trainer._utils.model_util import print_params


def get_model():
    return nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
    )


def customized_target_modules():
    net = get_model()
    print(net)
    config = LoraConfig(target_modules=['0'])
    model = get_peft_model(net, config)  # type: ignore
    print(model)


def shift_or_band_adapters():
    save_dir = Path(__file__).parent

    # 第 0 层的 adapter
    config1 = LoraConfig(target_modules=['0'])
    model1 = get_peft_model(get_model(), config1)  # type: ignore
    model1.save_pretrained(str(save_dir / "lora1"))

    # 第 2 层的 adapter
    config2 = LoraConfig(target_modules=['2'])
    model2 = get_peft_model(get_model(), config2)  # type: ignore
    model2.save_pretrained(str(save_dir / "lora2"))

    # 加载第 0 层的 lora
    model = PeftModel.from_pretrained(get_model(), save_dir / "lora1", adapter_name='lora1')

    # 加载第 2 层的 lora
    model.load_adapter(str(save_dir / "lora2"), adapter_name='lora2')
    print(model)
    print_params(model)

    print('原始模型，默认全随机始化 Lora_A, 全零初始化 Lora_B')
    for name, param in model.named_parameters():
        if name in [
            'base_model.model.0.lora_A.lora1.weight',
            'base_model.model.0.lora_B.lora1.weight',
            'base_model.model.2.lora_A.lora2.weight',
            'base_model.model.2.lora_B.lora2.weight',
        ]:
            print(name, param)
    print('===')
    print('当前激活适配器', model.active_adapter)
    print('原始结果')
    print(model(torch.arange(0, 10).view(1, 10).float()))

    print('修改 lora1 权重为 1')
    for name, param in model.named_parameters():
        if name in [
            'base_model.model.0.lora_B.lora1.weight',
        ]:
            param.data = torch.ones_like(param)
    print(model(torch.arange(0, 10).view(1, 10).float()))

    print('修改 lora2 权重为 1')
    for name, param in model.named_parameters():
        if name in [
            'base_model.model.2.lora_B.lora2.weight',
        ]:
            param.data = torch.ones_like(param)
    print(model(torch.arange(0, 10).view(1, 10).float()))

    print("===")
    # 通过切换适配器可以为不同任务分配不同的权重,无需每次都加载主模型
    model.set_adapter('lora2')
    print('切换适配器为', model.active_adapter)

    print('原始结果')
    print(model(torch.arange(0, 10).view(1, 10).float()))

    print('修改 lora1 权重为 0')
    for name, param in model.named_parameters():
        if name in [
            'base_model.model.0.lora_B.lora1.weight',
        ]:
            param.data = torch.zeros_like(param)
    print(model(torch.arange(0, 10).view(1, 10).float()))

    print('修改 lora2 权重为 0')
    for name, param in model.named_parameters():
        if name in [
            'base_model.model.2.lora_B.lora2.weight',
        ]:
            param.data = torch.zeros_like(param)
    print(model(torch.arange(0, 10).view(1, 10).float()))

    print('===')
    # DPO 或者 基于Lora做强化学习的时候，有需要禁用adapters的场景
    print('当前激活适配器', model.active_adapter)
    print('原始结果')
    print(model(torch.arange(0, 10).view(1, 10).float()))

    print('禁用适配器')
    with model.disable_adapter():
        print('修改 lora1 权重为 1')
        for name, param in model.named_parameters():
            if name in [
                'base_model.model.0.lora_B.lora1.weight',
            ]:
                param.data = torch.ones_like(param)
        print(model(torch.arange(0, 10).view(1, 10).float()))

        print('修改 lora2 权重为 1')
        for name, param in model.named_parameters():
            if name in [
                'base_model.model.2.lora_B.lora2.weight',
            ]:
                param.data = torch.ones_like(param)
        print(model(torch.arange(0, 10).view(1, 10).float()))


if __name__ == '__main__':
    # customized_target_modules()
    shift_or_band_adapters()
