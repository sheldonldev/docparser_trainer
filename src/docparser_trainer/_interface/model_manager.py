import os
from functools import lru_cache
from pathlib import Path
from typing import Type

import evaluate  # type: ignore
from transformers import (  # type: ignore
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from util_common.decorator import proxy


def get_pretrained_folder() -> Path | None:
    if os.environ.get('PRETRAINED_ROOT'):
        pretrained_path = Path(os.environ['PRETRAINED_ROOT'])
        if pretrained_path.is_dir():
            return pretrained_path
    return None


@lru_cache(maxsize=None)
def get_pretrained_model_folder(model_name) -> Path:
    pretrained_folder = get_pretrained_folder()
    if pretrained_folder is None:
        raise ValueError('PRETRAINED_ROOT is not set')

    return pretrained_folder.joinpath(f'models/{model_name}')


def load_local_model(
    model_dir,
    model_cls: Type = AutoModel,
    **kwargs,
) -> PreTrainedModel:
    model = model_cls.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **kwargs,
    )
    return model


def load_base_model(
    model_id,
    model_cls: Type = AutoModel,
    local_directory: Path | None = None,
    **kwargs,
) -> PreTrainedModel:
    if local_directory is None:
        local_directory = get_pretrained_model_folder(model_id)
    try:
        model = load_local_model(local_directory, model_cls=model_cls, **kwargs)
    except Exception:
        model = load_remote_model(model_id, model_cls=model_cls, **kwargs)
        model.save_pretrained(str(local_directory))  # type: ignore
    return model


def load_model(
    model_id: str | None = None,
    pretrained_dir: Path | None = None,
    ckpt_dir: Path | None = None,
    model_cls: Type = AutoModel,
    **kwargs,
) -> PreTrainedModel:
    """
    model_id: hugginginge model id, if pretrained_dir not exist,
        will download from huggingface and save to pretrained_dir (
        if pretrained_dir is None, will save to $PRETRAINED_DIR/model_id)
    model_cls: model structure class
    ckpt_dir: if model is finetuned and saved, pass ckpt_dir to load from ckpt_dir
    pretrained_dir: if model is not finetuned, load from pretrained_dir or model_id

    ckpt_dir, pretrained_dir, model_id, at least one is not None.
    """
    if ckpt_dir is None:
        if pretrained_dir is None and model_id is None:
            raise ValueError("model_id or pretrained_dir must be provided")
        model = load_base_model(
            model_id,
            model_cls=model_cls,
            local_directory=pretrained_dir,
            **kwargs,
        )
    else:
        model = load_local_model(
            ckpt_dir,
            model_cls=model_cls,
            **kwargs,
        )
    return model


@proxy(http_proxy=os.environ.get('HTTP_PROXY', ''), https_proxy=os.environ.get('HTTPS_PROXY', ''))
def load_remote_model(
    model_id,
    model_cls: Type = AutoModel,
    **kwargs,
) -> PreTrainedModel:
    model = model_cls.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **kwargs,
    )
    return model


def load_local_tokenizer(
    model_dir, tokenizer_cls: Type = AutoTokenizer, **kwargs
) -> PreTrainedTokenizer:
    tokenizer = tokenizer_cls.from_pretrained(str(model_dir), trust_remote_code=True, **kwargs)
    return tokenizer


@proxy(http_proxy=os.environ.get('HTTP_PROXY', ''), https_proxy=os.environ.get('HTTPS_PROXY', ''))
def load_remote_tokenizer(
    model_id, tokenizer_cls: Type = AutoTokenizer, **kwargs
) -> PreTrainedTokenizer:
    tokenizer = tokenizer_cls.from_pretrained(model_id, trust_remote_code=True, **kwargs)
    return tokenizer


def load_tokenizer(model_id, tokenizer_cls: Type = AutoTokenizer, **kwargs) -> PreTrainedTokenizer:
    local_directory = get_pretrained_model_folder(model_id)
    try:
        tokenizer = load_local_tokenizer(local_directory, tokenizer_cls=tokenizer_cls, **kwargs)
    except Exception:
        tokenizer = load_remote_tokenizer(model_id, tokenizer_cls=tokenizer_cls, **kwargs)
        tokenizer.save_pretrained(str(local_directory))  # type: ignore
    return tokenizer


@proxy(http_proxy=os.environ.get('HTTP_PROXY', ''), https_proxy=os.environ.get('HTTPS_PROXY', ''))
def load_evaluator(name):
    return evaluate.load(name)
