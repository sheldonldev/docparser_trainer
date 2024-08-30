from pathlib import Path

from datasets import load_dataset  # type: ignore
from util_common.decorator import proxy

from docparser_trainer._cfg import DATA_ROOT, HTTP_PROXY, HTTPS_PROXY


@proxy(http_proxy=HTTP_PROXY, https_proxy=HTTPS_PROXY)
def get_datasets(datasets_id, sub_name=None, data_files=None, split=None):
    if Path(datasets_id).exists():
        cache_dir = datasets_id
    if sub_name is None:
        cache_dir = DATA_ROOT.joinpath(datasets_id)
    else:
        cache_dir = DATA_ROOT.joinpath(datasets_id) / sub_name
    return load_dataset(
        datasets_id,
        name=sub_name,
        data_files=data_files,
        cache_dir=str(cache_dir),
        split=split,
        trust_remote_code=True,
    )
