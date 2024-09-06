from pathlib import Path

from datasets import load_dataset  # type: ignore
from util_common.decorator import proxy

from docparser_trainer._cfg import DATA_ROOT, HTTP_PROXY, HTTPS_PROXY


@proxy(http_proxy=HTTP_PROXY, https_proxy=HTTPS_PROXY)
def get_datasets(datasets_id_or_extension, sub_name=None, data_files=None, split=None):
    cache_dir = None
    if Path(datasets_id_or_extension).is_file():
        # 本地文件
        cache_dir = datasets_id_or_extension.parent
    else:
        if sub_name is None:
            cache_dir = str(DATA_ROOT.joinpath(datasets_id_or_extension))
        else:
            cache_dir = str(DATA_ROOT.joinpath(datasets_id_or_extension) / sub_name)

    print(">>> Cache datasets:", cache_dir)
    return load_dataset(
        datasets_id_or_extension,
        name=sub_name,
        data_files=data_files,
        cache_dir=cache_dir,
        split=split,
        trust_remote_code=True,
    )
