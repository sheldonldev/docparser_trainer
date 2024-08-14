from datasets import load_dataset  # type: ignore
from util_common.decorator import proxy

from docparser_trainer._cfg import DATA_ROOT


@proxy(http_proxy='127.0.0.1:17890', https_proxy='127.0.0.1:17890')
def get_datasets(datasets_name):
    return load_dataset(
        datasets_name,
        cache_dir=str(DATA_ROOT.joinpath(datasets_name)),
        trust_remote_code=True,
    )
