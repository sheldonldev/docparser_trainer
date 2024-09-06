import importlib.metadata
import os
from pathlib import Path

APP_NAME = "docparser_trainer"
__info__ = importlib.metadata.metadata(APP_NAME)

VERSION = __info__.get("version")
AUTHOR_EMAIL = __info__.get("author_email")

DATA_ROOT = Path(os.environ.get('DATA_ROOT', '/mnt/ssd0/datasets'))
PRETRAINED_ROOT = Path(os.environ.get('PRETRAINED_ROOT', '/mnt/ssd1/pretrained'))
MODEL_ROOT = PRETRAINED_ROOT / 'models'


PROJECT_ROOT = Path(__file__).parent.parent.parent
CKPT_ROOT = PROJECT_ROOT / 'checkpoints'


HTTP_PROXY = os.environ.get('HTTP_PROXY', 'http://127.0.0.1:17890')
HTTPS_PROXY = os.environ.get('HTTPS_PROXY', 'http://127.0.0.1:17890')


def setup_env():
    os.environ['PRETRAINED_ROOT'] = str(PRETRAINED_ROOT)
