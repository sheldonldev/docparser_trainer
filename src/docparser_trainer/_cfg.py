import importlib.metadata
import os
from pathlib import Path

APP_NAME = "docparser_trainer"
__info__ = importlib.metadata.metadata(APP_NAME)

VERSION = __info__.get("version")
AUTHOR_EMAIL = __info__.get("author_email")

PRETRAINED_ROOT = Path(os.environ.get('PRETRAINED_ROOT', '/mnt/ssd1/pretrained'))
MODEL_ROOT = PRETRAINED_ROOT / 'models'

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = Path(os.environ.get('DATA_ROOT', PROJECT_ROOT / 'data'))


def setup_env():
    os.environ['PRETRAINED_ROOT'] = str(PRETRAINED_ROOT)
