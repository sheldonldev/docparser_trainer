import importlib.metadata
import os

APP_NAME = "docparser_trainer"
__info__ = importlib.metadata.metadata(APP_NAME)

VERSION = __info__.get("version")
AUTHOR_EMAIL = __info__.get("author_email")


def setup_env():
    os.environ['TRANSFORMERS_CACHE'] = os.environ.get(
        'TRANSFORMERS_CACHE', '/mnt/ssd1/models'
    )
    os.environ['HF_HOME'] = os.environ.get('HF_HOME', '/mnt/ssd1/models')
