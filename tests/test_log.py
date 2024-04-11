import logging
from typing import Optional

from rich.logging import RichHandler

from docparser_trainer._cfg import APP_NAME
from docparser_trainer._log import log


def _init_logger(
    name: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, "INFO"))
    logger.handlers.clear()
    logger.addHandler(
        RichHandler(
            rich_tracebacks=True,
            show_time=True,
            omit_repeated_times=True,
            show_level=True,
            show_path=True,
            enable_link_path=True,
        )
    )
    return logger


def test_log():
    _init_logger(name=APP_NAME)
    log.info("log is available.")
