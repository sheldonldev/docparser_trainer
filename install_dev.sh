#!/bin/bash
pip install -e ".[dev]"

pre-commit install
mypy --install-types
