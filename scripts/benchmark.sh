#!/bin/bash

pip install .[tesing]
python scripts/benchmark.py
pytest .
