#!/usr/bin/env bash
pipenv lock --requirements > requirements.txt
pipenv lock --requirements --dev > requirements.dev.txt
# These packages are preloaded on Colab and would require a runtime restart
grep -ivE "^matplotlib==|^numpy==|^pillow==" requirements.txt > requirements.colab.txt
