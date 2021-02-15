#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail
command -v shellcheck >/dev/null && shellcheck "$0"

poetry export --without-hashes --output requirements.txt
poetry export --dev --without-hashes --output requirements.dev.txt
# These packages are preloaded on Colab and would require a runtime restart
grep -ivE "^matplotlib==|^numpy==|^pillow==" requirements.txt >requirements.colab.txt
