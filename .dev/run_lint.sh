#!/bin/bash
set -euo pipefail
set -x


_script_loc=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$(dirname $_script_loc)"


echo "******************************** linting start ********************************"
poetry run isort . && \
poetry run black . && \
poetry run pylint -j 4 --rcfile=./.pylintrc --reports=y -v ./feature_analysis ./tests && \
poetry run mypy .
echo "******************************** linting end ********************************"
