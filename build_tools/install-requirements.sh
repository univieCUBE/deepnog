#!/usr/bin/env bash

set -e

MINICONDA_DIR="$HOME/miniconda"

# Alternative to "conda init bash"
source "$HOME/miniconda/etc/profile.d/conda.sh"

conda activate test
hash -r

echo "pip installing required python packages"
pip install -r requirements/dev.txt


python --version
