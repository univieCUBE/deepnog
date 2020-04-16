#!/usr/bin/env bash

set -e

MINICONDA_DIR="$HOME/miniconda"
source "$HOME/miniconda/etc/profile.d/conda.sh"
# export PATH="${MINICONDA_DIR}/bin:$PATH"
"$MINICONDA_DIR/bin/conda" activate test
hash -r

echo "which pip"
which pip
echo "which python"
which python
echo "which python3"
which python3
echo "PATH"
echo "$PATH"

# This is a work-around for a bug with Ubuntu Linux and PyTorch 1.4.
# Fetch the nightly 1.5 builds, until the official release.
if [[ "$TRAVIS_OS_NAME" == 'linux' ]]; then
  echo "install and upgrade PyTorch nightly"
  conda install --yes pytorch cpuonly -c pytorch-nightly
  # conda upgrade --yes pytorch cpuonly -c pytorch-nightly
  echo "pip installing required python packages"
  pip install -r requirements/travis_ubuntu.txt
  python -c "import torch; print(f'PyTorch version = {torch.__version__}')"
elif [[ "$TRAVIS_OS_NAME" == 'osx' ]]; then
  echo "pip installing required python packages"
  pip install -r requirements/dev.txt
fi

python --version
