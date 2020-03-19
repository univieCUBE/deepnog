#!/usr/bin/env bash

set -e

# This is a work-around for a bug with Ubuntu Linux and PyTorch 1.4.
# Fetch the nightly 1.5 builds, until the offcial release.


if [[ "$TRAVIS_OS_NAME" == 'linux' ]]; then
    MINICONDA_DIR="$HOME/miniconda"
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    export PATH="${MINICONDA_DIR}/bin:$PATH"
    conda activate test
    hash -r

    conda install pytorch cpuonly -c pytorch-nightly
fi
