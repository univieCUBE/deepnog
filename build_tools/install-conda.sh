#!/usr/bin/env bash

set -e

# the miniconda directory may exist if it has been restored from cache
if [[ -d "$MINICONDA_DIR" ]] && [[ -e "$MINICONDA_DIR/bin/conda" ]]; then
    echo "Miniconda install already present from cache: $MINICONDA_DIR"
else # if it does not exist, we need to install miniconda
    rm -rf "$MINICONDA_DIR" # remove the directory in case we have an empty cached directory

    if [[ "$TRAVIS_OS_NAME" == 'linux' ]]; then
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    elif [[ "$TRAVIS_OS_NAME" == 'osx' ]]; then
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
    fi

    bash miniconda.sh -b -p "$MINICONDA_DIR"
    chown -R "$USER" "$MINICONDA_DIR"
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda info -a # for debugging
    echo "$TRAVIS_PYTHON_VERSION"
fi

# Is the specified Python version available? If not, install it.
if [[ ! -e "${MINICONDADIR}/envs/test/bin/python${TRAVIS_PYTHON_VERSION}" ]]; then
    "$MINICONDA_DIR/bin/conda" create --yes -n test python="$TRAVIS_PYTHON_VERSION"
fi
