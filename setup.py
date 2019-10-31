#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

import codecs
from os import path
import re
import setuptools


here = path.abspath(path.dirname(__file__))

with open('README.md', 'r') as fh:
    long_description = fh.read()


# Single-sourcing the package version: Read from __init__
def read(*parts):
    with codecs.open(path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


install_requires = ['numpy',
                    'pandas',
                    'torch >= 1.2',
                    'Biopython',
                    'tqdm',
                    'pytest',
                    ]

setuptools.setup(
    name='deepnog',
    version=find_version('deepnog', '__init__.py'),
    author='Lukas Gosch',
    author_email='gosch.lukas@gmail.com',
    description='Deep learning based command line tool for protein family '
                + 'predictions.',
    keywords='deep learning bioinformatics neural networks protein families',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    url='',
    packages=setuptools.find_packages(),
    package_data={
        'deepnog': ['parameters/*/*.pth',    # Include parameters of NNs trained directly on
                                             # a whole database (currently not supported by DeepNOG).
                    'parameters/*/*/*.pth',  # Include parameters of NNs trained on specific levels/parts of a db
                    ],
        'tests': ['data/*.faa',  # Include data and parameters for tests, edit if necessary!
                  'parameters/*.pth',
                  ]
    },
    entry_points={
        'console_scripts': [
            'deepnog = deepnog.deepnog:main'
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3',
)
