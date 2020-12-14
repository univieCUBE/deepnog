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
                    'scikit-learn',
                    'torch >= 1.2',
                    'Biopython',
                    'PyYAML',
                    'tqdm',
                    'tensorboard',
                    ]

setuptools.setup(
    name='deepnog',
    version=find_version('deepnog', '__init__.py'),
    author='Roman Feldbauer',
    author_email='roman.feldbauer@univie.ac.at',
    description='Deep learning tool for protein orthologous group assignment',
    keywords=('deep-learning neural-networks '
              'bioinformatics computational-biology '
              'protein-families orthologous-groups orthology eggnog'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    url='',
    packages=setuptools.find_packages(),
    package_data={
        'deepnog': ['utils/certifi',   # Work-around: include certificate chain
                                       # for fileshare.csb.univie.ac.at
                    'config/*.yml',    # DeepNOG configuration
                    ],
        'tests': ['data/*.faa',        # Include data and parameters for tests,
                  'parameters/*.pth',  # edit if necessary!
                  ]
    },
    entry_points={
        'console_scripts': [
            'deepnog = deepnog.client:main'
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3',
)
