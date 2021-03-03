============
Installation
============

Installation from PyPI
======================

The current release of ``deepnog`` can be installed from PyPI:

.. code-block:: bash

   pip install deepnog

For typical use cases, and quick start, this is sufficient.
Note that this guide assumes Linux, and may work under macOS.
We currently don't provide detailed instructions for Windows.

Alternative: Installation from bioconda
=======================================

Alternatively, ``deepnog`` can be installed from bioconda
with channel setup as described in the `bioconda docs`_ like this:

.. code-block:: bash

    conda install deepnog

Note: Previously, bioconda required installing PyTorch from Facebook's pytorch
channel manually.
Since PyTorch is now available on conda-forge, this is not necessary anymore.

.. _`bioconda docs`: https://bioconda.github.io/user/install.html#install-conda>

Dependencies and model files
============================

All package dependencies of ``deepnog`` are automatically installed
by ``pip`` or ``conda``.
We also require model files (= networks parameters/weights),
which are too large for GitHub/PyPI/bioconda.
Models are hosted on separate servers,
and downloaded automatically by ``deepnog``, when required. By default,
models are cached in `$HOME/deepnog_data/`.

You can change this path by setting the DEEPNOG_DATA environment variable.
Choose among the following options to do so:

.. code-block:: bash

   # Set data path temporarily
   DEEPNOG_DATA="/custom/path/models" deepnog infer sequences.fa

   # Set data path for the current shell
   export DEEPNOG_DATA="/custom/path/models"

   # Set data path permanently
   printf "\n# Set path to DeepNOG models\nexport DEEPNOG_DATA=\"/custom/path/models\"\n" >> ~/.bashrc


.. _install-from-source:

Installation from source
========================

You can always grab the latest version of ``deepnog`` directly from GitHub:

.. code-block:: bash

   cd install_dir
   git clone git@github.com:univieCUBE/deepnog.git
   cd deepnog
   pip install -e .

This is the recommended approach, if you want to contribute
to the development of ``deepnog``.


Supported platforms
===================

``deepnog`` currently supports all major operating systems:

- Linux
- MacOS X
- Windows
