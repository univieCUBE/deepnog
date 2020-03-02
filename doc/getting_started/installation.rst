============
Installation
============

Installation from PyPI
======================

The current release of ``deepnog`` can be installed from PyPI:

.. code-block:: bash

   pip install deepnog

For typical use cases, and quick start, this is sufficient.

Dependencies and model files
============================

All package dependencies of ``deepnog`` are automatically installed
by ``pip``. We also require model files (networks parameters/weights),
which are too large for GitHub/PyPI. These are hosted on separate servers,
and downloaded automatically by ``deepnog``, when required. By default,
models are cached in `$HOME/deepnog_data/`.

You can change this path by setting the DEEPNOG_DATA environment variable.

.. code-block:: bash

   DEEPNOG_DATA="/custom/path/models" deepnog sequences.fa


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
