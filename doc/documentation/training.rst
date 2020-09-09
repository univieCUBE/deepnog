====================================
Deepnog New Models and Architectures
====================================

``deepnog`` is developed with extensibility in mind,
and allows to plug in additional models (for different taxonomic levels,
or different orthology databases).
It also supports addition of new network architectures.

In order to register a new network architecture,
we recommend an editable installation with pip,
as described in :ref:`Installation from Source <install-from-source>`.


Training scripts
----------------

Starting with v1.2.0, ``deepnog`` ships with functions for training custom models.
Consider we are training a DeepNOG model for eggNOG 5, level 1239 (Firmicutes):

.. code-block:: bash

    deepnog train \
        -a "deepnog" \
        -o /path/to/output/ \
        -db "eggNOG5" \
        -t "1239" \
        --shuffle \
        train.faa.gz \
        val.faa.gz \
        train.csv.gz \
        val.csv.gz

Run ``deepnog train --help`` for additional options.

In order to assess the new model's quality, run the following commands:

.. code-block:: bash

    deepnog infer \
        -a "deepnog" \
        -w /path/to/output/MODEL_FILENAME.pth \
        -o /path/to/output/assignments.csv \
        --test_labels test.csv.gz \
        test.faa.gz
    cat /path/to/output/assignments.performance.csv

This provides a number of performance measures, including accurcay,
macro averaged precision and recall, among others.


Register new models
-------------------

New models for additional taxonomic levels in eggNOG 5 or even different orthology databases
using existing network architectures must be placed in the deepnog data directory
as specified by the DEEPNOG_DATA environment variable (default: ``$HOME/deepnog_data``).

The directory looks like this:

.. code-block::

    | deepnog_data
    | ├── eggNOG5
    | │   ├── 1
    | │   |   └── deepnog.pth
    | │   └── 2
    | │       └── deepnog.pth
    | ├── ...
    |
    |

In order to add a root level model for "MyOrthologyDB",
we place the serialized PyTorch parameters like this:

.. code-block::
    :emphasize-lines: 7-9

    | deepnog_data
    | ├── eggNOG5
    | │   ├── 1
    | │   |   └── deepnog.pth
    | │   └── 2
    | │       └── deepnog.pth
    | ├── MyOrthologyDB
    | |   └── 1
    | |       └── deepnog.pth
    | ├── ...
    |


Register new network architectures
----------------------------------

Create a Python module ``deepnog/models/<my_network.py>``.
You can use ``deepnog.py`` as a template. A new architecture ``MyNetworkA``
would look like so:

.. code-block:: Python

    import torch.nn as nn

    
    class MyNetworkA(nn.Module):
        """ A revolutionary network for orthology prediction. """
        def __init__(self, model_dict):
            super().__init__()
            param1 = model_dict['param1']
            param2 = model_dict['param2']
            param3 = model_dict.get('param3', 0.)
            ...
        def forward(self, x):
            ...
            return x

When the new module is in place, also edit ``deepnog/config/deepnog_config.py``
to expose the new network to the user:

.. code-block:: Python
    :emphasize-lines: 2-11

    architecture:
      netA:
        module: my_network
        class: MyNetworkA
        param1: 'settingXYZ'
        param2:
          - 2
          - 4
          - 8
        param3: 150
        # ... all hyperparameters required for class init

      deepnog:
        module: deepnog
        class: DeepNOG
        encoding_dim: 10
        kernel_size:
          - 8
          - 12
          - 16
          - 20
          - 24
          - 28
          - 32
          - 36
        n_filters: 150
        dropout: 0.3
        pooling_layer_type: 'max'

The new network can now be used in ``deepnog`` by specifying parameter ``-a netA``.


Assuming we want to compare ``deepnog`` to ``netA``,
we add the trained network parameters like this:

.. code-block::
    :emphasize-lines: 5,8,12

    | deepnog_data
    | ├── eggNOG5
    | │   ├── 1
    | │   |   ├── deepnog.pth
    | │   |   └── netA.pth
    | │   └── 2
    | │       ├── deepnog.pth
    | │       └── netA.pth
    | ├── MyOrthologyDB
    | |   └── 1
    | │       ├── deepnog.pth
    | │       └── netA.pth
    | ├── ...
    |

Finally, expose the new models to the user by modifying ``deepnog/config/deepnog_config.py``
again. The relevant section is ``database``.

.. code-block:: python
    :emphasize-lines: 7-9

    database:
      eggNOG5:
        # taxonomic levels
        - 1
        - 2
        - 1236
        - 1239        # Example 1: Uncomment this line, if you created a Firmicutes model
      MyOrthologyDB:  # Example 2: Uncomment this line and the following, if you
        - 1           #            created a model for the '1' level of MyOrthologyDB.

Notes:

* Currently, a level must be provided, even if the database does not use levels.
  Simply use a placeholder 1 or similar.
* Indentation matters
