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


Register new network architectures
----------------------------------

Create a Python module under ``deepnog/models/<my_network.py>``.
You can use ``deepnog.py`` as a template.
When the new module is in place, also edit ``deepnog/client.py``
to expose the new network to the user:

.. code-block:: Python
    :emphasize-lines: 4

    parser.add_argument("-a", "--architecture",
                        default='deepencoding',
                        choices=['deepencoding',
                                 'my_network',
                                 ],
                        help="Network architecture to use for classification.")


Register new models
-------------------

New models for additional taxnomic levels or even different orthology databases
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

Assuming we want to compare ``deepnog`` to ``my_network``,
we add the trained network parameters like this:

.. code-block::

    | deepnog_data
    | ├── eggNOG5
    | │   ├── 1
    | │   |   ├── deepnog.pth
    | │   |   └── my_network.pth
    | │   └── 2
    | │       ├── deepnog.pth
    | │       └── my_network.pth
    | ├── MyOrthologyDB
    | |   └── 1
    | │       ├── deepnog.pth
    | │       └── my_network.pth
    | ├── ...
    |

Finally, expose the new models to the user by modifying ``deepnog/client.py``
again. The relevant section is argument parsing for ``--database``,
and ``--tax``, if new taxonomic levels are introduced as well.

.. code-block:: python
    :emphasize-lines: 4

    parser.add_argument("-db", "--database",
                        type=str,
                        choices=['eggNOG5',
                                 'MyOrthologyDB',
                                 ],
                        default='eggNOG5',
                        help="Orthologous group/family database to use.")
    parser.add_argument("-t", "--tax",
                        type=int,
                        choices=[1, 2, ],
                        default=2,
                        help="Taxonomic level to use in specified database "
                             "(1 = root, 2 = bacteria)")


Training scripts
----------------

Please note, that no training scripts are currently shipped with
``deepnog``, as scripts used for the available models rely on in-house
software libraries and databases, such as SIMAP2.
We are currently working on standalone training scripts,
that will be made public asap.
