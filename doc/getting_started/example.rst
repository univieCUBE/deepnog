===================
Quick Start Example
===================

The following example shows all these steps for predicting protein orthologous groups
with the command line interface of ``deepnog`` as well as using the Python API.
Please make sure you have installed ``deepnog`` (`installation instructions <installation.html>`_).

CLI Usage Example
=================

Using ``deepnog`` from the command line is the simple, and preferred way of interacting with the
``deepnog`` package.

Here, we predict orthologous groups (OGs) of proteins using a model trained on the eggNOG 5.0
database and using only bacterial OGs (default settings),
and redirect the output from stdout to a file:

.. code-block:: bash

   deepnog input.fa > prediction.csv

Alternatively, the output file and other settings can be specified explicitly like so:

.. code-block:: bash

   deepnog input.fa --out prediction.csv --outformat csv -db eggNOG5 --tax 2

For a detailed explanation of flags and further settings,
please consult the `User Guide <../documentation/user_guide.html>`_.

Note that deepnog masks predictions below a certain confidence threshold.
The default confidence threshold baked into the model at 0.99
can be overridden from the command line interface:

.. code-block:: bash

   deepnog input.fa --confidence-threshold 0.8 > prediction.csv


The output comma-separated values (CSV) file `prediction.csv` then looks something like:

::

   sequence_id,prediction,confidence
   WP_004995615.1,COG5449,0.99999964
   WP_004995619.1,COG0340,1.0
   WP_004995637.1,COG4285,1.0
   WP_004995655.1,COG4118,1.0
   WP_004995678.1,COG0184,1.0
   WP_004995684.1,COG1137,1.0
   WP_004995690.1,COG0208,1.0
   WP_004995697.1,,
   WP_004995703.1,COG0190,1.0


The file contains a single line for each protein in the input sequence file,
and the following fields:

* ``sequence_id``, the name of the input protein sequence.
* ``prediction``, the name of the predicted protein OG. Empty if masked by confidence threshold.
* ``confidence``, the confidence value (0-1 inclusive) that ``deepnog`` ascribes to this prediction.
  Empty if masked by confidence threshold.

API Example Usage
=================

.. code-block:: python

   import torch
   from deepnog.dataset import ProteinDataset
   from deepnog.inference import load_nn, predict
   from deepnog.io import create_df, get_weights_path
   from deepnog.utils import set_device


   PROTEIN_FILE = '/path/to/file.faa'
   DATABASE = 'eggNOG5'
   TAX = 2
   ARCH = 'deepencoding'
   CONF_THRESH = 0.8

   # load protein sequence file into a ProteinDataset
   dataset = ProteinDataset(PROTEIN_FILE, f_format='fasta')

   # Construct path to saved parameters deepnog model.
   weights_path = get_weights_path(
       database=DATABASE,
       level=str(TAX),
       architecture=ARCH,
   )

   # Set up device for prediction
   device = set_device('auto')
   torch.set_num_threads(1)

   # Load neural network parameters
   model_dict = torch.load(weights_path, map_location=device)

   # Load neural network model and class names
   model = load_nn(ARCH, model_dict, device)
   class_labels = model_dict['classes']

   # perform prediction
   preds, confs, ids, indices = predict(
       model=model,
       dataset=dataset,
       device=device,
       batch_size=1,
       num_workers=1,
       verbose=3
   )

   # Construct results (a pandas DataFrame)
   df = create_df(
       class_labels=class_labels,
       preds=preds,
       confs=confs,
       ids=ids,
       indices=indices,
       threshold=threshold,
       verbose=3
   )
