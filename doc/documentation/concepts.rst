========
Concepts
========

``deepnog`` is a command line tool written in Python 3. It uses deep networks for extremely
fast protein orthology assignments. Currently, it is based on a deep convolutional network
architecture called DeepNOG trained on the root and bacterial level of the eggNOG 5.0 database
(Huerta-Cepas et al. (2019)).

Two subcommand are available:

- ``deepnog infer`` for assigning sequences to orthologous groups, using precomputed models, and
- ``deepnog train`` for training such models (e.g. other taxonomic levels or future versions of eggNOG, different
  orthology databases, etc.)

``deepnog infer`` for orthology assignments
===========================================

Input Data
----------

DeepNOG expects a protein sequence file as input.
It is tested for the FASTA file format, but in general should support all
file formats supported by the Bio.SeqIO module of Biopython.
Compressed files (.gz or .xz) are supported as well.
Protein sequences without IDs in the input data file are skipped
and not used for the following assignment phase.
Furthermore, if two sequences in the input data file have the same
associated ID, only the sequence encountered first in the input data file
will be kept and all others discarded before the output file is created.
The user will be notified if such cases are encountered.

Assignment Phase
----------------

In the assignment phase, ``deepnog`` loads a predefined deep network and the corresponding trained
weights (defaults to DeepNOG trained on eggNOG 5.0, bacterial level). Then it performs the
assignment by forwarding the input sequences through the network performing the calculations
either on a CPU or GPU. ``deepnog`` offers single-process data loading aimed for calculations on a
single CPU core to produce as little overhead as possible. Additionally, it offers parallel multiprocess
data loading aimed for very fast GPU calculations. This is to provide the GPU with data following
up the previous forward pass fast enough such that the GPU does not experience idling. In its
default parametrization, ``deepnog`` is optimized for single core CPU calculations,
or massively parallel GPU calculations.

Output Data
-----------

As an output ``deepnog`` generates a CSV file which consists of three columns:

#. The unique name or ID of the protein extracted from the sequence file,
#. the assigned orthologous group, and
#. the network's confidence in the assignment.

Each deep network model has the possibility to define an assignment confidence threshold
below which, the network's output layer is treated as having predicted that the input protein
sequence is not associated to any orthologous group in the model. Therefore, if the highest assignment
confidence for any OG for a given input protein sequence is below this threshold, the assignment is
left empty. Per default, using DeepNOG on eggNOG 5.0, the prediction confidence threshold is
set to a strict 99%. This threshold can be adjusted by the user.


``deepnog train`` for creating custom models
============================================

For details on training new models, see `New models/architectures <training.html>`_.
