=========================
Deepnog CLI Documentation
=========================

Invocation:

::

    deepnog infer SEQUENCE_FILE [options] > assignments.csv

Basic Commands
==============

These options may be commonly tuned for a basic invocation for orthologous group assignment.

::

    positional arguments:
      SEQUENCE_FILE         File containing protein sequences for classification.

    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      -db {eggNOG5, cog2020}, --database {eggNOG5, cog2020}
                            Orthologous group/family database to use. (default:
                            eggNOG5)
      -t {1,2,[]}, --tax {1,2}
                            Taxonomic level to use in specified database
                            (1 = root, 2 = bacteria) (default: 2)
      -o FILE, --out FILE   Store orthologous group assignments to output file.
                            Per default, write predictions to stdout. (default: None)
      -c FLOAT, --confidence-threshold FLOAT
                            The confidence value below which predictions are
                            masked by deepnog. By default, apply the confidence
                            threshold saved in the model if one exists, and else
                            do not apply a confidence threshold. (default: None)

Advanced Commands
=================

These options are unlikely to require manual tuning for the average user.

::

    --verbose INT       Define verbosity of DeepNOGs output written to stdout
                        or stderr. 0 only writes errors to stderr which cause
                        DeepNOG to abort and exit. 1 also writes warnings to
                        stderr if e.g. a protein without an ID was found and
                        skipped. 2 additionally writes general progress
                        messages to stdout.3 includes a dynamic progress bar
                        of the prediction stage using tqdm. (default: 3)
    -ff STR, --fformat STR
                        File format of protein sequences. Must be supported by
                        Biopythons Bio.SeqIO class. (default: fasta)
    -of {csv,tsv,legacy}  --outformat {csv,tsv,legacy}
                        The file format of the output file produced by
                        deepnog. (default: csv)
    -d {auto,cpu,gpu}, --device {auto,cpu,gpu}
                        Define device for calculating protein sequence
                        classification. Auto chooses GPU if available,
                        otherwise CPU. (default: auto)
    -nw INT, --num-workers INT
                        Number of subprocesses (workers) to use for data
                        loading. Set to a value <= 0 to use single-process
                        data loading. Note: Only use multi-process data
                        loading if you are calculating on a gpu (otherwise
                        inefficient)! (default: 0)
    -a {deepnog}, --architecture {deepnog}
                        Network architecture to use for classification.
                        (default: deepnog)
    -w FILE, --weights FILE
                        Custom weights file path (optional) (default: None)
    -bs INT, --batch-size INT
                        The batch size determines how many sequences are
                        processed by the network at once. If 1, process the
                        protein sequences sequentially (recommended
                        on CPUs). Larger batch sizes speed up the inference and
                        training on GPUs. Batch size can influence the
                        learning process.
    --test_labels TEST_LABELS_FILE
                        Measure model performance on a test set.
                        If provided, this file must contain the ground-truth
                        labels for the provided sequences.
                        Otherwise, only perform inference.

