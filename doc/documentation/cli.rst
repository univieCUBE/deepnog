=========================
Deepnog CLI Documentation
=========================

Invocation:

::

    deepnog SEQUENCE_FILE [options] > predictions.csv

Basic Commands
==============

These options may be commonly tuned for a basic invocation for OG prediction.

::

    positional arguments:
      SEQUENCE_FILE         File containing protein sequences for classification.

    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      -o FILE, --out FILE   Store orthologous group predictions to outputfile. Per
                            default, write predictions to stdout. (default: None)
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
    -db {eggNOG5}, --database {eggNOG5}
                        Orthologous group/family database to use. (default:
                        eggNOG5)
    -t {1,2}, --tax {1,2}
                        Taxonomic level to use in specified database
                        (1 = root, 2 = bacteria) (default: 2)
    -nw INT, --num-workers INT
                        Number of subprocesses (workers) to use for data
                        loading. Set to a value <= 0 to use single-process
                        data loading. Note: Only use multi-process data
                        loading if you are calculating on a gpu (otherwise
                        inefficient)! (default: 0)
    -a {deepencoding}, --architecture {deepencoding}
                        Network architecture to use for classification.
                        (default: deepencoding)
    -w FILE, --weights FILE
                        Custom weights file path (optional) (default: None)
    -bs INT, --batch-size INT
                        Batch size used for prediction.Defines how many
                        sequences should be forwarded in the network at once.
                        With a batch size of one, the protein sequences are
                        sequentially classified by the network without
                        leveraging parallelism. Higher batch-sizes than the
                        default can speed up the prediction significantly if
                        on a gpu. On a cpu, however, they can be slower than
                        smaller ones due to the increased average sequence
                        length in the convolution step due to zero-padding
                        every sequence in each batch. (default: 1)

