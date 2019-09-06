"""
Author: Lukas Gosch
Date: 5.9.2019
Description:
    Main
"""

import time
import argparse
from torch.utils.data import DataLoader
from dataset import ProteinDataset
from dataset import collate_sequences


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File containing protein sequences for "
                        + "classification.")
    parser.add_argument("-ff", "--format", default='fasta',
                        help = "File format of protein sequences. Must be "
                        + "supported by Biopythons Bio.SeqIO class.")
    parser.add_argument("-db", "--database", default='eggNOG5',
                        help="Database to classify against.")
    parser.add_argument("-t", "--tax", type=int, default=2,
                        help="Taxonomic level to use in specified database.")
    parser.add_argument("-a", "--architecture", default='v1',
                        help="Neural network architecture to use for "
                        + "classification.")
    return parser


def main(args = None):
    dataset = ProteinDataset(args.file, f_format=args.format)

    #batch_size = 16
    for i, el in enumerate(DataLoader(dataset, batch_size=5, num_workers=4,
                                    collate_fn=collate_sequences)):
        print(el.)
        if (i+1)%5 == 0:
            break

    return


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)