"""
Author: Lukas Gosch
Date: 4.9.2019
Description:
	Main
"""

import argparse

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="FASTA file containing protein sequences"
    				    + " for classification.")
    parser.add_argument("-db", "--database", default='eggNOG5',
    					help="Database to classify against.")
    parser.add_argument("-t", "--tax", type=int, default=2,
    					help="Taxonomic level to use in specified database.")
    parser.add_argument("-a", "--architecture", default='v1',
    					help="Neural network architecture to use for "
    					+ "classification.")
    return parser


def main(args = None):
	print(args)
	return



if __name__ == '__main__':
	parser = get_parser()
	args = parser.parse_args()
	main(args)