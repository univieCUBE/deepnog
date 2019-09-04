"""
Author: Lukas Gosch
Date: 4.9.2019
Description:
	Functions to preprocess proteins for classification.
"""

class ProteinDataset(Dataset):
	""" Protein dataset holding the proteins to classify. """

	def __init__(self, file, format = 'FASTA', max_length = None,
				 zero_padding = True):
		""" Initialize dataset.
			
		Parameters:
		-----------
		file : str
			Path to directory or file storing the protein sequences.
		format : str
			Format in which to expect the protein sequences.
		max_length : int
			If only proteins up to a certain length should be loaded.
			Defaults to None, meaning no length constraint
		zero_padding : bool
			Default behaviour is to zero pad all sequences up to
			the length of the longest one by appending zeros at the end. 
			If max_length is set, zero pads all sequences up to 
			max_length. False deactivates any zero padding.
		"""

		if format is 'FASTA':
			self.
		elif:
			raise NotImplementedError('Protein file format not supported.')