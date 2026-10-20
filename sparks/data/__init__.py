"""
Datasets of spiking recordings. Each dataset yields (spikes, target) trials, with spikes as
[T, N] float32 tensors (see BaseDataset), and each `make_*_data` function returns a DataSplit.

The loaders of each source live in their own subpackage, with their own optional dependencies:
`sparks.data.allen` (the `allen` extra), `sparks.data.nlb` (the `nlb` extra) and `sparks.data.mec`.
"""
from sparks.data.base import BaseDataset, DataSplit, make_split

__all__ = ['BaseDataset', 'DataSplit', 'make_split']
