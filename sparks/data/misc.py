import numpy as np
import torch


def normalize(x):
    """
    Normalizes input tensor to [0, 1]
    """

    if isinstance(x, np.ndarray):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    elif isinstance(x, torch.Tensor):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    else:
        raise TypeError(f"normalize expects a numpy array or a torch tensor, got {type(x).__name__}.")


def smooth(x, window=10):
    return np.convolve(x, np.ones(window) / window, mode='same')


# ======================================================================================================================
# The cycling helpers below are adapted from neuralpredictors
# (https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/training/cyclers.py),
# Copyright (c) 2019 Sinz Lab, released under the MIT License -- see THIRD_PARTY_NOTICES.md for the
# full license text.


def cycle(iterable):
    # see https://github.com/pytorch/pytorch/issues/23900
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class LongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders]
        for loader, _ in zip((cycle(cycles)), range(len(self.loaders) * self.max_batches), ):
            yield next(loader)

    def __len__(self):
        return len(self.loaders) * self.max_batches
