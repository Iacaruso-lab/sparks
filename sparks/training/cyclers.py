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
    Cycles through dataloaders in turn -- one batch from each -- until the one with the most
    batches is exhausted, restarting the shorter ones as needed. Needed for dataloaders of unequal
    size, e.g. one per recording session.

    Yields (loader_index, batch) pairs, loader_index being the position in `loaders` of the
    dataloader the batch comes from.
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders]
        for step in range(len(self)):
            loader_idx = step % len(self.loaders)
            yield loader_idx, next(cycles[loader_idx])

    def __len__(self):
        return len(self.loaders) * self.max_batches
