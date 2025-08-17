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
        return NotImplementedError


def smooth(x, window=10):
    return np.convolve(x, np.ones(window) / window, mode='same')


"""
=======================================================================================================================

Adapted from https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/training/cyclers.py
"""


def cycle(iterable):
    # see https://github.com/pytorch/pytorch/issues/23900
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def alternate(*args):
    """
    Given multiple iterators, returns a generator that alternatively visit one element from each iterator at a time.

    Examples:
        >>> list(alternate(['a', 'b', 'c'], [1, 2, 3], ['Mon', 'Tue', 'Wed']))
        ['a', 1, 'Mon', 'b', 2, 'Tue', 'c', 3, 'Wed']

    Args:
        *args: one or more iterables (e.g. tuples, list, iterators) separated by commas

    Returns:
        A generator that alternatively visits one element at a time from the list of iterables
    """
    for row in zip(*args):
        yield from row


def cycle_datasets(loaders):
    """
    Given a dictionary mapping data_key into dataloader objects, returns a generator that alternately yields
    output from the loaders in the dictionary. The order of data_key traversal is determined by the first invocation to `.keys()`.
    To obtain deterministic behavior of key traversal, recommended to use OrderedDict.

    The generator terminates as soon as any one of the constituent loaders is exhausted.

    Args:
        loaders (dict): Dict mapping a data_key to a dataloader object.

    Yields:
        string, Any: data_key  and and the next output from the data loader corresponding to the data_key
    """
    keys = list(loaders.keys())
    # establish a consistent ordering across loaders
    ordered_loaders = [loaders[k] for k in keys]
    for data_key, outputs in zip(cycle(loaders.keys()), alternate(*ordered_loaders)):
        yield data_key, outputs


class Exhauster:
    """
    Given a dictionary of data loaders, mapping data_key into a data loader, steps through each data loader, moving onto the next data loader
    only upon exhausing the content of the current data loader.
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        for data_key, loader in self.loaders.items():
            for batch in loader:
                yield data_key, batch

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])


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


class ShortCycler:
    """
    Cycles through trainloaders until the loader with smallest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.min_batches = min([len(loader) for loader in self.loaders])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders]
        for loader, _ in zip((cycle(cycles)), range(len(self.loaders) * self.min_batches), ):
            yield next(loader)

    def __len__(self):
        return len(self.loaders) * self.min_batches


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class MultiSessionDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)


class RandomCycler:
    """
    Randomly selects a dataloader and yields its next item.
    Aware not to select a dataloader that has already been exhausted.
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.iterators = [iter(loader) for loader in loaders]
        self.active_loaders = list(range(len(self.loaders)))

    def __iter__(self):
        while self.active_loaders:
            loader_idx = random.choice(self.active_loaders)
            try:
                yield loader_idx, next(self.iterators[loader_idx])
            except StopIteration:
                self.active_loaders.remove(loader_idx)
                self.__iter__()
        
    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

"""
=======================================================================================================================
"""
