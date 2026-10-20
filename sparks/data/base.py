from abc import ABCMeta, abstractmethod
from typing import Any, NamedTuple, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    A recording cut into trials. Item i is a (spikes, target) pair:

    - spikes: float32 tensor [T, N] -- timesteps, then neurons.
    - target: what the model is trained to predict for that trial, with time first when it
      varies over time (e.g. [T, F] stimulus features, or [T] class labels).

    Subclasses set `dt`, the duration of a timestep in seconds, and implement `__len__`,
    `n_neurons`, `get_spikes` and `get_target`.
    """
    dt: float

    @abstractmethod
    def __len__(self) -> int:
        """The number of trials."""

    @property
    @abstractmethod
    def n_neurons(self) -> int:
        """The number of neurons, N."""

    @abstractmethod
    def get_spikes(self, index: int) -> torch.Tensor:
        """The [T, N] spikes of trial `index`."""

    @abstractmethod
    def get_target(self, index: int) -> Any:
        """The target of trial `index`."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        return self.get_spikes(index), self.get_target(index)


class DataSplit(NamedTuple):
    """
    The train and test sets of one recording (or pseudo-population), with their dataloaders --
    what every `make_*_data` function returns. Unpacks like a tuple:

        train_dataset, test_dataset, train_loader, test_loader = make_monkey_reaching_data(...)
    """
    train_dataset: BaseDataset
    test_dataset: BaseDataset
    train_loader: DataLoader
    test_loader: DataLoader


def make_split(train_dataset: BaseDataset, test_dataset: BaseDataset,
               batch_size: int = 1, num_workers: int = 0) -> DataSplit:
    """Wraps a train and a test set in dataloaders: the training one shuffled, the test one in order."""
    return DataSplit(train_dataset, test_dataset,
                     DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                     DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers))
