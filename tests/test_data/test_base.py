import pytest
import torch

from sparks.data.base import BaseDataset, DataSplit, make_split


class TrialsDataset(BaseDataset):
    dt = 0.01

    def __init__(self, n_trials=6):
        self.spikes = torch.arange(n_trials * 4 * 3, dtype=torch.float32).reshape(n_trials, 4, 3)

    def __len__(self):
        return len(self.spikes)

    @property
    def n_neurons(self):
        return self.spikes.shape[-1]

    def get_spikes(self, index):
        return self.spikes[index]

    def get_target(self, index):
        return torch.tensor([index])


def test_base_dataset_is_abstract():
    with pytest.raises(TypeError):
        BaseDataset()

    class MissingTarget(BaseDataset):
        dt = 0.01

        def __len__(self):
            return 1

        @property
        def n_neurons(self):
            return 1

        def get_spikes(self, index):
            return torch.zeros(1, 1)

    with pytest.raises(TypeError):  # an incomplete subclass can't be instantiated either
        MissingTarget()


def test_items_are_spikes_and_target():
    spikes, target = TrialsDataset()[2]

    assert spikes.shape == (4, 3) and target.tolist() == [2]


def test_make_split_shuffles_only_the_training_set():
    train, test = TrialsDataset(), TrialsDataset()

    split = make_split(train, test, batch_size=6)

    assert isinstance(split, DataSplit)
    train_dataset, test_dataset, train_loader, test_loader = split  # unpacks like a tuple
    assert train_dataset is train and test_dataset is test
    assert next(iter(test_loader))[1].flatten().tolist() == list(range(6))
    orders = {tuple(next(iter(train_loader))[1].flatten().tolist()) for _ in range(5)}
    assert len(orders) > 1
