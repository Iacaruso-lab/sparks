from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import numpy as np
import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:

        """
        Abstract Dataset for spike loading

        """

        super(BaseDataset).__init__()

        if type(self) is BaseDataset:
            raise TypeError("BaseDataset is an abstract class and cannot be instantiated directly")

    @abstractmethod
    def get_spikes(self, index: int):
        pass

    @abstractmethod
    def get_target(self, index: int):
        pass

    def __getitem__(self, index: int):
        """
        :param: index: int
        Index
        :return: tuple: (data, idx)
        """

        return self.get_spikes(index), self.get_target(index)
