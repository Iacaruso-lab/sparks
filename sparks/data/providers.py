from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import numpy as np
import torch


class TargetProvider(ABC):
    """Abstract base class for providing targets at specific timesteps."""

    def __init__(self, targets: torch.tensor):
        self.targets = targets

    @abstractmethod
    def get_target(self, batch_idxs: Union[List, np.ndarray], timestep: int, 
                   tau_f: int, device: torch.device) -> torch.Tensor:
        """Get target for given batch index and timestep."""
        pass


class StandardTargetProvider(TargetProvider):
    """Standard target provider that loads targets from pre-computed tensors."""

    def __init__(self, targets: torch.tensor):
        super().__init__(targets=targets)

    def get_target(self, batch_idxs: Union[List, np.ndarray], timestep: int, 
                   tau_f: int, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
        """Return standard target slice."""
        return self.targets[..., timestep:timestep + tau_f].reshape(self.targets.shape[0], -1).to(device)


class AllenMoviesTargetProvider(TargetProvider):
    """Target provider for Allen movies that loads frames on-demand."""
    
    def __init__(self, 
                 frames: torch.tensor, 
                 mode: str,
                 dt: float):
    
        self.time_bin_edges = np.concatenate((np.arange(0, 30., dt), np.array([30.])))
        self.frames_bin_edges = np.concatenate((np.arange(1/30, 30, 1/30), np.array([30.])))
    
        self.frames_indices_per_timestep = torch.tensor([np.where(self.time_bin_edges[i] <= self.frames_bin_edges)[0][0]
                                                         for i in range(1, len(self.time_bin_edges))]).type(torch.long)

        self.mode = mode
        if mode == 'prediction':
            targets = self.frames_indices_per_timestep
        elif mode == 'reconstruction':
            targets = frames
        else:
            raise ValueError("Mode must be either 'prediction' or 'reconstruction'")

        super().__init__(targets=targets)

    def get_target(self, batch_idxs: Tuple[List, np.ndarray], timestep: int,
                   tau_f: int, device: torch.device) -> torch.Tensor:
        """Load and return frames for the given timestep."""

        if self.mode == 'prediction':
            return self.targets[timestep].unsqueeze(0).repeat_interleave(len(batch_idxs), dim=0).to(device)
        else:
            # Load frames for this timestep and tau_f window
            frames_indices = self.frames_indices_per_timestep[timestep: timestep + tau_f]
            target = self.targets[..., frames_indices].unsqueeze(0).repeat_interleave(len(batch_idxs), dim=0)
            return target.reshape(target.shape[0], -1).to(device)


class ShuffleTargetProvider(TargetProvider):
    """Standard target provider that loads targets from pre-computed tensors."""

    def __init__(self, targets: torch.tensor):
        super().__init__(targets=targets)

    def get_target(self, batch_idxs: Union[List, np.ndarray], timestep: int, 
                   tau_f: int, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
        """Return standard target slice."""
        return self.targets[torch.randperm(len(batch_idxs)), ..., 
                            timestep:timestep + tau_f].reshape(self.targets.shape[0], -1).to(device)
