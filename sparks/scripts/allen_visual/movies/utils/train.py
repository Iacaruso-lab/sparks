from typing import Any, List

import numpy as np
import torch

from sparks.utils.misc import LongCycler
from sparks.utils.train import train_on_batch
from sparks.data.base import StandardTargetProvider, AllenMoviesTargetProvider, DenoisingTargetProvider
from sparks.models.sparks import SPARKS



def train(sparks: SPARKS,
          train_dls: List,
          loss_fn: Any,
          optimizer: torch.optim.Optimizer,
          beta: float = 0.,
          **kwargs):
    """
    Trains the model on a batch of inputs.

    Args:
        sparks (SPARKS): The SPARKS model instance.
        train_dls (List): List of Dataloaders to train on, typically one per session.
        loss_fn (Any): The loss function used to evaluate the model's predictions.
        optimizer (torch.optim.Optimizer): The optimizer algorithm used to update the model's parameters.
        latent_dim (int): The dimensionality of the latent space.
        tau_p (int): The size of the past window for the model to consider.
        tau_f (int): The size of the future window for the model to predict.
        beta (float, optional): The regularization strength of the Kullback–Leibler divergence in the loss function.
                                Default is 0.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        **kwargs: Additional keyword arguments.

        Optional arguments include:
            - sess_ids (np.ndarray): Array of session identifiers when training with multiple sessions. Default: np.arange(len(train_dls)).
            - online (bool): If True, updates the model parameters at every time-step. Default is False.
            - burnin (int): The number of initial steps to exclude from training. Default is 0.

    Returns:
        None. The model parameters are updated inline.
    """

    mode = kwargs.get('mode', 'prediction')
    start_idx = kwargs.get('start_idx', 0)

    random_order = np.random.choice(np.arange(len(train_dls)), size=len(train_dls), replace=False)
    train_iterator = LongCycler([train_dls[i] for i in random_order])


    for i, (inputs, targets) in enumerate(train_iterator):
        if mode == 'unsupervised':
            target_provider = StandardTargetProvider(inputs)
        elif mode == 'prediction':
            target_provider = StandardTargetProvider(targets)
        elif mode == 'reconstruction':
            target_provider = AllenMoviesTargetProvider(frames=kwargs.get('frames'),
                                                        dt=kwargs.get('dt'))
        elif mode == 'denoising':
            target_provider = DenoisingTargetProvider(inputs)

        train_on_batch(sparks=sparks,
                       inputs=inputs,
                       target_provider=target_provider,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       beta=beta,
                       sess_id=random_order[i % len(train_dls)] + start_idx,
                       **kwargs)
