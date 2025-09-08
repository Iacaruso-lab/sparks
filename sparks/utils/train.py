from typing import Any, Union, List

import numpy as np
import torch
from torch.nn import NLLLoss

from sparks.data.misc import LongCycler
from sparks.utils.losses import kl_loss
from sparks.models.sparks import SPARKS
from sparks.data.providers import StandardTargetProvider, TargetProvider


def update_and_reset(sparks: SPARKS,
                     loss: Any,
                     optimizer: torch.optim.Optimizer,
                     max_val: float = 0.5):
    """
    Computes clipped gradients, updates the model's weights using the computed loss and the optimizer, 
    and then resets the gradients. This function is typically called after every batch during training.

    Args:
        sparks (SPARKS): The SPARKS model instance.
        loss (torch.Tensor): The computed loss for the current batch of data.
        optimizer (torch.optim.Optimizer): The optimizer algorithm used to update the model's parameters.
        max_val (float, optional): The maximum value for gradient clipping. Default is 0.5.

    Returns:
        None. The model parameters and gradients are updated inline.
    """

    loss.backward()
    torch.nn.utils.clip_grad_value_(sparks.parameters(), max_val) 

    optimizer.step()

    sparks.zero_grad(set_to_none=True)


def train_on_batch(sparks: SPARKS,
                   inputs: torch.tensor,
                   target_provider: TargetProvider,
                   loss_fn: Any,
                   optimizer: torch.optim.Optimizer,
                   beta: float = 0.,
                   device: Union[str, torch.device] = 'cpu',
                   **kwargs):
    """
    Trains the model on a batch of inputs.

    Args:
        sparks (SPARKS): The SPARKS model instance.
        inputs (torch.tensor): The input data for the batch of training.
        targets (torch.tensor): The target data for the batch of training.
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
            - online (bool): If True, updates the model parameters at every time-step. Default is False.
            - session_id (int): Session identifier when training with multiple sessions. Default is 0.
            - burnin (int): The number of initial steps to exclude from training. Default is 0.

    Returns:
        None. The model parameters are updated inline.
    """

    online = kwargs.get('online', False)
    session_id = kwargs.get('session_id', 0)
    batch_idxs = kwargs.get('batch_idxs', np.arange(len(inputs)))

    # Number of burn-in timesteps
    burnin = kwargs.get('burnin', 0)

    sparks.train()
    sparks.encoder.zero_()
    encoder_outputs = torch.zeros([len(inputs), sparks.latent_dim, sparks.tau_p]).to(device)
    loss = 0

    with torch.no_grad():
        for t in range(burnin):
            encoder_outputs, _, _, _ = sparks(inputs[..., t], encoder_outputs=encoder_outputs, session_id=session_id)

    for t in range(burnin, inputs.shape[-1] - sparks.tau_f + 1):
        encoder_outputs, decoder_outputs, mu, logvar = sparks(inputs[..., t], encoder_outputs=encoder_outputs,
                                                              session_id=session_id)

        target = target_provider.get_target(batch_idxs, t, sparks.tau_f, device)
        
        # Online updates the loss at every time-step
        if online:
            loss = kl_loss(decoder_outputs, target, loss_fn, mu, logvar, beta)
            update_and_reset(sparks, loss, optimizer)
            sparks.encoder.detach_()
            encoder_outputs.detach_()
        else:
            loss += kl_loss(decoder_outputs, target, loss_fn, mu, logvar, beta)
 
        torch.cuda.empty_cache()

    if not online:
        update_and_reset(sparks, loss, optimizer)

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
            - session_ids (np.ndarray): Array of session identifiers when training with multiple sessions. Default: np.arange(len(train_dls)).
            - online (bool): If True, updates the model parameters at every time-step. Default is False.
            - burnin (int): The number of initial steps to exclude from training. Default is 0.

    Returns:
        None. The model parameters are updated inline.
    """

    session_ids = kwargs.get('session_ids', np.arange(len(train_dls)))
    unsupervised = kwargs.get('unsupervised', False)

    random_order = np.random.choice(np.arange(len(train_dls)), size=len(train_dls), replace=False)
    train_iterator = LongCycler([train_dls[i] for i in random_order])

    for i, (inputs, targets) in enumerate(train_iterator):
        if unsupervised:
            target_provider = StandardTargetProvider(inputs)
        else:
            target_provider = StandardTargetProvider(targets)
        train_on_batch(sparks=sparks,
                       inputs=inputs,
                       target_provider=target_provider,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       beta=beta,
                       session_id=session_ids[random_order[i % len(train_dls)]],
                       **kwargs)
