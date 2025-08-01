from typing import Any, Union, List

import numpy as np
import torch
import tqdm
from torch.nn import NLLLoss

from sparks.utils.misc import LongCycler
from sparks.utils.vae import ae_forward, skip


def train_init(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               inputs: torch.tensor,
               latent_dim: int,
               tau_p: int,
               burnin: int = 0,
               device: Union[str, torch.device] = 'cpu',
               ):
    """
    Initializes training for a batch. Computes the number of time-steps, resets the state
    of the encoder, and initializes the encoder outputs.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
        inputs (torch.tensor): The input data for the batch of training.
        latent_dim (int): The dimensionality of the latent space.
        tau_p (int): The size of the past window for the model to consider.
        burnin (int, optional): The number of initial steps to exclude from training. Default is 0.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.

    Returns:
        encoder_outputs (torch.tensor): The initialized encoder outputs, with size (batch_size, latent_dim, tau_p).
        loss (int): Initial loss, set to 0.
        T (int): The number of time-steps in the inputs after excluding the burn-in period.
    """

    encoder.train()
    decoder.train()

    T = inputs.shape[-1] - burnin
    encoder.zero_()
    encoder_outputs = torch.zeros([len(inputs), latent_dim, tau_p]).to(device)
    loss = 0

    return encoder_outputs, loss, T


def update_and_reset(encoder: torch.nn.Module,
                     decoder: torch.nn.Module,
                     loss: Any,
                     optimizer: torch.optim.Optimizer,
                     max_val: float = 0.5):
    """
    Computes clipped gradients, updates the model's weights using the computed loss and the optimizer, 
    and then resets the gradients. This function is typically called after every batch during training.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
        loss (torch.Tensor): The computed loss for the current batch of data.
        optimizer (torch.optim.Optimizer): The optimizer algorithm used to update the model's parameters.
        max_val (float, optional): The maximum value for gradient clipping. Default is 0.5.

    Returns:
        None. The model parameters and gradients are updated inline.
    """

    loss.backward()
    torch.nn.utils.clip_grad_value_(encoder.parameters(), max_val) 
    torch.nn.utils.clip_grad_value_(decoder.parameters(), max_val) 

    optimizer.step()

    encoder.zero_grad(set_to_none=True)
    decoder.zero_grad(set_to_none=True)


def train_on_batch(encoder: torch.nn.Module,
                   decoder: torch.nn.Module,
                   inputs: torch.tensor,
                   targets: torch.tensor,
                   loss_fn: Any,
                   optimizer: torch.optim.Optimizer,
                   latent_dim: int,
                   tau_p: int,
                   tau_f: int,
                   beta: float = 0.,
                   device: Union[str, torch.device] = 'cpu',
                   **kwargs):
    """
    Trains the model on a batch of inputs.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
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
            - sess_id (int): Session identifier when training with multiple sessions. Default is 0.
            - burnin (int): The number of initial steps to exclude from training. Default is 0.

    Returns:
        None. The model parameters are updated inline.
    """

    online = kwargs.get('online', False)
    sess_id = kwargs.get('sess_id', 0)

    # Number of burn-in timesteps
    burnin = kwargs.get('burnin', 0)

    encoder_outputs, loss, T = train_init(encoder, decoder, inputs, latent_dim, tau_p, burnin=burnin, device=device)
    inputs, targets, encoder_outputs = skip(encoder, encoder_outputs, inputs,
                                            targets, device, num_steps=burnin, sess_id=sess_id)

    for t in range(T - tau_f + 1):
        # Forward pass of the autoencoder
        encoder_outputs, decoder_outputs, mu, logvar = ae_forward(encoder=encoder,
                                                                  decoder=decoder,
                                                                  inputs=inputs[..., t],
                                                                  encoder_outputs=encoder_outputs,
                                                                  tau_p=tau_p,
                                                                  device=device,
                                                                  sess_id=sess_id)

        target = targets[..., t:t + tau_f].reshape(targets.shape[0], -1).to(device)

        if isinstance(loss_fn, NLLLoss):  # NLLLoss expects 0d or 1d targets
            target = target[:, 0].long()

        # Online updates the loss at every time-step
        if online:
            loss = loss_fn(decoder_outputs, target) - beta * 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            update_and_reset(encoder, decoder, loss, optimizer)
            encoder.detach_()
            encoder_outputs.detach_()
        else:
            loss += loss_fn(decoder_outputs, target) - beta * 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        torch.cuda.empty_cache()

    if not online:
        update_and_reset(encoder, decoder, loss, optimizer)

def train(encoder: torch.nn.Module,
          decoder: torch.nn.Module,
          train_dls: List,
          loss_fn: Any,
          optimizer: torch.optim.Optimizer,
          latent_dim: int,
          tau_p: int,
          tau_f: int,
          beta: float = 0.,
          device: Union[str, torch.device] = 'cpu',
          **kwargs):
    """
    Trains the model on a batch of inputs.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
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
            - sess_id (np.ndarray): Array of session identifiers when training with multiple sessions. Default: np.arange(len(train_dls)).

    Returns:
        None. The model parameters are updated inline.
    """

    sess_ids = kwargs.get('sess_ids', np.arange(len(train_dls)))

    random_order = np.random.choice(np.arange(len(train_dls)), size=len(train_dls), replace=False)
    train_iterator = LongCycler([train_dls[i] for i in random_order])

    for i, (inputs, targets) in enumerate(tqdm.tqdm(train_iterator)):
        train_on_batch(encoder=encoder,
                       decoder=decoder,
                       inputs=inputs,
                       targets=targets,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       latent_dim=latent_dim,
                       tau_p=tau_p,
                       tau_f=tau_f,
                       beta=beta,
                       device=device,
                       sess_id=sess_ids[random_order[i % len(train_dls)]],
                       **kwargs)
