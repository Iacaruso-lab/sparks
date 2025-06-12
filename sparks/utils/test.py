from typing import Any, List

import numpy as np
import torch
import tqdm

from sparks.utils.misc import identity
from sparks.utils.vae import skip, ae_forward


def test_init(encoder: torch.nn.Module,
              inputs: torch.tensor,
              latent_dim: int,
              tau_p: int,
              device: torch.device,
              burnin: int = 0):
    """
    Initializes testing for a batch. Computes the number of time-steps, resets the state
    of the encoder, and initializes the encoder outputs.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        inputs (torch.tensor): The input data for the batch of testing.
        latent_dim (int): The dimensionality of the latent space.
        tau_p (int): The size of the past window for the model to consider.
        device (torch.device): The device where the tensors will be allocated.
        burnin (int, optional): The number of initial steps to exclude from testing. Default is 0.

    Returns:
        encoder_outputs_batch (torch.tensor): The initialized encoder outputs, with size (batch_size, latent_dim, tau_p).
        decoder_outputs_batch (torch.tensor): The initialized decoder outputs, with size (batch_size, -1).
        T (int): The number of time-steps in the inputs after excluding the burn-in period.
    """
    encoder.eval()
    T = inputs.shape[-1] - burnin
    encoder.zero_()
    encoder_outputs_batch = torch.zeros([len(inputs), latent_dim, tau_p]).to(device)
    decoder_outputs_batch = torch.Tensor().to(device)

    return encoder_outputs_batch, decoder_outputs_batch, T


@torch.no_grad()
def test(encoder: torch.nn.Module,
         decoder: torch.nn.Module,
         test_dls: List,
         latent_dim: int,
         tau_p: int,
         tau_f: int = 1,
         loss_fn: Any = None,
         device: torch.device = 'cpu',
         **kwargs):
    """
    Tests the model on a dataset represented by a dataloader and computes the loss.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
        test_dls  (List[torch.utils.data.DataLoader]): Dataloaders for the testing data.
        latent_dim (int): The dimensionality of the latent space.
        tau_p (int): The size of the past window for the model to consider.
        tau_f (int, optional): The size of the future window for the model to predict. Default is 1.
        loss_fn (Any, optional): The loss function used to evaluate the model's predictions. Default is None.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        **kwargs (dict, optional): Additional keyword arguments for advanced configurations.

    Returns:
        test_loss (float): The computed loss for the test dataset.
        encoder_outputs (torch.tensor): The outputs from the encoder.
        decoder_outputs (torch.tensor): The outputs from the decoder.
    """

    encoder_outputs = torch.Tensor()
    decoder_outputs = torch.Tensor()
    test_loss = 0

    sess_ids = kwargs.get('sess_ids', np.arange(len(test_dls)))

    for i, test_dl in tqdm.tqdm(enumerate(test_dls)):
        test_iterator = iter(test_dl)
        for inputs, targets in test_iterator:
            test_loss, encoder_outputs_batch, decoder_outputs_batch = test_on_batch(encoder=encoder,
                                                                                    decoder=decoder,
                                                                                    inputs=inputs,
                                                                                    targets=targets,
                                                                                    latent_dim=latent_dim,
                                                                                    tau_p=tau_p,
                                                                                    tau_f=tau_f,
                                                                                    test_loss=test_loss,
                                                                                    loss_fn=loss_fn,
                                                                                    device=device,
                                                                                    sess_id=sess_ids[i],
                                                                                    **kwargs)

            encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch), dim=0)
            decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch), dim=0)

    return test_loss, encoder_outputs, decoder_outputs


@torch.no_grad()
def test_on_batch(encoder: torch.nn.Module,
                  decoder: torch.nn.Module,
                  inputs: torch.tensor,
                  latent_dim: int,
                  tau_p: int,
                  tau_f: int = 1,
                  loss_fn: Any = None,
                  test_loss: float = 0,
                  targets: torch.tensor = None,
                  device: torch.device = 'cpu',
                  **kwargs):
    """
    Tests the model on a batch of inputs and computes the loss.

    Args:
        encoder (torch.nn.Module): The encoder module of the model.
        decoder (torch.nn.Module): The decoder module of the model.
        inputs (torch.tensor): The input data for the batch of evaluation.
        latent_dim (int): The dimensionality of the latent space.
        tau_p (int): The size of the past window for the model to consider.
        tau_f (int, optional): The size of the future window for the model to predict. Default is 1.
        loss_fn (Any, optional): The loss function used to evaluate the model's predictions. Default is None.
        test_loss (float, optional): Initial value of loss for testing. Default is 0.
        targets (torch.tensor, optional): The target data for the batch of evaluation. Default is None.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        **kwargs (dict, optional): Additional keyword arguments for advanced configurations.

    Returns:
        test_loss (float, Optional): The computed loss for the current batch of data.
                                           Returns None if loss_fn is None.
        encoder_outputs_batch (torch.tensor): The outputs from the encoder.
        decoder_outputs_batch (torch.tensor): The outputs from the decoder.
    """

    sess_id = kwargs.get('sess_id', 0)
    burnin = kwargs.get('burnin', 0)
    act = kwargs.get('act', identity)

    encoder_outputs_batch, decoder_outputs_batch, T = test_init(encoder=encoder,
                                                                inputs=inputs,
                                                                latent_dim=latent_dim,
                                                                tau_p=tau_p,
                                                                device=device,
                                                                burnin=burnin)
    inputs, targets, encoder_outputs = skip(encoder, encoder_outputs_batch, inputs,
                                            targets, device, num_steps=burnin, sess_id=sess_id)

    for t in range(T):
        encoder_outputs_batch, decoder_outputs, _, _ = ae_forward(encoder=encoder,
                                                                  decoder=decoder,
                                                                  inputs=inputs[..., t],
                                                                  encoder_outputs=encoder_outputs_batch,
                                                                  tau_p=tau_p,
                                                                  device=device,
                                                                  sess_id=sess_id)
        decoder_outputs_batch = torch.cat((decoder_outputs_batch,
                                           act(decoder_outputs).unsqueeze(2)), dim=-1)
    
        if loss_fn is not None:
            if t < T - tau_f + 1:
                target = targets[..., t:t + tau_f].reshape(targets.shape[0], -1).to(device)
                if isinstance(loss_fn, torch.nn.NLLLoss):
                    target = target[:, 0].long()
                test_loss += loss_fn(decoder_outputs, target).cpu() / (T - tau_f + 1)
        else:
            test_loss = None
    
    return test_loss, encoder_outputs_batch.cpu(), decoder_outputs_batch.cpu()
