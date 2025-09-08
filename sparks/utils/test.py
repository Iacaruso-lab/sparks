from typing import Any, List

import numpy as np
import torch

from sparks.utils.misc import identity
from sparks.models.sparks import SPARKS
from sparks.data.providers import StandardTargetProvider, TargetProvider


@torch.no_grad()
def test_on_batch(sparks: SPARKS,
                  inputs: torch.tensor,
                  target_provider: TargetProvider = None,
                  loss_fn: Any = None,
                  test_loss: float = 0,
                  **kwargs):
    """
    Tests the model on a batch of inputs and computes the loss.

    Args:
        sparks (SPARKS): The SPARKS model instance.
        inputs (torch.tensor): The input data for the batch of evaluation.
        latent_dim (int): The dimensionality of the latent space.
        tau_p (int): The size of the past window for the model to consider.
        tau_f (int, optional): The size of the future window for the model to predict. Default is 1.
        loss_fn (Any, optional): The loss function used to evaluate the model's predictions. Default is None.
        test_loss (float, optional): Initial value of loss for testing. Default is 0.
        targets (torch.tensor, optional): The target data for the batch of evaluation. Default is None.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        **kwargs (dict, optional): Additional keyword arguments for advanced configurations.

        Optional arguments include:
            - session_id (int): Session identifier when training with multiple sessions. Default is 0.
            - online (bool): If True, updates the model parameters at every time-step. Default is False.
            - burnin (int): The number of initial steps to exclude from training. Default is 0.
            - act (callable): Activation function to apply to the decoder outputs. Default is identity.

    Returns:
        test_loss (float, Optional): The computed loss for the current batch of data.
                                           Returns None if loss_fn is None.
        encoder_outputs_batch (torch.tensor): The outputs from the encoder.
        decoder_outputs_batch (torch.tensor): The outputs from the decoder.
    """

    session_id = kwargs.get('session_id', 0)
    burnin = kwargs.get('burnin', 0)
    act = kwargs.get('act', identity)
    batch_idxs = kwargs.get('batch_idxs', np.arange(len(inputs)))

    sparks.eval()
    sparks.encoder.zero_()
    encoder_outputs_batch = torch.zeros([len(inputs), sparks.latent_dim, sparks.tau_p]).to(sparks.device)
    decoder_outputs_batch = torch.Tensor().to(sparks.device)

    for t in range(burnin):
        encoder_outputs_batch, _, _, _ = sparks(inputs[..., t], encoder_outputs=encoder_outputs_batch, session_id=session_id)

    for t in range(burnin, inputs.shape[-1]):
        encoder_outputs_batch, decoder_outputs, _, _ = sparks(inputs[..., t], encoder_outputs=encoder_outputs_batch,
                                                              session_id=session_id)

        decoder_outputs_batch = torch.cat((decoder_outputs_batch,
                                           act(decoder_outputs).unsqueeze(2)), dim=-1)
    
        if loss_fn is not None:
            if t < inputs.shape[-1] - sparks.tau_f + 1:
                target = target_provider.get_target(batch_idxs, t, sparks.tau_f, sparks.device)
                test_loss += loss_fn(decoder_outputs, target).cpu() / (inputs.shape[-1] - sparks.tau_f + 1)
        else:
            test_loss = None

    return test_loss, encoder_outputs_batch.cpu(), decoder_outputs_batch.cpu()


@torch.no_grad()
def test(sparks: SPARKS,
         test_dls: List,
         loss_fn: Any = None,
         device: torch.device = 'cpu',
         **kwargs):
    """
    Tests the model on a dataset represented by a dataloader and computes the loss.

    Args:
        sparks (SPARKS): The SPARKS model instance.
        test_dls  (List[torch.utils.data.DataLoader]): Dataloaders for the testing data.
        latent_dim (int): The dimensionality of the latent space.
        tau_p (int): The size of the past window for the model to consider.
        tau_f (int, optional): The size of the future window for the model to predict. Default is 1.
        loss_fn (Any, optional): The loss function used to evaluate the model's predictions. Default is None.
        device (torch.device, optional): The device where the tensors will be allocated. Default is 'cpu'.
        **kwargs (dict, optional): Additional keyword arguments for advanced configurations.

        Optional arguments include:
            - session_ids (np.ndarray): Array of session identifiers when training with multiple sessions. Default: np.arange(len(test_dls)).
            - online (bool): If True, updates the model parameters at every time-step. Default is False.
            - burnin (int): The number of initial steps to exclude from training. Default is 0.
            - act (callable): Activation function to apply to the decoder outputs. Default is identity.

    Returns:
        test_loss (float): The computed loss for the test dataset.
        encoder_outputs (torch.tensor): The outputs from the encoder.
        decoder_outputs (torch.tensor): The outputs from the decoder.
    """

    encoder_outputs = torch.Tensor()
    decoder_outputs = torch.Tensor()
    test_loss = 0

    session_ids = kwargs.get('session_ids', np.arange(len(test_dls)))
    unsupervised = kwargs.get('unsupervised', False)

    for i, test_dl in enumerate(test_dls):
        test_iterator = iter(test_dl)
        for inputs, targets in test_iterator:
            if unsupervised:
                target_provider = StandardTargetProvider(inputs)
            else:
                target_provider = StandardTargetProvider(targets)
            test_loss, encoder_outputs_batch, decoder_outputs_batch = test_on_batch(sparks=sparks,
                                                                                    inputs=inputs,
                                                                                    target_provider=target_provider,
                                                                                    test_loss=test_loss,
                                                                                    loss_fn=loss_fn,
                                                                                    device=device,
                                                                                    session_id=session_ids[i],
                                                                                    **kwargs)

            encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch), dim=0)
            decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch), dim=0)

    return test_loss, encoder_outputs, decoder_outputs
