from typing import Any, List, Union

import numpy as np
import torch

from sparks.utils.misc import identity
from sparks.models.sparks import SPARKS
from sparks.models.transformer import HebbianTransformer
from sparks.data.providers import StandardTargetProvider, TargetProvider


@torch.no_grad()
def test_on_batch(model: Union[SPARKS, HebbianTransformer],
                  inputs: torch.tensor,
                  target_provider: TargetProvider = None,
                  loss_fn: Any = None,
                  test_loss: float = 0,
                  **kwargs):
    """
    Tests the model on a batch of inputs and computes the loss.

    Args:
        model (Union[SPARKS, HebbianTransformer]): The model instance.
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
    tau_f = getattr(model, 'tau_f', 1)

    model.eval()
    model.zero_()
    if isinstance(model, SPARKS):
        encoder_outputs_batch = torch.zeros([len(inputs), model.latent_dim, model.tau_p]).to(model.device)
    else:
        encoder_outputs_batch = None

    decoder_outputs_batch = torch.Tensor()

    for t in range(burnin):
        encoder_outputs_batch, _, _, _ = model(inputs[..., t], encoder_outputs=encoder_outputs_batch, 
                                               session_id=session_id)

    for t in range(burnin, inputs.shape[-1]):
        encoder_outputs_batch, decoder_outputs, _, _ = model(inputs[..., t], encoder_outputs=encoder_outputs_batch,
                                                              session_id=session_id)

        decoder_outputs_batch = torch.cat((decoder_outputs_batch, act(decoder_outputs).unsqueeze(2).cpu()), dim=-1)
    
        if loss_fn is not None:
            if t < inputs.shape[-1] - tau_f + 1:
                target = target_provider.get_target(batch_idxs, t, tau_f, model.device)
                test_loss += loss_fn(decoder_outputs, target).cpu() / (inputs.shape[-1] - tau_f + 1)
        else:
            test_loss = None

    if encoder_outputs_batch is not None:
        encoder_outputs_batch = encoder_outputs_batch.cpu()

    return test_loss, encoder_outputs_batch, decoder_outputs_batch


@torch.no_grad()
def test(model: Union[SPARKS, HebbianTransformer],
         test_dls: List,
         loss_fn: Any = None,
         device: torch.device = 'cpu',
         **kwargs):
    """
    Tests the model on a dataset represented by a dataloader and computes the loss.

    Args:
        model (Union[SPARKS, HebbianTransformer]): The model instance.
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

    if isinstance(model, SPARKS):
        encoder_outputs = torch.Tensor()
    else:
        encoder_outputs = None

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
            test_loss, encoder_outputs_batch, decoder_outputs_batch = test_on_batch(model=model,
                                                                                    inputs=inputs,
                                                                                    target_provider=target_provider,
                                                                                    test_loss=test_loss,
                                                                                    loss_fn=loss_fn,
                                                                                    device=device,
                                                                                    session_id=session_ids[i],
                                                                                    **kwargs)

            if encoder_outputs is not None:
                encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch), dim=0)
            decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch), dim=0)

    return test_loss, encoder_outputs, decoder_outputs


@torch.no_grad()
def test_on_batch_attn(model: Union[SPARKS, HebbianTransformer],
                       inputs: torch.tensor,
                       target_provider: TargetProvider = None,
                       loss_fn: Any = None,
                       test_loss: float = 0,
                       **kwargs):
    """
    Tests the model on a batch of inputs and computes the loss.

    Args:
        model (Union[SPARKS, HebbianTransformer]): The model instance.
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
        attn_coeffs_batch (torch.tensor): The attention coefficients from the Hebbian attention block.
        encoder_outputs_batch (torch.tensor): The outputs from the encoder.
        decoder_outputs_batch (torch.tensor): The outputs from the decoder.
    """

    session_id = kwargs.get('session_id', 0)
    burnin = kwargs.get('burnin', 0)
    act = kwargs.get('act', identity)
    batch_idxs = kwargs.get('batch_idxs', np.arange(len(inputs)))
    tau_f = getattr(model, 'tau_f', 1)

    model.eval()
    model.zero_()
    if isinstance(model, SPARKS):
        encoder_outputs_batch = torch.zeros([len(inputs), model.latent_dim, model.tau_p])
    else:
        encoder_outputs_batch = None
    decoder_outputs_batch = torch.Tensor()
    attn_coeffs_batch = torch.Tensor()

    for t in range(burnin):
        encoder_outputs, _, _, _ = model(inputs[..., t], encoder_outputs=encoder_outputs_batch, 
                                               session_id=session_id)

    for t in range(burnin, inputs.shape[-1]):
        encoder_outputs, decoder_outputs, _, _ = model(inputs[..., t], encoder_outputs=encoder_outputs_batch,
                                                       session_id=session_id)
        
        decoder_outputs_batch = torch.cat((decoder_outputs_batch,
                                           act(decoder_outputs).unsqueeze(2)).cpu(), dim=-1)
                
        if isinstance(model, SPARKS):
            encoder_outputs_batch = torch.cat((encoder_outputs_batch, encoder_outputs.unsqueeze(2)).cpu(), dim=-1)
            attn_coeffs_batch = torch.cat((attn_coeffs_batch,
                                           model.encoder.hebbian_blocks[str(session_id)].attention_layer.heads[0].attention.unsqueeze(3)).cpu(), dim=-1)
        else:
            attn_coeffs_batch = torch.cat((attn_coeffs_batch,
                                           model.hebbian_blocks[str(session_id)].attention_layer.heads[0].attention.unsqueeze(3)).cpu(), dim=-1)
    
        if loss_fn is not None:
            if t < inputs.shape[-1] - tau_f + 1:
                target = target_provider.get_target(batch_idxs, t, tau_f, model.device)
                test_loss += loss_fn(decoder_outputs, target).cpu() / (inputs.shape[-1] - tau_f + 1)
        else:
            test_loss = None

    return test_loss, encoder_outputs_batch, decoder_outputs_batch, attn_coeffs_batch