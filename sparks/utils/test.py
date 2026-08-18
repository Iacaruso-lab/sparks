from typing import Any, List, Union

import numpy as np
import torch

from sparks.utils.misc import identity
from sparks.models.sparks import SPARKS
from sparks.models.transformer import HebbianTransformer


@torch.no_grad()
def test_on_batch(model: Union[SPARKS, HebbianTransformer],
                  inputs: torch.tensor,
                  targets: torch.tensor,
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
            - chunk_length (int): If given, walks through `inputs`/`targets` in consecutive,
                non-overlapping chunks of this many timesteps instead of one forward pass over the
                whole sequence, carrying the model's recurrent state across chunk boundaries via
                `carry_state=True` (see `SPARKS.forward`/`train_on_batch`) -- use this when a
                recording is long enough that a control block's internal per-timestep coefficient
                tensor (e.g. the [B, T, N, N] Hebbian/correlation history in
                `sparks.models.controls`) doesn't fit in memory over the whole sequence at once.
                Each chunk's outputs are moved to CPU before being concatenated back together,
                since holding every chunk on the accelerator simultaneously would defeat the point
                of chunking. Calls `model.reset_state(session_id)` once before the first chunk, so
                this always starts a fresh sequence -- call this function once per trial/recording,
                not once per chunk. Unlike the unchunked path (which runs the model on the full,
                untruncated `inputs` and only trims `decoder_outputs_batch` for the loss when
                tau_f > 1), the chunked path truncates `inputs`/`decoder_outputs_batch` by
                `tau_f - 1` up front, exactly like `train_on_batch`, so a tau_f > 1 window never
                needs future context spanning a chunk boundary; the returned decoder outputs are
                therefore `tau_f - 1` steps shorter than the unchunked path's when tau_f > 1. Only
                supported for SPARKS models (which implement `reset_state`/`carry_state`), not
                HebbianTransformer. `test_loss` is accumulated as a sum of each chunk's own
                (mean-reduced) loss -- like the sum over dataloader batches `test()` already does
                -- so it is not directly comparable to the unchunked path's single loss value for
                the same data; compare per-chunk-count-normalized values, or the returned
                decoder/encoder outputs directly, instead. Default is None (single pass over the
                whole sequence, as before).

    Returns:
        test_loss (float, Optional): The computed loss for the current batch of data.
                                           Returns None if loss_fn is None.
        encoder_outputs_batch (torch.tensor): The outputs from the encoder.
        decoder_outputs_batch (torch.tensor): The outputs from the decoder.
    """

    session_id = kwargs.get('session_id', 0)
    act = kwargs.get('act', identity)
    tau_f = getattr(model, 'tau_f', 1)
    device = kwargs.get('device', 'cpu')
    chunk_length = kwargs.get('chunk_length', None)

    model.eval()

    has_targets = (loss_fn is not None) and (targets is not None)

    if chunk_length is None:
        encoder_outputs_batch, decoder_outputs_batch, _, _ = model(inputs, session_id=session_id)
        if encoder_outputs_batch is not None:
            encoder_outputs_batch = encoder_outputs_batch.cpu()
        decoder_outputs_batch = act(decoder_outputs_batch).cpu()
    else:
        model.reset_state(session_id)
        T = inputs.shape[1]
        encoder_chunks = []
        decoder_chunks = []

        for start in range(0, T, chunk_length):
            end = min(start + chunk_length, T)
            chunk_inputs = inputs[:, start:end]

            encoder_outputs_chunk, decoder_outputs_chunk, _, _ = model(chunk_inputs, session_id=session_id,
                                                                        carry_state=True)
            decoder_outputs_chunk = act(decoder_outputs_chunk)

            if encoder_outputs_chunk is not None:
                encoder_chunks.append(encoder_outputs_chunk.cpu())
            decoder_chunks.append(decoder_outputs_chunk.cpu())

        encoder_outputs_batch = torch.cat(encoder_chunks, dim=1) if encoder_chunks else None
        decoder_outputs_batch = torch.cat(decoder_chunks, dim=1)

    if has_targets:
        if tau_f > 1:
            targets = targets.unfold(dimension=1, size=tau_f, step=1).permute(0, 1, 3, 2).reshape(decoder_outputs_batch.shape[0], decoder_outputs_batch.shape[1] - tau_f + 1, -1)
            test_loss += loss_fn(decoder_outputs_batch[:, :-(tau_f-1)], targets.cpu())
        else:
            test_loss += loss_fn(decoder_outputs_batch, targets.cpu())
    else:
        test_loss = None

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
            - chunk_length (int): Forwarded to `test_on_batch`. If given, each trial is walked
                through in consecutive chunks with recurrent state carried across them instead of
                one forward pass over the whole sequence -- see `test_on_batch` for details.
                Default is None (single pass, as before).

    Returns:
        test_loss (float): The computed loss for the test dataset.
        encoder_outputs (torch.tensor): The outputs from the encoder.
        decoder_outputs (torch.tensor): The outputs from the decoder.
    """

    test_loss = 0

    session_ids = kwargs.get('session_ids', np.arange(len(test_dls)))
    unsupervised = kwargs.get('unsupervised', False)

    decoder_outputs_list = []
    encoder_outputs_list = []

    for i, test_dl in enumerate(test_dls):
        encoder_outputs = torch.Tensor()
        decoder_outputs = torch.Tensor()

        test_iterator = iter(test_dl)
        for inputs, targets in test_iterator:
            test_loss, encoder_outputs_batch, decoder_outputs_batch = test_on_batch(model=model,
                                                                                    inputs=inputs,
                                                                                    targets=targets if not unsupervised else inputs,
                                                                                    test_loss=test_loss,
                                                                                    loss_fn=loss_fn,
                                                                                    device=device,
                                                                                    session_id=session_ids[i],
                                                                                    **kwargs)

            if encoder_outputs is not None:
                encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch.cpu()), dim=0)
            decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch.cpu()), dim=0)
    
        encoder_outputs_list.append(encoder_outputs)
        decoder_outputs_list.append(decoder_outputs)

    if len(encoder_outputs_list) == 1:
        encoder_outputs_list = encoder_outputs_list[0]
        decoder_outputs_list = decoder_outputs_list[0]

    return test_loss, encoder_outputs_list, decoder_outputs_list
