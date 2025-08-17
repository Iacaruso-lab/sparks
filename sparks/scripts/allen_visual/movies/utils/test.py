from typing import Any, List

import numpy as np
import torch

from sparks.data.misc import LongCycler
from sparks.utils.test import test_on_batch
from sparks.data.base import StandardTargetProvider, AllenMoviesTargetProvider
from sparks.models.sparks import SPARKS


@torch.no_grad()
def calculate_test_acc(decoder_outputs: torch.tensor, frames: torch.tensor, tol: int = 30):
    """
    Calculate 1s-test accuracy for the Allen movies datasets
    Args:
        decoder_outputs (torch.tensor): SPARKS predictions.
        frames (torch.tensor): Targets.
        tol (int, optional): Tolerance in number of frames, defaults to 30 (corresponding to 1s).

    Returns:
        test_acc (float): The testing accuracy on this batch.
                          Accuracy is computed as the fraction of time-steps for which the predicted frame was within
                          1s of the correct one.
        encoder_outputs_batch (torch.Tensor): The outputs of the encoder.
        decoder_outputs_batch (torch.Tensor): The outputs of the decoder.
    """

    target_windows = [np.arange(t - tol, t + tol)[None, :] for t in frames[0].numpy()]
    test_acc = np.mean(np.array([[np.isin(decoder_outputs[k, :, t].cpu().argmax(dim=-1), target_windows[t])
                                  for t in range(decoder_outputs.shape[-1])]
                                 for k in range(decoder_outputs.shape[0])]))

    return test_acc

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
            - sess_ids (np.ndarray): Array of session identifiers when training with multiple sessions. Default: np.arange(len(test_dls)).
            - online (bool): If True, updates the model parameters at every time-step. Default is False.
            - burnin (int): The number of initial steps to exclude from training. Default is 0.
            - act (callable): Activation function to apply to the decoder outputs. Default is identity.

    Returns:
        test_loss (float): The computed loss for the test dataset.
        encoder_outputs (torch.tensor): The outputs from the encoder.
        decoder_outputs (torch.tensor): The outputs from the decoder.
    """

    mode = kwargs.get('mode', 'prediction')
    sess_ids = kwargs.get('sess_ids', np.arange(len(test_dls)))

    random_order = np.random.choice(np.arange(len(test_dls)), size=len(test_dls), replace=False)
    test_iterator = LongCycler([test_dls[i] for i in random_order])

    for i, test_dl in enumerate(test_dls):
        test_iterator = iter(test_dl)
        for inputs, targets in test_iterator:
            if np.isin(mode, ['unsupervised', 'denoising']):
                target_provider = StandardTargetProvider(inputs)
            elif mode == 'prediction':
                target_provider = StandardTargetProvider(targets)
            elif mode == 'reconstruction':
                target_provider = AllenMoviesTargetProvider(frames=kwargs.get('frames'),
                                                            dt=kwargs.get('dt'))
            test_loss, encoder_outputs_batch, decoder_outputs_batch = test_on_batch(sparks=sparks,
                                                                                    inputs=inputs,
                                                                                    target_provider=target_provider,
                                                                                    test_loss=test_loss,
                                                                                    loss_fn=loss_fn,
                                                                                    device=device,
                                                                                    sess_id=sess_ids[i],
                                                                                    **kwargs)

            encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch), dim=0)
            decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch), dim=0)

    if mode == 'prediction':
        test_acc = calculate_test_acc(decoder_outputs, frames=kwargs.get('frames'))
    else:
        test_acc = - test_loss
    
    return test_acc, encoder_outputs, decoder_outputs
