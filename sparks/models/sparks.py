from typing import Union, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from sparks.models.encoders import HebbianEncoder
from sparks.models.dataclasses import HebbianAttentionConfig, AttentionConfig, ProjectionConfig
from sparks.models.decoders import mlp


class SPARKS(torch.nn.Module):
    """
    SPARKS model class.
    SPARKS is a VAE-based model designed for neural data.
    It consists of an encoder encompassing a Hebbian attention layer, and a decoder that can generally
    be any neural network architecture.
    """

    def __init__(self,
                 n_neurons_per_session: Union[int, List[int]],
                 embed_dim: int,
                 latent_dim: int,
                 bottleneck_dim: int,
                 tau_s: float = 1.0,
                 tau_p: int = 1,
                 tau_f: Optional[int] = 1,
                 id_per_session: Optional[List[Union[str, int]]] = None,
                 hebbian_config: Union[HebbianAttentionConfig, List[HebbianAttentionConfig]] = HebbianAttentionConfig(),
                 attention_config: AttentionConfig = AttentionConfig(),
                 projection_config: ProjectionConfig = ProjectionConfig(),
                 share_projection_head: bool = False,
                 decoder: Optional[torch.nn.Module] = None,
                 output_dim_per_session: Optional[Union[int, List[int]]] = None,
                 joint_decoder: bool = False,
                 device: torch.device = torch.device('cpu')) -> None:

        super().__init__()

        self.tau_p = tau_p
        self.tau_f = tau_f
        self.tau_s = tau_s
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.device = device

        if isinstance(n_neurons_per_session, int):
            n_neurons_per_session = [n_neurons_per_session]

        if id_per_session is None:
            id_per_session = [str(i) for i in range(len(n_neurons_per_session))]
        elif len(id_per_session) != len(n_neurons_per_session):
            raise ValueError("`id_per_session` must have the same length as `n_neurons_per_session`.")

        self.encoder = HebbianEncoder(n_neurons_per_session=n_neurons_per_session,
                                      embed_dim=embed_dim,
                                      latent_dim=latent_dim,
                                      bottleneck_dim=bottleneck_dim,
                                      tau_s=self.tau_s,
                                      id_per_session=id_per_session,
                                      hebbian_config=hebbian_config,
                                      attention_config=attention_config,
                                      projection_config=projection_config,
                                      share_projection_head=share_projection_head,
                                      device=device)
        
        if decoder is None:
            n_inputs_decoder = latent_dim * tau_p
            hid_dims = [int(np.mean([n_inputs_decoder, np.mean(output_dim_per_session)]))]
            if output_dim_per_session is None:
                raise ValueError("`output_dim_per_session` must be provided if `decoder` is None.")
            if isinstance(output_dim_per_session, int) and not joint_decoder:
                if len(n_neurons_per_session) == 1:
                    output_dim_per_session = [output_dim_per_session]
                else:
                    raise ValueError("`output_dim_per_session` must be a list if `joint_decoder` is False "
                    "and there is more than one session.")
            if isinstance(output_dim_per_session, int):
                output_dim_per_session = output_dim_per_session * tau_f
            else:
                output_dim_per_session = [dim * tau_f for dim in output_dim_per_session]

            decoder = mlp(in_dim=n_inputs_decoder, hidden_dims=hid_dims,
                          output_dim_per_session=output_dim_per_session,
                          id_per_session=id_per_session, joint_decoder=joint_decoder).to(device)

        self.decoder = decoder

    def forward(self, x: torch.Tensor, session_id: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor,
                                                                             torch.Tensor, torch.Tensor]:
        """
        The forward pass of the autoencoder.

        The steps of the forward pass are:
        1. Compute the latent representation of the input.
        2. Stack the latent representation in the encoder_outputs buffer of length tau_p.
        3. Obtain the prediction of the decoder.

        Args:
            inputs (torch.tensor): The input data tensor.
            encoder_outputs (torch.tensor): The collection of past encoder output tensors.
            session_id (int, optional): The session id. Default is 0.

        Returns:
            encoder_outputs (torch.tensor): The encoder outputs tensor with the newly computed encoder output appended.
            decoder_outputs (torch.tensor): The output tensor of the decoder.
            mu (torch.tensor): The mean of the latent distribution computed by the encoder.
            logvar (torch.tensor): The log-variance of the latent distribution computed by the encoder.
        """

        mu, logvar = self.encoder(x.float().to(self.device), session_id)
        enc_outputs = self.encoder.reparametrize(mu, logvar)

        decoder_outputs = self.decoder(self.pad_z(enc_outputs), session_id)

        return enc_outputs, decoder_outputs, mu, logvar

    def generate(self, z: torch.Tensor, session_id: Union[str, int]) -> torch.Tensor:
        """
        Generates an output from a given latent vector `z`.

        This method is useful for exploring the latent space by bypassing the encoder.

        Args:
            z (torch.Tensor): A tensor representing a point (or batch of points)
                              in the latent space.

        Returns:
            The output generated by the decoder.
        """
        return self.decoder(self.pad_z(z), session_id)

    def pad_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transforms a sequence of latents into sliding context windows.
        
        Args:
            z: Tensor of shape [B, T, D] (Batch, Time, Latent Dimension)
            
        Returns:
            Tensor of shape [B, T, tau_p * D]
        """
        B, T, D = z.shape
        
        # If the window is 1, no unfolding is necessary
        if self.tau_p == 1:
            return z
            
        # Padding
        # To predict step 't', we need [t - tau_p + 1 ... t]. 
        # For early time steps (t < tau_p), we must pad the start of the sequence with zeros.
        # PyTorch F.pad format for a 3D tensor: (last_dim_left, last_dim_right, second_last_left, second_last_right)
        # We pad the Time dimension (second to last) on the left by (tau_p - 1).
        z_padded = F.pad(z, (0, 0, self.tau_p - 1, 0))  # Shape: [B, T + self.tau_p - 1, D]
        
        # Sliding Window
        # .unfold(dimension, size, step) extracts sliding windows across the target dimension.
        # We unfold across Time (dim 1).
        # Shape becomes: [B, T, D, tau_p]
        z_windows = z_padded.unfold(dimension=1, size=self.tau_p, step=1)
        
        # Reorder and Flatten
        # We want the window elements to be chronological, so we swap D and tau_p.
        # .permute transforms [B, T, D, tau_p] -> [B, T, tau_p, D]
        z_reordered = z_windows.permute(0, 1, 3, 2).contiguous()
        
        # Finally, flatten the window and feature dimensions together
        # Shape: [B, T, tau_p * D]
        z_flat = z_reordered.view(B, T, self.tau_p * D)
        
        return z_flat

    def add_session(self, n_neurons: int, session_id: Union[str, int],
                    hebbian_config: Optional[HebbianAttentionConfig] = None,
                    new_output_dim: Optional[int] = None) -> None:

        """ 
        Adds a new session to the SPARKS model by incorporating a new Hebbian attention block in the encoder
        and adjusting the decoder to accommodate the new session.
        This method allows the model to adapt to new sessions with potentially different numbers of neurons.
        Args:
            n_neurons (int): The number of input neurons for the new session.
            session_id (Union[str, int]): A unique identifier for the new session.
            hebbian_config (HebbianAttentionConfig, optional): Configuration parameters for the new Hebbian block.
                                                              If None, default parameters are used.
        """
        self.encoder.add_neural_block(n_neurons=n_neurons,
                                      session_id=session_id,
                                      hebbian_config=hebbian_config)
        
        if not self.decoder.joint_decoder:
            self.decoder.out_layers[str(session_id)] = torch.nn.Linear(self.decoder.layers[-2].out_features, 
                                                                       new_output_dim).to(self.device)
    