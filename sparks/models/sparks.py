from typing import Union, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from sparks.models.encoders import HebbianEncoder
from sparks.models.dataclasses import HebbianAttentionConfig, ConvConfig, ProjectionConfig
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
                 tau_p: int = 1,
                 tau_f: Optional[int] = 1,
                 id_per_session: Optional[List[Union[str, int]]] = None,
                 hebbian_config: Union[HebbianAttentionConfig, List[HebbianAttentionConfig]] = HebbianAttentionConfig(),
                 conv_config: ConvConfig = ConvConfig(),
                 projection_config: ProjectionConfig = ProjectionConfig(),
                 share_projection_head: bool = False,
                 decoder: Optional[torch.nn.Module] = None,
                 output_dim_per_session: Optional[Union[int, List[int]]] = None,
                 joint_decoder: bool = False,
                 device: torch.device = torch.device('cpu')) -> None:

        super().__init__()

        self.tau_p = tau_p
        self.tau_f = tau_f
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.device = device
        self._carried_z_tail = {}  # session_id (str) -> last (tau_p - 1) latents, for pad_z

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
                                      id_per_session=id_per_session,
                                      hebbian_config=hebbian_config,
                                      conv_config=conv_config,
                                      projection_config=projection_config,
                                      share_projection_head=share_projection_head,
                                      device=device)
        
        if decoder is None:
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

            n_inputs_decoder = latent_dim * tau_p
            hid_dims = [4 * n_inputs_decoder]
            decoder = mlp(in_dim=n_inputs_decoder, hidden_dims=hid_dims,
                          output_dim_per_session=output_dim_per_session,
                          id_per_session=id_per_session, joint_decoder=joint_decoder).to(device)

        self.decoder = decoder

    def forward(self, x: torch.Tensor, session_id: Union[str, int],
               carry_state: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
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
            carry_state (bool): If True, the encoder's Hebbian attention/Perceiver recurrent
                state -- and, if tau_p > 1, the decoder's sliding-window tail -- carry forward
                from this session's previous forward() call instead of starting from zero/being
                zero-padded, so a long recording can be processed as consecutive chunks with
                continuous dynamics throughout. See HebbianEncoder.forward and pad_z. Call
                reset_state() before the first chunk of a new, unrelated sequence. Default is
                False (matches the original per-call-independent behavior).

        Returns:
            encoder_outputs (torch.tensor): The encoder outputs tensor with the newly computed encoder output appended.
            decoder_outputs (torch.tensor): The output tensor of the decoder.
            mu (torch.tensor): The mean of the latent distribution computed by the encoder.
            logvar (torch.tensor): The log-variance of the latent distribution computed by the encoder.
        """

        mu, logvar = self.encoder(x.float().to(self.device), session_id, carry_state=carry_state)
        enc_outputs = self.encoder.reparametrize(mu, logvar)

        sid = str(session_id)
        prev_tail = self._carried_z_tail.get(sid) if carry_state else None
        decoder_outputs = self.decoder(self.pad_z(enc_outputs, prev_tail=prev_tail), session_id)

        if carry_state and self.tau_p > 1:
            tail_source = enc_outputs if prev_tail is None else torch.cat([prev_tail, enc_outputs], dim=1)
            self._carried_z_tail[sid] = tail_source[:, -(self.tau_p - 1):].detach()

        return enc_outputs, decoder_outputs, mu, logvar

    def reset_state(self, session_id: Optional[Union[str, int]] = None) -> None:
        """
        Clears carried recurrent state (see forward's `carry_state`). Call before the first
        chunk of a new, unrelated sequence.

        Args:
            session_id: If given, resets only that session's encoder state. If None (default),
                resets every session's.
        """
        self.encoder.reset_state(session_id)

        if session_id is None:
            self._carried_z_tail.clear()
        else:
            self._carried_z_tail.pop(str(session_id), None)

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

    def pad_z(self, z: torch.Tensor, prev_tail: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transforms a sequence of latents into sliding context windows.

        Args:
            z: Tensor of shape [B, T, D] (Batch, Time, Latent Dimension)
            prev_tail: Optional tensor of shape [B, tau_p - 1, D] -- the last (tau_p - 1) latents
                from the end of the previous chunk, used to fill the window for this chunk's
                early timesteps instead of zeros. See forward's `carry_state`. Default is None
                (zero-pad, as before).

        Returns:
            Tensor of shape [B, T, tau_p * D]
        """
        B, T, D = z.shape

        # If the window is 1, no unfolding is necessary
        if self.tau_p == 1:
            return z

        # Padding
        # To predict step 't', we need [t - tau_p + 1 ... t].
        # For early time steps (t < tau_p), we must pad the start of the sequence with the tail
        # of the previous chunk if given (continuous dynamics across a chunk boundary), or zeros
        # otherwise (e.g. the true start of a sequence).
        if prev_tail is None:
            z_padded = F.pad(z, (0, 0, self.tau_p - 1, 0))  # Shape: [B, T + self.tau_p - 1, D]
        else:
            z_padded = torch.cat([prev_tail, z], dim=1)  # Shape: [B, T + self.tau_p - 1, D]
        
        # Sliding Window
        # .unfold(dimension, size, step) extracts sliding windows across the target dimension.
        # We unfold across Time (dim 1).
        # Shape becomes: [B, T, D, tau_p]
        z_windows = z_padded.unfold(dimension=1, size=self.tau_p, step=1)
        
        # Reorder and Flatten
        # We want the window elements to be chronological, so we swap D and tau_p.
        # .permute transforms [B, T, D, tau_p] -> [B, T, tau_p, D]
        # z_reordered = z_windows.permute(0, 1, 3, 2).contiguous()
        
        # Finally, flatten the window and feature dimensions together
        # Shape: [B, T, tau_p * D]
        z_flat = z_windows.flatten(start_dim=2)  # Shape: [B, T, tau_p * D]
        
        return z_flat
    
    def hebbian_forward(self, x: torch.Tensor, session_id: Union[str, int]) -> torch.Tensor:
        """
        Forward pass through only the Hebbian attention block for a given session.

        This method allows for isolating the contribution of the Hebbian attention mechanism,
        which can be useful for analysis or ablation studies.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, n_neurons) or (batch, seq_len, n_neurons).
            session_id (Union[str, int]): The identifier for the session being processed.
        Returns:
            torch.Tensor: The stdp coefficients tensor from the Hebbian attention block.
        """
        return self.encoder.hebbian_forward(x, session_id)

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
            # Read the width the new head needs off an existing one, rather than off the decoder's
            # hidden stack: every decoder in sparks.models.decoders owns `out_layers`, but only
            # `mlp` has a `layers` attribute, so keying on the latter broke for `linear`.
            in_features = next(iter(self.decoder.out_layers.values())).in_features
            self.decoder.out_layers[str(session_id)] = torch.nn.Linear(in_features,
                                                                       new_output_dim).to(self.device)
    