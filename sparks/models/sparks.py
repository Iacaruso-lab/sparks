from typing import Union, List, Optional, Tuple
import torch
import torch.nn.functional as F

from sparks.models.configs import HebbianAttentionConfig, SharedConfig
from sparks.models.decoders import MLPDecoder
from sparks.models.encoders import HebbianEncoder
from sparks.models.state import CarriedState


class SPARKS(CarriedState):
    """
    SPARKS: Sequential Predictive Autoencoder for the Representation of spiKing Signals.

    The encoder embeds each timestep of a session's spikes with a Hebbian attention block, which
    tracks the correlations between neurons as they unfold, and compresses them into a
    stochastic latent with a KL-regularised Gaussian posterior. The decoder, which can be any
    network, predicts the next `tau_f` targets from the last `tau_p` latents. Trained to predict
    its own inputs, the model is a variational autoencoder; trained to predict other targets
    (e.g. behaviour or labels), a variational information bottleneck.

    One model can encode several recording sessions with different numbers of neurons into a
    single latent space: each session has its own Hebbian attention layers (and optionally its
    own decoder output head), everything else is shared. Sessions can be added after training
    with `add_session`.

    Args:
        n_neurons_per_session (Union[int, List[int]]): The number of neurons of each session.
        embed_dim (int): The dimension of each neuron's embedding.
        latent_dim (int): The dimension of the latent space.
        bottleneck_dim (int): The number of latent slots the Perceiver compresses each
            timestep's neurons into.
        tau_p (int): The number of past latents the decoder reads at each timestep. Default is 1.
        tau_f (int): The number of future timesteps the decoder predicts at each timestep.
            Default is 1.
        id_per_session (List[Union[str, int]], optional): The id of each session. Default is
            None (0, 1, ...).
        hebbian_config (HebbianAttentionConfig, optional): The encoder's front-end block.
            Default is None (HebbianAttentionConfig()).
        shared_config (SharedConfig, optional): The sequence blocks between the front end and the
            latent projection. Default is None (SharedConfig(), i.e. none).
        decoder (torch.nn.Module, optional): The decoder, called as decoder(windows, session_id)
            on [B, T, tau_p * latent_dim] windows of latents. Default is None, which builds an
            MLPDecoder from `output_dim_per_session`.
        output_dim_per_session (Union[int, List[int]], optional): The output dimension of each
            session's decoder head (or of the joint head), before the tau_f window. Required
            when `decoder` is None.
        joint_decoder (bool): If True, the default decoder has one output head shared by every
            session. Default is False.
        device (torch.device): The device to create the model on. Default is CPU.
    """

    def __init__(self,
                 n_neurons_per_session: Union[int, List[int]],
                 embed_dim: int,
                 latent_dim: int,
                 bottleneck_dim: int,
                 tau_p: int = 1,
                 tau_f: int = 1,
                 id_per_session: Optional[List[Union[str, int]]] = None,
                 hebbian_config: Optional[HebbianAttentionConfig] = None,
                 shared_config: Optional[SharedConfig] = None,
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
                                      shared_config=shared_config)
        
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
            decoder = MLPDecoder(in_dim=n_inputs_decoder, hidden_dims=hid_dims,
                                 output_dim_per_session=output_dim_per_session,
                                 id_per_session=id_per_session, joint_decoder=joint_decoder)

        self.decoder = decoder
        self.to(device)

    @property
    def device(self) -> torch.device:
        """The device the model's parameters are on (follows `.to()`)."""
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, session_id: Union[str, int],
               carry_state: bool = False) -> Tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor, torch.Tensor]:
        """
        Encodes a session's spikes into latents, and decodes each timestep's window of the last
        tau_p latents into a prediction of the next tau_f targets.

        Args:
            x (torch.Tensor): [B, T, N] spikes of the session's neurons.
            session_id (Union[str, int]): The session the spikes come from.
            carry_state (bool): If True, the encoder's recurrent state (Hebbian attention and
                Perceiver) and, if tau_p > 1, the decoder's window of past latents carry over from
                this session's previous carrying call instead of starting from zero, so that a long
                recording can be processed as consecutive chunks with continuous dynamics (see
                `CarriedState`). Call reset_state() before the first chunk of a new, unrelated
                sequence. Default is False.

        Returns:
            z (torch.Tensor): [B, T, latent_dim], the latents: samples from the posterior in train
                mode, its mean in eval mode (see HebbianEncoder.reparametrize).
            decoder_outputs (torch.Tensor): [B, T, output_dim * tau_f], the decoder's predictions.
            mu (torch.Tensor): [B, T, latent_dim], the mean of the latents' posterior.
            logvar (torch.Tensor): [B, T, latent_dim], the log-variance of the latents' posterior.
        """

        mu, logvar = self.encoder(x.float().to(self.device), session_id, carry_state=carry_state)
        enc_outputs = self.encoder.reparametrize(mu, logvar)

        sid = str(session_id)
        prev_tail = self.load_carried_state(sid, carry_state)  # the previous chunk's last tau_p - 1 latents
        decoder_outputs = self.decoder(self.pad_z(enc_outputs, prev_tail=prev_tail), session_id)

        if carry_state and self.tau_p > 1:
            tail_source = enc_outputs if prev_tail is None else torch.cat([prev_tail, enc_outputs], dim=1)
            self.store_carried_state(sid, tail_source[:, -(self.tau_p - 1):], carry_state)

        return enc_outputs, decoder_outputs, mu, logvar

    def generate(self, z: torch.Tensor, session_id: Union[str, int]) -> torch.Tensor:
        """
        Decodes given latents, bypassing the encoder, e.g. to explore the latent space.

        Args:
            z (torch.Tensor): [B, T, latent_dim] latents.
            session_id (Union[str, int]): The session whose decoder head to use.

        Returns:
            The decoder's [B, T, ...] predictions.
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
                (zero-pad).

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
        
        # Flatten the window and feature dimensions together. The result is grouped by latent
        # dimension (each dimension's tau_p window values are contiguous), not by timestep; the
        # decoder is trained on this layout, so reordering it would break existing checkpoints.
        # Shape: [B, T, D * tau_p]
        z_flat = z_windows.flatten(start_dim=2)
        
        return z_flat
    
    def hebbian_forward(self, x: torch.Tensor, session_id: Union[str, int],
                        carry_state: bool = False) -> torch.Tensor:
        """
        Forward pass through only the Hebbian attention block for a given session.

        This method allows for isolating the contribution of the Hebbian attention mechanism,
        which can be useful for analysis or ablation studies.

        Args:
            x (torch.Tensor): [B, T, N] spikes of the session's neurons.
            session_id (Union[str, int]): The session the spikes come from.
            carry_state (bool): If True, the Hebbian attention layer(s) pick up the STDP/trace
                state left by the previous carrying call instead of starting from zero, so that a
                recording too long to hold a [B, T, N, N] coefficient tensor for can be extracted
                in consecutive chunks with continuous dynamics (see
                `sparks.training.evaluation.get_hebbian_coeffs_on_batch`). Call `reset_state(session_id)`
                before the first chunk. Default is False.
        Returns:
            torch.Tensor: The [B, T, N, N] coefficients, or [n_heads, B, T, N, N] with several
                Hebbian attention heads.
        """
        return self.encoder.hebbian_forward(x.float().to(self.device), session_id,
                                            carry_state=carry_state)

    def add_session(self, n_neurons: int, session_id: Union[str, int],
                    new_output_dim: Optional[int] = None, **hebbian_overrides) -> None:
        """
        Adds a new session, with its own number of neurons, to the model.

        The new session gets its own Hebbian attention layer(s) and Perceiver neuron identity in
        the shared Hebbian attention block and, if the decoder has one output head per session,
        its own output head (any decoder with an `add_session(session_id, output_dim)` method,
        unless its `joint_decoder` is True). Everything else is shared with the existing sessions.

        Args:
            n_neurons (int): The number of input neurons for the new session.
            session_id (Union[str, int]): A unique identifier for the new session.
            new_output_dim (int, optional): The new session's output dimension, before the tau_f
                future window is applied (as for `output_dim_per_session`). Required when the
                decoder has one output head per session, and must be left out otherwise (joint or
                custom decoder).
            **hebbian_overrides: Settings of the new session's own Hebbian attention layer(s),
                e.g. `tau_s=0.5` or `data_type='calcium'` (see
                `HebbianAttentionBlock.add_session` for the full list). Any setting not
                given is inherited from the model.
        """
        has_session_heads = (hasattr(self.decoder, 'add_session')
                             and not getattr(self.decoder, 'joint_decoder', False))
        if has_session_heads and new_output_dim is None:
            raise ValueError("`new_output_dim` is required: the decoder has one output head per session.")
        if not has_session_heads and new_output_dim is not None:
            raise ValueError("`new_output_dim` can only be given when the decoder has one output head per "
                             "session, which this (joint or custom) decoder doesn't.")

        self.encoder.add_session(n_neurons=n_neurons, session_id=session_id, **hebbian_overrides)

        if has_session_heads:
            self.decoder.add_session(session_id, new_output_dim * self.tau_f)
    