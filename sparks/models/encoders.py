from typing import List, Union, Optional, Tuple

import torch
import torch.nn as nn

from sparks.models.dataclasses import HebbianAttentionConfig, ConvConfig


class HebbianEncoder(nn.Module):
    """
    Initialize a Hebbian Transformer Encoder.

    This transformer encoder includes a Hebbian Attention Block, and optional conventional attention blocks.

    Args:
        n_neurons_per_session (Union[int, List[int]]): Number of input neurons for one or more sessions.
        embed_dim (int): The embedding dimension for the attention mechanism.
        latent_dim (int): The dimensionality of the latent space.
        id_per_session (Optional[List[Union[str, int]]]): Unique identifiers for each session.
                                                    If None, defaults to `[0, 1, ..., N-1]`.
        hebbian_config (HebbianConfig): Configuration for the Hebbian attention block.
        attention_config (AttentionConfig): Configuration for conventional attention layers.
        device (torch.device): The device for computations.

    Returns:
        None
    """
    # Without beta (KL weight) tuned carefully, logvar has no pressure keeping it in a sane
    # range -- it's the raw, unconstrained output of a Linear layer. Left unclamped, this can (a)
    # inject unstable, run-to-run-varying amounts of reparameterization noise during training
    # (large logvar -> large std -> large eps*std, fresh every forward pass), and (b) genuinely
    # overflow float32 in kl_loss's logvar.exp() term for large enough values. Clamping bounds
    # both failure modes without needing beta to be perfectly tuned first.
    _LOGVAR_MIN = -4.0
    _LOGVAR_MAX = 2.0

    def __init__(self,
                 n_neurons_per_session: Union[int, List[int]],
                 embed_dim: int,
                 latent_dim: int,
                 bottleneck_dim: int,
                 id_per_session: Optional[List[Union[str, int]]] = None,
                 hebbian_config: Union[HebbianAttentionConfig, List[HebbianAttentionConfig]] = HebbianAttentionConfig(),
                 conv_config: ConvConfig = ConvConfig(),
                 device: torch.device = torch.device('cpu')):
        super().__init__()

        # The Hebbian attention block is now a single instance shared across every session (see
        # add_neural_block), so only one HebbianAttentionConfig applies at construction time --
        # a list is accepted for call-signature compatibility, but only its first element is used.
        if isinstance(hebbian_config, list):
            hebbian_config = hebbian_config[0]

        if isinstance(n_neurons_per_session, int):
            n_neurons_per_session = [n_neurons_per_session]
        if id_per_session is None:
            id_per_session = range(len(n_neurons_per_session))

        self.session_ids = [str(session_id) for session_id in id_per_session] # Ensure string keys for ModuleDict
        self.n_neurons_map = {session_id: n for session_id, n in zip(self.session_ids, n_neurons_per_session)}

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim
        self.device = device

        # Hebbian Attention Block (shared across sessions): each session contributes its own
        # Hebbian attention layer(s), sized to that session's own n_neurons, while the
        # head-mixing projection, FeedForward network, and Perceiver's cross/self-attention are
        # shared across sessions. See HebbianAttentionBlock.add_hebbian_layer.
        self.hebbian_block = hebbian_config.block_class(
            n_neurons_per_session=self.n_neurons_map,
            embed_dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
            **hebbian_config.params
        )

        # Conventional Blocks (Shared)
        self.conventional_blocks = nn.Sequential(*[
            conv_config.block_class(d_model=embed_dim * bottleneck_dim, **conv_config.params)
            for _ in range(conv_config.n_layers)
        ])

        # The Hebbian block and every conventional block are pre-norm (each sub-layer normalizes
        # its own input before transforming it) but none of them apply a final norm to what they
        # return -- so without this, mu/logvar's input directly inherits whatever scale the
        # residual stream happens to have accumulated, unbounded, especially with n_layers=0
        # (the default), where nothing at all sits between the raw Hebbian/Perceiver output and
        # the projection heads.
        self.pre_projection_norm = nn.LayerNorm(embed_dim * bottleneck_dim)

        self.projection_head = self._create_projection_head()
    
        self.to(device)

    def _create_projection_head(self) -> nn.Module:
        """Factory method to build the projection head."""
        
        mu_layer = nn.Linear(self.bottleneck_dim * self.embed_dim, self.latent_dim)
        logvar_layer = nn.Linear(self.bottleneck_dim * self.embed_dim, self.latent_dim)

        # wrap in a small container module
        return nn.ModuleDict({'mu': mu_layer, 'logvar': logvar_layer})

    def add_neural_block(self, n_neurons: int, session_id: Union[str, int], **hebbian_overrides):
        """
        Dynamically adds a new Hebbian attention layer for a new session to the shared Hebbian
        attention block.

        This is useful for fine-tuning or transfer learning, allowing the model to adapt to new
        data without retraining from scratch. Only the new session's own Hebbian attention
        layer(s) and Perceiver neuron identity are created -- the head-mixing projection,
        FeedForward network, Perceiver cross/self-attention, and projection head are already
        shared and are left untouched.

        Args:
            n_neurons (int): The number of input neurons for the new session.
            session_id (Union[str, int]): A unique identifier for the new session.
            **hebbian_overrides: Settings of the new session's own Hebbian attention layer(s),
                e.g. `tau_s=0.5` (see `HebbianAttentionBlock.add_hebbian_layer`). Any setting not
                given is inherited from this encoder's Hebbian attention block. Settings of the
                shared parts (n_heads, dropout, drop_path) can't differ between sessions.
        """

        sid = str(session_id)
        if sid in self.session_ids:
            raise ValueError(f"Session ID '{sid}' already exists. Please choose a unique ID.")

        self.hebbian_block.add_hebbian_layer(n_neurons=n_neurons, session_id=sid, **hebbian_overrides)
        self.hebbian_block.to(self.device)

        self.session_ids.append(sid)
        self.n_neurons_map[sid] = n_neurons

    def forward(self, x: torch.Tensor, session_id: Union[str, int],
               carry_state: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the HebbianEncoder.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, n_neurons) or (batch, seq_len, n_neurons).
            session_id (Union[str, int]): The identifier for the session being processed.
            carry_state (bool): If True, this session's Hebbian block carries its recurrent
                (STDP/trace/linear-attention) state forward from the previous forward() call for
                this session instead of starting from zero -- see HebbianAttentionBlock.forward.
                Call reset_state() before the first chunk of a new, unrelated sequence. Default
                is False (matches the original per-call-independent behavior).

        Returns:
            A tuple containing the mean (mu) and log-variance (logvar) of the latent distribution.
        """

        session_id = str(session_id)
        if session_id not in self.session_ids:
            raise ValueError(f"Session ID '{session_id}' not found.")

        # Hebbian Attention
        h = self.hebbian_block(x, session_id, carry_state=carry_state)

        # Conventional Attention
        _, T, _ = h.shape
        for conv_block in self.conventional_blocks:
            h = conv_block(h) # Shape: [B, T, bottleneck_dim * D]

        h = self.pre_projection_norm(h)

        # Projection Head
        mu = self.projection_head['mu'](h)
        logvar = self.projection_head['logvar'](h).clamp(self._LOGVAR_MIN, self._LOGVAR_MAX)

        return mu, logvar

    def reset_state(self, session_id: Optional[Union[str, int]] = None):
        """
        Clears carried recurrent state (see forward's `carry_state`). Call before the first
        chunk of a new, unrelated sequence.

        Args:
            session_id: If given, resets only that session's Hebbian attention layer(s). If None
                (default), resets every session.
        """
        if session_id is None:
            self.hebbian_block.reset_state()
        else:
            self.hebbian_block.reset_state(str(session_id))

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor,
                      sample: Optional[bool] = None) -> torch.Tensor:
        """
        Reparameterizes the input tensors using the reparameterization trick from the Variational AutoEncoder
        (VAE, Kingma et al. 2014).

        Args:
            mu (torch.Tensor): The mean of the normally-distributed latent space.
            logvar (torch.Tensor): The log variance of the normally-distributed latent space.
            sample (bool, optional): Whether to draw a sample from the posterior rather than return
                its mean. Default is None, which follows the module's train/eval mode: sampling is
                required during training (it is what makes the KL term meaningful and the gradient
                estimator unbiased), but at evaluation time it injects noise into every prediction
                the decoder makes, which is pure loss for any metric scoring the decoder's output
                per time-step. Reconstruction likelihoods are especially sensitive: since the
                output nonlinearity is not affine, the noise both adds variance and biases the
                predicted rate toward the middle of its range, which is the wrong direction for
                sparse spike trains. Pass True to force sampling in eval mode, e.g. to average
                several posterior draws into a posterior predictive estimate.

        Returns:
            torch.Tensor: A posterior sample if sampling, else `mu`.
        """

        if sample is None:
            sample = self.training

        if not sample:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def hebbian_forward(self, x: torch.Tensor, session_id: Union[str, int],
                        carry_state: bool = False) -> torch.Tensor:
        """
        Forward pass through only the Hebbian attention block for a given session.

        This method allows for isolating the contribution of the Hebbian attention mechanism,
        which can be useful for analysis or ablation studies.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, n_neurons) or (batch, seq_len, n_neurons).
            session_id (Union[str, int]): The identifier for the session being processed.
            carry_state (bool): If True, the Hebbian attention layer(s) carry their STDP/trace
                state forward from the previous carrying call instead of starting from zero, for
                walking a long recording through in consecutive chunks. Default is False.

        Returns:
            torch.Tensor: The stdp coefficients tensor from the Hebbian attention block.
        """

        session_id = str(session_id)
        if session_id not in self.session_ids:
            raise ValueError(f"Session ID '{session_id}' not found.")

        return self.hebbian_block.hebbian_forward(x, session_id, carry_state=carry_state)

