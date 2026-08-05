from typing import List, Union, Optional, Tuple

import torch
import torch.nn as nn

from sparks.models.dataclasses import HebbianAttentionConfig, ConvConfig, ProjectionConfig
from sparks.models.blocks import AttentionBlock


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
        projection_config (ProjectionConfig): Configuration for the final output projection head.
        share_projection_head (bool): If True, all sessions share a single projection head.
                                    Requires all sessions to have the same number of neurons.
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
                 projection_config: ProjectionConfig = ProjectionConfig(),
                 share_projection_head: bool = False,
                 device: torch.device = torch.device('cpu')):
        super().__init__()

        if isinstance(hebbian_config, HebbianAttentionConfig):
            hebbian_config = [hebbian_config] * len(n_neurons_per_session)

        self.session_ids = [str(session_id) for session_id in id_per_session] # Ensure string keys for ModuleDict
        self.n_neurons_map = {session_id: n for session_id, n in zip(self.session_ids, n_neurons_per_session)}
        
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim
        self.share_projection_head = share_projection_head
        self.projection_config = projection_config
        self.device = device

        # Hebbian Attention Blocks (Session-specific)
        self.hebbian_blocks = nn.ModuleDict({
            session_id: hebbian_config[i].block_class(
                n_neurons=self.n_neurons_map[session_id],
                embed_dim=embed_dim,
                bottleneck_dim=bottleneck_dim,
                **hebbian_config[i].params
            ) for i, session_id in enumerate(self.session_ids)
        })
    
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
        # use a pre-built module if provided
        if self.projection_config.custom_head:
            return self.projection_config.custom_head

        mu_layer = nn.Linear(self.bottleneck_dim * self.embed_dim, self.latent_dim)
        logvar_layer = nn.Linear(self.bottleneck_dim * self.embed_dim, self.latent_dim)

        # wrap in a small container module
        return nn.ModuleDict({'mu': mu_layer, 'logvar': logvar_layer})


    def add_neural_block(self, n_neurons: int, session_id: Union[str, int],
                         hebbian_config: Optional[HebbianAttentionConfig] = None):
        """
        Dynamically adds a new Hebbian attention block for a new session.

        This is useful for fine-tuning or transfer learning, allowing the model
        to adapt to new data without retraining from scratch. It creates and registers a new
        Hebbian block and a corresponding projection head for the given session ID.

        Args:
            n_neurons (int): The number of input neurons for the new session.
            session_id (Union[str, int]): A unique identifier for the new session.
            hebbian_config (HebbianConfig): Configuration parameters for the new Hebbian block.

        """

        sid = str(session_id)
        if sid in self.session_ids:
            raise ValueError(f"Session ID '{sid}' already exists. Please choose a unique ID.")
        
        if hebbian_config is None:
            hebbian_config = HebbianAttentionConfig()

        # Create and Register the New Hebbian Block ---
        # The configuration is inherited from the parent model's setup.
        new_hebbian_block = hebbian_config.block_class(
            n_neurons=n_neurons,
            embed_dim=self.embed_dim,
            bottleneck_dim=self.bottleneck_dim,
            **hebbian_config.params
        ).to(self.device)

        self.hebbian_blocks[sid] = new_hebbian_block
        
        self.session_ids.append(sid)
        self.n_neurons_map[sid] = n_neurons

    def forward(self, x: torch.Tensor, session_id: Union[str, int],
               carry_state: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the HebbianTransformerEncoder.

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
        if session_id not in self.hebbian_blocks:
            raise ValueError(f"Session ID '{session_id}' not found.")

        # Hebbian Attention
        h = self.hebbian_blocks[session_id](x, carry_state=carry_state)

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
            session_id: If given, resets only that session's Hebbian block. If None (default),
                resets every session's block.
        """
        if session_id is None:
            for block in self.hebbian_blocks.values():
                block.reset_state()
        else:
            self.hebbian_blocks[str(session_id)].reset_state()

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

        session_id = str(session_id)
        if session_id not in self.hebbian_blocks:
            raise ValueError(f"Session ID '{session_id}' not found.")

        return self.hebbian_blocks[session_id].hebbian_forward(x)


class TransformerEncoder(torch.nn.Module):
    """
    Linear encoder with attention layer
    """

    def __init__(self, n_inputs, embed_dim, latent_dim, n_layers=1, n_heads=1):
        """
        :param n_inputs: number of input neurons
        :param hidden_dims:
        :param latent_dim:
        :param device: device to use
        """

        super(TransformerEncoder, self).__init__()

        self.layers = torch.nn.ModuleList([torch.nn.Linear(n_inputs, embed_dim)])
        for _ in range(n_layers):
            self.layers.append(AttentionBlock(embed_dim, n_heads))

        self.fc_mu = torch.nn.Linear(embed_dim, latent_dim)
        self.fc_var = torch.nn.Linear(embed_dim, latent_dim)

    def forward(self, x, id=None):
        """
        Forward pass of the encoder
        :param spikes: spikes of the neurons [batch_size, n_neurons, n_timesteps]
        :return: encoded signal the output neurons [batch_size, latent_dim]
        """

        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)

        mu = self.fc_mu(x.flatten(1))
        logvar = self.fc_var(x.flatten(1))

        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu
