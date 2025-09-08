from typing import List, Union, Optional, Tuple

import torch
import torch.nn as nn

from sparks.models.dataclasses import HebbianAttentionConfig, AttentionConfig, ProjectionConfig


class HebbianTransformer(nn.Module):
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
    def __init__(self,
                 n_neurons_per_session: Union[int, List[int]],
                 embed_dim: int,
                 latent_dim: int,
                 id_per_session: Optional[List[Union[str, int]]] = None,
                 hebbian_config: Union[HebbianAttentionConfig, List[HebbianAttentionConfig]] = HebbianAttentionConfig(),
                 attention_config: AttentionConfig = AttentionConfig(),
                 projection_config: ProjectionConfig = ProjectionConfig(),
                 share_projection_head: bool = False,
                 device: torch.device = torch.device('cpu')):
        super().__init__()

        # --- Parameter Normalization ---
        if isinstance(hebbian_config, HebbianAttentionConfig):
            hebbian_config = [hebbian_config] * len(n_neurons_per_session)

        self.session_ids = [str(session_id) for session_id in id_per_session] # Ensure string keys for ModuleDict
        self.n_neurons_map = {session_id: n for session_id, n in zip(self.session_ids, n_neurons_per_session)}
        
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.share_projection_head = share_projection_head
        self.projection_config = projection_config
        self.device = device

        # --- Layer Construction ---
        # 1. Hebbian Attention Blocks (Session-specific)
        self.hebbian_blocks = nn.ModuleDict({
            session_id: hebbian_config[i].block_class(
                n_neurons=self.n_neurons_map[session_id],
                embed_dim=embed_dim,
                **hebbian_config[i].params
            ) for i, session_id in enumerate(self.session_ids)
        })

        # 2. Conventional Attention Blocks (Shared)
        self.conventional_blocks = nn.Sequential(*[
            attention_config.block_class(embed_dim=embed_dim, **attention_config.params)
            for _ in range(attention_config.n_layers)
        ])

        # 3. Projection Heads (session-specific if sessions do not have the same number of neurons)
        self.projection_heads = self._create_projection_heads()
        
        self.to(device)

    def _create_projection_head(self, n_neurons: int) -> nn.Module:
        """Factory method to build the projection head."""
        # use a pre-built module if provided
        if self.projection_config.custom_head:
            return self.projection_config.custom_head

        head = nn.Sequential(nn.Flatten())
        mu_layer = nn.Linear(n_neurons * self.embed_dim, self.latent_dim)
        logvar_layer = nn.Linear(n_neurons * self.embed_dim, self.latent_dim)

        # wrap in a small container module
        return nn.ModuleDict({'head': head, 'mu': mu_layer, 'logvar': logvar_layer})

    def _create_projection_heads(self) -> nn.ModuleDict:
        """Creates projection heads for each session or a single shared one."""
        if self.share_projection_head:
            # Check for compatibility
            neurons_list = list(self.n_neurons_map.values())
            if not all(n == neurons_list[0] for n in neurons_list):
                raise ValueError("To share projection heads, all sessions must have the same number of neurons.")
            
            shared_head = self._create_projection_head(n_neurons=neurons_list[0])
            return nn.ModuleDict({session_id: shared_head for session_id in self.session_ids})
        else:
            return nn.ModuleDict({
                session_id: self._create_projection_head(n_neurons=n)
                for session_id, n in self.n_neurons_map.items()
            })

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

        # --- 1. Create and Register the New Hebbian Block ---
        # The configuration is inherited from the parent model's setup.
        new_hebbian_block = hebbian_config.block_class(
            n_neurons=n_neurons,
            embed_dim=self.embed_dim,
            **hebbian_config.params
        ).to(self.device)

        self.hebbian_blocks[sid] = new_hebbian_block
        
        # --- 2. Create and Register the New Projection Head ---
        if self.share_projection_head:
            # If heads are shared, simply map the new session ID to the existing head.
            # We can grab the head from the first session in the list.
            first_sid = self.session_ids[0]
            new_projection_head = self.projection_heads[first_sid]
        else:
            # If heads are session-specific, create a new one using our factory.
            # This reuses the logic from __init__ without any code duplication.
            new_projection_head = self._create_projection_head(n_neurons=n_neurons).to(self.device)

        self.projection_heads[sid] = new_projection_head

        # --- 3. Update Internal State ---
        self.session_ids.append(sid)
        self.n_neurons_map[sid] = n_neurons

    def forward(self, x: torch.Tensor, session_id: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the HebbianTransformerEncoder.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, n_neurons) or (batch, seq_len, n_neurons).
            session_id (Union[str, int]): The identifier for the session being processed.

        Returns:
            A tuple containing the mean (mu) and log-variance (logvar) of the latent distribution.
        """

        session_id = str(session_id)
        if session_id not in self.hebbian_blocks:
            raise ValueError(f"Session ID '{session_id}' not found.")

        # 1. Hebbian Attention
        h = self.hebbian_blocks[session_id](x) # Expected output: (batch, seq_len, embed_dim)

        # 2. Conventional Attention
        h = self.conventional_blocks(h)

        # 3. Projection Head
        projection = self.projection_heads[session_id]
        z = projection['head'](h)
        mu = projection['mu'](z)
        logvar = projection['logvar'](z)

        return mu, logvar

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterizes the input tensors using the reparameterization trick from the Variational AutoEncoder
        (VAE, Kingma et al. 2014).

        Args:
            mu (torch.Tensor): The mean of the normally-distributed latent space.
            logvar (torch.Tensor): The log variance of the normally-distributed latent space.

        Returns:
            torch.Tensor: The reparameterized tensor.
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def detach_(self):
        """
        Detach the attention layer of each Hebbian attention block from the computational graph.

        No Args.

        No Returns.
        """

        for session_id in self.session_ids:
            session_id = str(session_id)
            self.hebbian_blocks[session_id].detach_()

    def zero_(self):
        """
        Resets the values of the attention layer for each Hebbian attention block.

        No Args.

        No Returns.
        """

        for session_id in self.session_ids:
            session_id = str(session_id)
            self.hebbian_blocks[session_id].zero_()
