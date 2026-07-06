from typing import Any, List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparks.models.dataclasses import HebbianAttentionConfig, ConvConfig


class HebbianTransformer(nn.Module):
    """
    Initialize a Hebbian Transformer Decoder.

    This architecture mirrors the encoder structure but is adapted for decoding tasks 
    (e.g., autoregressive spike generation or mapping contexts back to spikes).
    It features session-specific Hebbian Attention Blocks followed by shared 
    conventional attention blocks, ending with a projection head to the target output space.

    Args:
        n_neurons_per_session (Union[int, List[int]]): Number of input neurons for one or more sessions.
        embed_dim (int): The embedding dimension for the attention mechanism.
        output_dim_per_session (Optional[Union[int, List[int]]]): Output dimension for the predictions. 
                                                                  Defaults to n_neurons_per_session if None.
        id_per_session (Optional[List[Union[str, int]]]): Unique identifiers for each session.
        hebbian_config (Union[HebbianAttentionConfig, List[HebbianAttentionConfig]]): Config for Hebbian block.
        attention_config (AttentionConfig): Configuration for conventional attention layers.
        share_output_head (bool): If True, all sessions share a single output projection head.
        device (torch.device): The device for computations.
    """
    def __init__(self,
                 n_neurons_per_session: Union[int, List[int]],
                 embed_dim: int,
                 bottleneck_dim: int,
                 output_dim_per_session: Optional[Union[int, List[int]]] = None,
                 id_per_session: Optional[List[Union[str, int]]] = None,
                 hebbian_config: Union[HebbianAttentionConfig, List[HebbianAttentionConfig]] = HebbianAttentionConfig(),
                 conv_config: ConvConfig = ConvConfig(),
                 share_output_head: bool = False,
                 device: torch.device = torch.device('cpu')):
        super().__init__()

        # --- Parameter Normalization ---
        if isinstance(n_neurons_per_session, int):
            n_neurons_per_session = [n_neurons_per_session]
            
        if id_per_session is None:
            id_per_session = [i for i in range(len(n_neurons_per_session))]

        if output_dim_per_session is None:
            output_dim_per_session = n_neurons_per_session
        elif isinstance(output_dim_per_session, int):
            output_dim_per_session = [output_dim_per_session] * len(n_neurons_per_session)

        if isinstance(hebbian_config, HebbianAttentionConfig):
            hebbian_config = [hebbian_config] * len(n_neurons_per_session)

        self.session_ids = [str(session_id) for session_id in id_per_session]
        self.n_neurons_map = {session_id: n for session_id, n in zip(self.session_ids, n_neurons_per_session)}
        self.output_dim_map = {session_id: o for session_id, o in zip(self.session_ids, output_dim_per_session)}
        
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.share_output_head = share_output_head
        self.device = device

        # --- Layer Construction ---
        # Hebbian Attention Blocks (Session-specific)
        self.hebbian_blocks = nn.ModuleDict({
            session_id: hebbian_config[i].block_class(
                n_neurons=self.n_neurons_map[session_id],
                embed_dim=embed_dim,
                bottleneck_dim=bottleneck_dim,
                **hebbian_config[i].params
            ) for i, session_id in enumerate(self.session_ids)
        }).to(self.device)

        # Conventional Blocks (Shared)
        self.conventional_blocks = nn.Sequential(*[
            conv_config.block_class(d_model=embed_dim * bottleneck_dim, **conv_config.params)
            for _ in range(conv_config.n_layers)
        ]).to(self.device)

        # 3. Output Projection Heads (Session-specific or Shared)
        self.output_heads = self._create_output_heads()

        self.to(device)

    def _create_output_head(self, in_dim: int, output_dim: int) -> nn.Module:
        """Factory method to build the output projection head."""
        # Flattens the [batch, n_neurons, embed_dim] output and projects it to output_dim target
        head = nn.Linear(in_dim, output_dim)
        
        return head

    def _create_output_heads(self) -> nn.ModuleDict:
        """Creates output projection heads for each session or a single shared one."""
        if self.share_output_head:
            out_dim_list = list(self.output_dim_map.values())
            
            if not all(o == out_dim_list[0] for o in out_dim_list):
                raise ValueError("To share output heads, all sessions must have the same number of input/output dimensions.")
            
            shared_head = self._create_output_head(self.embed_dim * self.bottleneck_dim, out_dim_list[0])

            return nn.ModuleDict({session_id: shared_head for session_id in self.session_ids}).to(self.device)
        else:
            return nn.ModuleDict({
                session_id: self._create_output_head(self.embed_dim * self.bottleneck_dim, 
                                                     self.output_dim_map[session_id])

                for session_id in self.session_ids
            }).to(self.device)

    def add_neural_block(self, n_neurons: int, session_id: Union[str, int],
                         output_dim: Optional[int] = None,
                         hebbian_config: Optional[HebbianAttentionConfig] = None):
        """
        Dynamically adds a new Hebbian attention block and output head for a new session.
        """
        sid = str(session_id)
        if sid in self.session_ids:
            raise ValueError(f"Session ID '{sid}' already exists.")
        
        output_dim = output_dim if output_dim is not None else n_neurons
        hebbian_config = hebbian_config or HebbianAttentionConfig()

        # Create and Register the New Hebbian Block
        new_hebbian_block = hebbian_config.block_class(
            n_neurons=n_neurons,
            embed_dim=self.embed_dim,
            tau_s=self.tau_s,
            **hebbian_config.params
        ).to(self.device)

        self.hebbian_blocks[sid] = new_hebbian_block
        
        # 2. Create and Register the New Output Head
        if self.share_output_head:
            first_sid = self.session_ids[0]
            new_output_head = self.output_heads[first_sid]
        else:
            new_output_head = self._create_output_head(in_dim=self.embed_dim * self.bottleneck_dim, 
                                                       output_dim=output_dim).to(self.device)
        self.output_heads[sid] = new_output_head

        # 3. Update Internal State
        self.session_ids.append(sid)
        self.n_neurons_map[sid] = n_neurons
        self.output_dim_map[sid] = output_dim

    def forward(self, x: torch.Tensor, session_id: Union[str, int], **kwargs) -> torch.Tensor:
        """
        Forward pass through the HebbianTransformer.

        Args:
            x (torch.Tensor): Input tensor (e.g., preceding spikes). Shape: (batch, n_neurons).
            session_id (Union[str, int]): The identifier for the session being processed.

        Returns:
            torch.Tensor: The predicted output logits or reconstructed spikes.
        """
        session_id = str(session_id)
        if session_id not in self.hebbian_blocks:
            raise ValueError(f"Session ID '{session_id}' not found.")

        # Hebbian Attention
        h = self.hebbian_blocks[session_id](x.float().to(self.device)) # Shape: [B, T, N, D]

        # Conventional Attention
        _, T, _ = h.shape
        for conv_block in self.conventional_blocks:
            h = conv_block(h) # Shape: [B, T, bottleneck_dim * D]

        # 3. Output Projection
        out = self.output_heads[session_id](h)
        return None, out, None, None

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
