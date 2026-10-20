from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sparks.models.layers import DropPath, FeedForward
from sparks.models.scan import parallel_affine_scan
from sparks.models.state import CarriedState

# CUDA's attention kernels fail ("invalid configuration argument") on a batch above 65535, which
# the latent self-attention's batch of B * T timesteps easily exceeds, e.g. 141 trials of 600 bins.
_MAX_ATTENTION_BATCH = 65535


class PerceiverBlock(CarriedState):
    """
    Compresses the [N, D] per-neuron embeddings of each timestep into [bottleneck_dim, D] latents,
    so that the rest of the model does not depend on the number of neurons: learned latent queries
    read out a linear-attention key-value memory of the neuron population, decaying and
    accumulating over time, then exchange information through self-attention and a feed-forward.

    Each session has its own per-neuron identity embedding; everything else is shared across
    sessions.

    Args:
        n_neurons_per_session (Dict[str, int]): {session_id: number of neurons}.
        embed_dim (int): The embedding dimension, D. Must be divisible by n_heads.
        bottleneck_dim (int): The number of latents, K.
        n_heads (int): The number of attention heads. Default is 1.
        dropout (float): Dropout probability of the self-attention and feed-forward. Default is 0.
        drop_path (float): Stochastic-depth probability of each residual branch. Default is 0.
    """
    def __init__(self, n_neurons_per_session: Dict[str, int], embed_dim, bottleneck_dim, n_heads=1,
                 dropout=0., drop_path=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_heads = n_heads

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # Per-neuron identity, one per session since n_neurons differs across sessions; every
        # other component below (latents, cross/self-attention, ffn) is shared across sessions.
        self.neuron_identity = nn.ParameterDict()
        for session_id, n in n_neurons_per_session.items():
            self._init_neuron_identity(n, session_id)

        # Learnable latent queries
        self.latents = nn.Parameter(torch.empty(1, bottleneck_dim, embed_dim))
        nn.init.trunc_normal_(self.latents, std=1/np.sqrt(self.embed_dim))
        self.norm_latents = nn.LayerNorm(embed_dim)

        # Kernelized linear-attention readout that carries a decaying KV state across time
        # (Katharopoulos et al., "Transformers are RNNs") -- mirroring the Hebbian STDP layer's own
        # decaying eligibility traces, so the Perceiver's compression is recurrent too instead of
        # being the one stateless step between two recurrent stages.
        self.lin_head_dim = embed_dim // n_heads
        self.lin_q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Per-head decay of the recurrent KV state, sigmoid-bounded to (0, 1) and learnable
        # (analogous to the Hebbian layer's tau_s-derived trace decay). Init at 0 -> sigmoid(0)
        # = 0.5, a neutral starting point left for training to sharpen in either direction.
        self.linear_attn_decay = nn.Parameter(torch.zeros(n_heads))
        # Latent self-attention: lets the K compressed slots exchange information with each
        # other after reading from the neuron population, instead of each being an independent,
        # non-interacting summary. Cheap since K is small (O(K^2) vs. the O(N*K) cross-attention
        # already paid above).
        self.latent_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_self = nn.LayerNorm(embed_dim)
        # Feed-forward network for the latents
        self.ffn = FeedForward(embed_dim, embed_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path)

    def _init_neuron_identity(self, n_neurons: int, session_id: Union[str, int], device=None, dtype=None):
        param = nn.Parameter(torch.empty(1, 1, n_neurons, self.embed_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(param, std=1/np.sqrt(self.embed_dim))
        self.neuron_identity[str(session_id)] = param

    def add_session(self, n_neurons: int, session_id: Union[str, int]):
        """Adds a new session's per-neuron identity. Everything else in this block is shared."""
        sid = str(session_id)
        if sid in self.neuron_identity:
            raise ValueError(f"Session ID '{sid}' already exists in this Perceiver block.")
        self._init_neuron_identity(n_neurons, sid, device=self.latents.device, dtype=self.latents.dtype)

    def _linear_attn_readout(self, x, state=None):
        """
        The K latents read out a decaying key-value memory accumulated over the neuron population
        across time, via the exact parallel affine scan used elsewhere for the Hebbian STDP traces.

        Args:
            x (torch.Tensor): [B, T, N, D] neuron features (after neuron_identity and norm1).
            state (Tuple[torch.Tensor, torch.Tensor], optional): The key-value memory and its
                normaliser at the end of the previous chunk. Default is None (start from zero).

        Returns:
            The [B*T, bottleneck_dim, D] readout, and the state at the end of this chunk.
        """
        B, T, N, D = x.shape
        H, Dh = self.n_heads, self.lin_head_dim

        k = self.lin_k_proj(x).view(B, T, N, H, Dh)
        v = self.lin_v_proj(x).view(B, T, N, H, Dh)
        phi_k = F.elu(k) + 1.0  # feature map ensuring positivity (Katharopoulos et al.)

        # Per-timestep contribution: sum over the neuron axis N
        kv_step = torch.einsum('btnhk,btnhv->bthkv', phi_k, v)  # [B, T, H, Dh, Dh]
        z_step = phi_k.sum(dim=2)                                # [B, T, H, Dh]

        decay = torch.sigmoid(self.linear_attn_decay)            # [H], in (0, 1)
        a_kv = decay.view(1, 1, H, 1, 1).expand_as(kv_step)
        a_z = decay.view(1, 1, H, 1).expand_as(z_step)

        kv_init, z_init = state if state is not None else (None, None)
        state_kv = parallel_affine_scan(a_kv, kv_step, x_init=kv_init)  # [B, T, H, Dh, Dh], causal cumulative state
        state_z = parallel_affine_scan(a_z, z_step, x_init=z_init)     # [B, T, H, Dh]

        q = self.lin_q_proj(self.norm_latents(self.latents))  # [1, bottleneck_dim, D]
        phi_q = (F.elu(q) + 1.0).view(self.bottleneck_dim, H, Dh)

        numer = torch.einsum('khd,bthdv->btkhv', phi_q, state_kv)  # [B, T, K, H, Dh]
        denom = torch.einsum('khd,bthd->btkh', phi_q, state_z).unsqueeze(-1).clamp_min(1e-6)

        out = (numer / denom).reshape(B, T, self.bottleneck_dim, D)
        new_state = (state_kv[:, -1], state_z[:, -1])
        return out.reshape(B * T, self.bottleneck_dim, D), new_state

    def forward(self, x, session_id: Union[str, int], carry_state: bool = False):
        """
        Args:
            x (torch.Tensor): [B, T, N, D] per-neuron embeddings, from the Hebbian attention block.
            session_id (Union[str, int]): The session being processed -- selects its per-neuron
                identity and its carried KV state.
            carry_state (bool): If True, the linear-attention readout's KV state carries forward
                from the previous forward() call instead of starting from zero. State is detached
                before being stored (truncated BPTT). Default is False. Call reset_state() before
                the first chunk of a new, unrelated sequence.

        Returns:
            torch.Tensor: [B, T, bottleneck_dim * D] latents.
        """
        sid = str(session_id)
        B, T, N, D = x.shape

        x = self.norm1(x + self.neuron_identity[sid])

        # Expand learnable latents to match the flattened batch size
        # Shape: [B*T, bottleneck_dim, D]
        latents_expanded = self.latents.expand(B * T, -1, -1)  # Shape: [B*T, bottleneck_dim, D]

        # Recurrent readout: latents (Q) attend to a KV state decayed and accumulated over
        # time, instead of independently re-reading the neuron population at every timestep.
        state = self.load_carried_state(sid, carry_state)
        attn_out, new_state = self._linear_attn_readout(x, state=state)  # Shape: [B*T, bottleneck_dim, D]
        self.store_carried_state(sid, new_state, carry_state)

        latents = latents_expanded + self.drop_path(attn_out)  # Residual connection around cross attention

        # Latent self-attention over the K slots (pre-norm residual, same pattern as cross-attn above)
        latents_self_norm = self.norm_self(latents)
        self_attn_out = torch.cat([self.latent_self_attn(query=chunk, key=chunk, value=chunk, need_weights=False)[0]
                                   for chunk in latents_self_norm.split(_MAX_ATTENTION_BATCH)])
        latents = latents + self.drop_path(self_attn_out)

        ffn_out = self.ffn(latents)  # Shape: [B*T, bottleneck_dim, D]
        out = (latents + self.drop_path(ffn_out)).reshape(B, T, self.bottleneck_dim * D)  # Shape: [B,T, K * D]

        return out
