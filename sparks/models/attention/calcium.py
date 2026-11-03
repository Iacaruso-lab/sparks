import torch

from sparks.models.attention.base import HebbianAttentionLayer
from sparks.models.scan import clamped_affine_scan


class CalciumAttentionLayer(HebbianAttentionLayer):
    """
    Hebbian attention head for calcium imaging. A calcium trace is already a decaying memory of past spikes, so the
    traces are used directly, without spike deconvolution: the coefficient of each pair of neurons, i -> j, integrates
    the difference of their traces with a leak of time constant tau_s, and is clipped to
    [min_attn_value, max_attn_value]:

        a_ij,t = clip(exp(-dt / tau_s) * a_ij,t-1 + x_i,t - x_j,t, min_attn_value, max_attn_value)

    Apart from its projection, v_proj, the layer has no parameters.

    Args:
        n_neurons (int): Number of neurons.
        embed_dim (int): The embedding dimension.
        min_attn_value (float): Lower bound of the coefficients. Default is -0.5.
        max_attn_value (float): Upper bound of the coefficients. Default is 1.5.
        tau_s (float): The time constant of the coefficients' leak, in seconds. Default is 1.
        dt (float): The time-step, in seconds. Default is 0.001.
    """

    def __init__(self,
                 n_neurons: int,
                 embed_dim: int,
                 min_attn_value: float = -0.5,
                 max_attn_value: float = 1.5,
                 tau_s: float = 1.0,
                 dt: float = 0.001):
        super().__init__(n_neurons=n_neurons, embed_dim=embed_dim, tau_s=tau_s, dt=dt)
        self.min_attn_value = min_attn_value
        self.max_attn_value = max_attn_value

    def stdp_coefficients(self, spikes: torch.Tensor, state=None):
        """
        Computes the coefficients at every timestep.

        Args:
            spikes (torch.Tensor): [B, T, N] calcium traces.
            state: Optional [B, N, N] coefficients from the end of a previous chunk, used as the initial condition
                instead of zeros -- lets a long recording be processed in consecutive chunks.

        Returns:
            The [B, T, N, N] coefficients (rows: pre-synaptic neuron i, columns: post-synaptic neuron j), and those at
            the last timestep, to pass as `state` for the next chunk.
        """
        pre = spikes.unsqueeze(-1)   # [B, T, N, 1], x_i
        post = spikes.unsqueeze(-2)  # [B, T, 1, N], x_j

        # recurrence whose per-step clip makes the scan sequential (see clamped_affine_scan)
        A = pre - post
        decay = self.get_parameters()  # [1, 1, 1]
        coefficients = clamped_affine_scan(decay, A, self.min_attn_value, self.max_attn_value, x_init=state)

        if torch.isnan(coefficients).any():
            raise ValueError("NaN values detected in attention coefficients.")

        return coefficients, coefficients[:, -1]

    def get_parameters(self):
        """The fixed per-step decay of the coefficients, exp(-dt / tau_s), as [1, 1, 1]."""
        decay = torch.exp(-self.dt / self.tau_s).view(1, 1, 1)

        return decay
