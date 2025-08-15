import math

import torch


class FeedForward(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.RMSNorm(dim),
            torch.nn.Linear(dim, 2 * dim, bias=False),
            torch.nn.GLU(),
            torch.nn.Linear(dim, 2 * dim, bias=False),
            torch.nn.GLU()
        )

    def forward(self, x):
        return self.net(x)


def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    attention = torch.nn.functional.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(torch.nn.Module):
    """
    From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv_proj = torch.nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o


class AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, n_heads=1):
        super(AttentionBlock, self).__init__()
        self.attention = MultiheadAttention(embed_dim, n_heads)
        self.ff = FeedForward(embed_dim)
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [batch_size, n_inputs, input_dim]
        x = self.norm(x)
        x = self.attention(x) + x
        x = self.ff(x) + x

        return x  # [batch_size, n_inputs, input_dim]

    def zero_(self):
        """
        For compatibility
        :return:
        """
        return

    def detach_(self):
        """
        For compatibility
        :return:
        """
        return

class TransformerEncoder(torch.nn.Module):
    """
    Linear encoder with attention layer
    """

    def __init__(self, n_inputs, embed_dim, latent_dim, n_layers=1, n_heads=1):
        """
        :param n_inputs: number of input neurons
        :param hidden_dims:
        :param latent_dim:
        :param tau_s: time constant for the attention
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

    def detach_(self):
        self.layers[1].detach_()

    def zero_(self):
        self.layers[1].zero_()
