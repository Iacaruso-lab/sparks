from typing import Any, Optional, Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleList

from sparks.models.attention import MultiHeadedHebbianAttentionLayer
from sparks.models.transformer import FeedForward, AttentionBlock

class HebbianTransformerBlock(nn.Module):
    def __init__(self,
                 n_total_neurons: int,
                 embed_dim: int,
                 n_heads: int = 1,
                 tau_s: float = 0.5,
                 dt: int = 1,
                 neurons=None,
                 w_start: float = 1.,
                 alpha: float = 0.5,
                 data_type: str = 'ephys',
                 sliding: bool = False,
                 window_size: int = 1,
                 block_size: int = 1) -> None:

        """
        Initialize a Hebbian Transformer Block.

        This block includes a multi-headed Hebbian attention layer with pre- and post-synaptic weights and a
        linear output layer to mix the outputs of the different attention heads. It also includes a FeedForward network.

        Args:
            n_total_neurons (int): The total number of neurons.
            embed_dim (int): The number of dimensions for embedded output.
            n_heads (int, optional): The number of heads in the attention mechanism. Defaults to 1.
            tau_s (float, optional): The time constant for attention. Defaults to 0.5.
            dt (int, optional): The time step for attention update. Defaults to 1.
            neurons (optional): Neuron indices each head is going to pay attention to. If set to none,
                                    all neurons are attended.
            w_pre (float, optional): Pre-synaptic weight. Defaults to 1.0.
            w_post (float, optional): Post-synaptic weight. Defaults to 0.5.
            data_type (str, optional): Type of data being handled. Defaults to 'ephys'.
            sliding (bool, optional): whether to use the sliding window algorithm, default is False
            window_size (int, optional): window size for the sliding window, default is 10
            block_size (int, optional): block size for the sliding window, default is 3

        Returns:
            None
        """

        super(HebbianTransformerBlock, self).__init__()
        self.attention_layer = MultiHeadedHebbianAttentionLayer(n_total_neurons,
                                                                embed_dim,
                                                                n_heads=n_heads,
                                                                tau_s=tau_s,
                                                                dt=dt,
                                                                neurons=neurons,
                                                                w_start=w_start,
                                                                alpha=alpha,
                                                                data_type=data_type,
                                                                sliding=sliding,
                                                                window_size=window_size,
                                                                block_size=block_size)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.ff = FeedForward(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Forward pass of the encoder
        :param x: spikes of the neurons [batch_size, n_neurons]
        :return: encoded signal the output neurons [batch_size, n_inputs, embed_dim]
        """

        x = self.attention_layer(x)  # [batch_size, n_inputs, embed_dim]
        x = self.o_proj(x)  # [batch_size, n_inputs, embed_dim]
        x = self.ff(x) + x

        return x  # [batch_size, n_inputs, latent_dim]

    def detach_(self):
        """
        Detach the attention layer in the block from the computational graph.

        No Args.

        No Returns.
        """

        self.attention_layer.detach_()

    def zero_(self):
        """
        Resets the values of the attention layer in the current block.

        No Args.

        No Returns.
        """

        self.attention_layer.zero_()


class HebbianTransformerEncoder(nn.Module):
    def __init__(self,
                 n_neurons_per_sess: Union[int, List[int]],
                 embed_dim: int,
                 latent_dim: int,
                 tau_s_per_sess: Union[float, List[float]],
                 dt_per_sess: Union[float, List[float]],
                 n_layers: int = 0,
                 output_type: str = 'flatten',
                 share_outputs=False,
                 n_heads: int = 1,
                 id_per_sess: Optional[np.array] = None,
                 neurons_per_sess: Optional[Union[Any, List[Any]]] = None,
                 w_start: float = 1.,
                 alpha: float = 1.,
                 data_type: str = 'ephys',
                 sliding: bool = False,
                 window_size: int = 1,
                 block_size: int = 1,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize a Hebbian Transformer Encoder.

        This transformer encoder includes a Hebbian Attention Block, and optional conventional attention blocks.


        Args:
            n_neurons_per_sess (Union[int, List[int]]): Number of input neurons per session.
                                                        By passing a list, the model can learn to embed the data
                                                        from several sessions and animals within the same latent space
            embed_dim (int): The number of dimensions for the attention embeddings.
            latent_dim (int): The number of dimensions of the latent space.
            tau_s_per_sess (Union[float, List[float]]): The time constant for attention for each session.
            dt_per_sess (int): The time-step for attention update for each session.
            n_layers (Optional, int): The number of conventional attention layers. Default is 0.
            output_type (str, optional): defines the type of output (i.e., 'flatten'). Defaults to 'flatten'.
                                        If 'flatten': flattens the output of the last attention block and projects it
                                        onto the latent dimension
                                        If 'mean': compute the mean of the last attention block over the embedding
                                        dimension and projects it to the latent dimension
            n_heads (int, optional): The number of heads in the attention mechanism. Defaults to 1.
            id_per_sess (Optional[np.array]): Ids of the sessions to use the correct Hebbian attention layer during
                                    the forward pass. None will set an incremental id to each session, defaults to None.
            neurons_per_sess (Optional[Union[Any, List[Any]]]): Neuron indices each session is going to pay attention to
                                                None will make the block pay attention to all neurons, defaults to None.
            w_pre (float, optional): initial value for the weights of the presynaptic neurons, defaults to 1.0.
            w_post (float, optional): initial value for the weights of the postsynaptic neurons, defaults to 0.5.
            data_type (str, optional): 'ephys' or 'ca'.
            sliding (bool, optional): whether to use the sliding window algorithm, default is False
            window_size (int, optional): window size for the sliding window, default is 10
            block_size (int, optional): block size for the sliding window, default is 3
            device (torch.device, optional): The device to use for computation. Defaults to CPU.

        Returns:
            None
        """

        super(HebbianTransformerEncoder, self).__init__()

        if not hasattr(n_neurons_per_sess, '__iter__'):
            n_neurons_per_sess = [n_neurons_per_sess]
        if not hasattr(tau_s_per_sess, '__iter__'):
            tau_s_per_sess = [tau_s_per_sess] * len(n_neurons_per_sess)
        if not hasattr(dt_per_sess, '__iter__'):
            dt_per_sess = [dt_per_sess] * len(n_neurons_per_sess)
        if neurons_per_sess is not None:
            if not hasattr(neurons_per_sess, '__iter__'):
                neurons_per_sess = [neurons_per_sess] * len(n_neurons_per_sess)
        else:
            neurons_per_sess = [np.arange(n_neurons_sess) for n_neurons_sess in n_neurons_per_sess]

        if id_per_sess is None:
            id_per_sess = np.arange(len(n_neurons_per_sess))
        self.id_per_sess = id_per_sess

        self.output_type = output_type
        self.share_outputs = share_outputs
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.device = device

        self.sliding = sliding

        if self.sliding:
            self.block_size = block_size
            self.window_size = window_size
            for i in range(len(neurons_per_sess)):
                if (len(neurons_per_sess[i]) % block_size) != 0:
                    neurons_per_sess[i] = neurons_per_sess[i][:-(len(neurons_per_sess[i]) % block_size)]
                    n_neurons_per_sess[i] = len(neurons_per_sess[i])

        self.hebbian_attn_blocks = ModuleList([HebbianTransformerBlock(n_neurons,
                                                                       embed_dim,
                                                                       n_heads=n_heads,
                                                                       tau_s=tau_s,
                                                                       dt=dt,
                                                                       neurons=neurons,
                                                                       w_start=w_start,
                                                                       alpha=alpha,
                                                                       sliding=sliding,
                                                                       window_size=window_size,
                                                                       block_size=block_size,
                                                                       data_type=data_type)
                                               for (n_neurons, tau_s, dt, neurons) in zip(n_neurons_per_sess,
                                                                                          tau_s_per_sess,
                                                                                          dt_per_sess,
                                                                                          neurons_per_sess)])

        self.conventional_blocks = ModuleList()
        for _ in range(n_layers):
            self.conventional_blocks.append(AttentionBlock(embed_dim, n_heads))

        self.proj = None
        self.fc_mu_per_sess = None
        self.fc_var_per_sess = None
        self.norm_per_sess = None

        if share_outputs:
            assert (np.array(n_neurons_per_sess) == n_neurons_per_sess[0]).all(), 'Outputs can only be shared if all layers have the same number of neurons'
            self.init_weights(n_neurons_per_sess[:1])
        else:            
            self.init_weights(n_neurons_per_sess)

    def forward(self, x: torch.Tensor, sess_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Forward propagation through the HebbianTransformerEncoder.

        In the forward pass, the input signal is passed through the Hebbian attention blocks first,
        followed by the conventional blocks.
        These outputs are then normalized and passed through fully connected layers to obtain
        the mean and log variance of the latent distribution.

        Args:
            x (torch.Tensor): Input tensor containing spikes of the neurons.
            sess_id (int, optional): Session identifier for which the propagation should be performed, defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log variance of the latent distribution.

        """

        layer_idx = np.where(self.id_per_sess == sess_id)[0][0]
        if self.share_outputs:
            output_layer_idx = 0
        else:
            output_layer_idx = layer_idx

        x = self.hebbian_attn_blocks[layer_idx](x)

        for block in self.conventional_blocks:
            x = block(x)
        
        x = self.proj[output_layer_idx](x)
        x = self.norm_per_sess[output_layer_idx](x)
        mu = self.fc_mu_per_sess[output_layer_idx](x)
        logvar = self.fc_var_per_sess[output_layer_idx](x)

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

    def add_neural_block(self,
                         n_neurons: int,
                         tau_s: float,
                         dt: int,
                         layer_id: int,
                         w_pre: float = 1.,
                         w_post: float = 0.5,
                         data_type: str = 'ephys',
                         sliding: bool = False,
                         window_size: int = 1,
                         block_size: int = 1,
                         neurons: Optional[Any] = None) -> None:
        """
        Add a neural block to the HebbianTransformerEncoder.

        This function creates a new instance of the HebbianTransformerBlock with the given parameters,
        adds it to the list of existing blocks, and updates the session id list accordingly.

        Args:
            n_neurons (int): The number of neurons in the block.
            tau_s (float): The time constant for the attention mechanism of the block.
            dt (int): The time step used for updating the attention.
            layer_id (int): The id for the new layer.
            w_pre (float, optional): The initial value of weights for pre-synaptic neurons. Defaults to 1.0.
            w_post (float, optional): The initial value of weights for post-synaptic neurons. Defaults to 0.5.
            data_type (str, optional): 'ephys' or 'ca'.
            neurons (Any, optional): The specific neurons this block should pay attention to.
                If None, the block will pay attention to all neurons. Defaults to None.
            sliding (bool, optional): whether to use the sliding window algorithm, default is False
            window_size (int, optional): window size for the sliding window, default is 10
            block_size (int, optional): block size for the sliding window, default is 3

        Returns:
            None
        """

        if np.isin(layer_id, self.id_per_sess):
            raise ValueError('id already allocated to another layer')

        if neurons is None:
            neurons = np.arange(n_neurons)

        if sliding:
            if (len(neurons) % block_size) != 0:
                neurons = neurons[:-(len(neurons) % block_size)]
                n_neurons = len(neurons)

        self.hebbian_attn_blocks.append(HebbianTransformerBlock(n_neurons,
                                                                self.embed_dim,
                                                                n_heads=self.n_heads,
                                                                tau_s=tau_s,
                                                                dt=dt,
                                                                neurons=neurons,
                                                                w_pre=w_pre,
                                                                w_post=w_post,
                                                                data_type=data_type,
                                                                sliding=sliding,
                                                                window_size=window_size,
                                                                block_size=block_size).to(self.device))

        self.id_per_sess = np.concatenate((self.id_per_sess, np.array([layer_id])))

        if self.output_type == 'flatten':
            self.norm_per_sess.append(nn.LayerNorm(n_neurons * self.embed_dim).to(self.device))
            self.fc_mu_per_sess.append(nn.Linear(n_neurons * self.embed_dim, self.latent_dim).to(self.device))
            self.fc_var_per_sess.append(nn.Linear(n_neurons * self.embed_dim, self.latent_dim).to(self.device))
        elif self.output_type == 'mean':
            self.norm_per_sess.append(nn.LayerNorm(n_neurons))
            self.fc_mu_per_sess.append(nn.Linear(n_neurons, self.latent_dim))
            self.fc_var_per_sess.append(nn.Linear(n_neurons, self.latent_dim))
        elif 'mlp' in self.output_type:
            if self.output_type == 'mlp-tiny':
                hidden_dims = [self.embed_dim // 8, 
                               n_neurons * (self.embed_dim // 8),
                               n_neurons * self.embed_dim // 512]
            elif self.output_type == 'mlp-small':
                hidden_dims = [self.embed_dim // 64, 
                               n_neurons * (self.embed_dim // 64),
                               n_neurons * self.embed_dim // 2048]
            elif self.output_type == 'mlp-medium':
                hidden_dims = [self.embed_dim // 256,
                               n_neurons * (self.embed_dim // 256),
                               n_neurons * self.embed_dim // 4096]
            elif self.output_type == 'mlp-large':
                hidden_dims = [self.embed_dim // 256, 
                               n_neurons * (self.embed_dim // 256),
                               n_neurons * self.embed_dim // 16384]
            elif self.output_type == 'mlp-xlarge':
                hidden_dims = [self.embed_dim // 512,
                               n_neurons * (self.embed_dim // 512),
                               n_neurons * self.embed_dim // 131072]
            elif self.output_type == 'mlp-xxlarge':
                hidden_dims = [self.embed_dim // 512,
                               n_neurons * (self.embed_dim // 512), 
                               n_neurons * self.embed_dim // 524288] 

            self.proj.append(nn.Sequential(nn.Linear(self.embed_dim, hidden_dims[0]),
                                           nn.GELU(), nn.Flatten(),
                                           nn.Linear(hidden_dims[1], hidden_dims[2])))
            self.norm_per_sess.append(nn.LayerNorm(hidden_dims[2]))
            self.fc_mu_per_sess.append(nn.Linear(hidden_dims[2], self.latent_dim))
            self.fc_var_per_sess.append(nn.Linear(hidden_dims[2], self.latent_dim))

    def init_weights(self, n_neurons_per_sess: List[int]) -> None:
        """
        Initialize the weights of the neural blocks.

        Depending on the output_type of the model, this method initializes the conventional blocks,
        the normalization layers, and the fully connected layers for mean and log variance of the
        normal distribution in the latent space.

        Args:
            n_neurons_per_sess (List[int]): Number of neurons in each session.

        Raises:
            NotImplementedError: If an unknown output_type is provided.

        Returns:
            None
        """

        if self.output_type == 'flatten':
            self.proj = nn.ModuleList([nn.Flatten() for _ in n_neurons_per_sess])
            self.norm_per_sess = ModuleList([nn.LayerNorm(n_neurons * self.embed_dim)
                                             for n_neurons in n_neurons_per_sess])
            self.fc_mu_per_sess = ModuleList([nn.Linear(n_neurons * self.embed_dim, self.latent_dim)
                                              for n_neurons in n_neurons_per_sess])
            self.fc_var_per_sess = ModuleList([nn.Linear(n_neurons * self.embed_dim, self.latent_dim)
                                               for n_neurons in n_neurons_per_sess])
        elif 'mlp' in self.output_type:
            if self.output_type == 'mlp-tiny':
                hidden_dims_per_sess = np.array([[self.embed_dim // 8, 
                                                  n_neurons * (self.embed_dim // 8),
                                                  n_neurons * self.embed_dim // 512] 
                                                  for n_neurons in n_neurons_per_sess])
            elif self.output_type == 'mlp-small':
                hidden_dims_per_sess = np.array([[self.embed_dim // 64, 
                                                  n_neurons * (self.embed_dim // 64),
                                                  n_neurons * self.embed_dim // 2048]
                                                  for n_neurons in n_neurons_per_sess])
            elif self.output_type == 'mlp-medium':
                hidden_dims_per_sess = np.array([[self.embed_dim // 256,
                                                  n_neurons * (self.embed_dim // 256),
                                                  n_neurons * self.embed_dim // 4096] 
                                                  for n_neurons in n_neurons_per_sess])
            elif self.output_type == 'mlp-large':
                hidden_dims_per_sess = np.array([[self.embed_dim // 256, 
                                                  n_neurons * (self.embed_dim // 256),
                                                  n_neurons * self.embed_dim // 16384] 
                                                  for n_neurons in n_neurons_per_sess])
            elif self.output_type == 'mlp-xlarge':
                hidden_dims_per_sess = np.array([[self.embed_dim // 512, 
                                                  n_neurons * (self.embed_dim // 512),
                                                  n_neurons * self.embed_dim // 131072] 
                                                  for n_neurons in n_neurons_per_sess])
            elif self.output_type == 'mlp-xxlarge':
                hidden_dims_per_sess = np.array([[self.embed_dim // 512, 
                                                  n_neurons * (self.embed_dim // 512),
                                                  n_neurons * self.embed_dim // 524288] 
                                                  for n_neurons in n_neurons_per_sess])

            self.proj = ModuleList([nn.Sequential(nn.Linear(self.embed_dim, hidden_dims[0]),
                                                  nn.GELU(), nn.Flatten(),
                                                  nn.Linear(hidden_dims[1], hidden_dims[2]))
                                    for hidden_dims in hidden_dims_per_sess])

            self.norm_per_sess = ModuleList([nn.LayerNorm(hidden_dims[2])
                                    for hidden_dims in hidden_dims_per_sess])
            self.fc_mu_per_sess = ModuleList([nn.Linear(hidden_dims[2], self.latent_dim)
                                              for hidden_dims in hidden_dims_per_sess])
            self.fc_var_per_sess = ModuleList([nn.Linear(hidden_dims[2], self.latent_dim)
                                               for hidden_dims in hidden_dims_per_sess])
        else:
            raise NotImplementedError

    def detach_(self):
        """
        Detach the attention layer of each Hebbian attention block from the computational graph.

        No Args.

        No Returns.
        """

        for block in self.hebbian_attn_blocks:
            block.detach_()

    def zero_(self):
        """
        Resets the values of the attention layer for each Hebbian attention block.

        No Args.

        No Returns.
        """

        for block in self.hebbian_attn_blocks:
            block.zero_()
