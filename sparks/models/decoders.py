from typing import Any, List, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from sparks.models.blocks import AttentionBlock

class mlp(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dims: int,
                 output_dim_per_session: Any,
                 id_per_session: np.ndarray = np.array([0]),
                 joint_decoder: bool = False,
                 dropout: float = 0.0) -> None:

        """
        Initialize a Multi-Layer Perceptron (MLP).

        This MLP consists of an input layer, zero or more hidden layers, and an output layer.
        Each layer is a fully connected, or dense, layer, meaning each neuron in one layer is connected to all neurons
        in the previous layer. The last layer has either no activation function or log_softmax.

        If the model is trained to encode more than one session via unsupervised learning, the decoder is given
        one output layer for each to accommodate the varying output dimensions.

        Args:
            in_dim (int): The input dimension of the MLP.
            hidden_dims (Union[int, List[int]]): The number of hidden neurons in the hidden layers.
            output_dim_per_session (Union[int, List[int]]): The output dimension of the MLP.
            id_per_sess: Defaults to None, the ids of the sessions corresponding to the
                                                various output layers.
            joint_decoder (bool): If True, uses a single output layer for all sessions. Default is False.

        Returns:
            None
        """

        super(mlp, self).__init__()

        self.joint_decoder = joint_decoder

        layers = []

        for h_dim in hidden_dims:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim

        if joint_decoder:
            self.out_layers = nn.Linear(in_dim, int(output_dim_per_session))
        else:
            self.out_layers = nn.ModuleDict({str(sess_id): nn.Linear(in_dim, output_dim)
                                            for sess_id, output_dim in zip(id_per_session, output_dim_per_session)})

        self.layers = nn.Sequential(*layers)

    def forward(self, x, sess_id: int = 0) -> torch.Tensor:
        sess_id = str(sess_id)

        x = self.layers(x)

        if self.joint_decoder:
            return self.out_layers(x)
        else:
            return self.out_layers[sess_id](x)
        


class linear_mlp(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dims: int,
                 output_dim_per_session: Any,
                 id_per_session: np.ndarray = np.array([0]),
                 joint_decoder: bool = False) -> None:

        """
        Initialize a Multi-Layer Perceptron (MLP) with linear activations.

        This MLP consists of an input layer, zero or more hidden layers, and an output layer.
        Each layer is a fully connected, or dense, layer, meaning each neuron in one layer is connected to all neurons
        in the previous layer. The last layer has either no activation function or log_softmax.

        If the model is trained to encode more than one session via unsupervised learning, the decoder is given
        one output layer for each to accommodate the varying output dimensions.

        Args:
            in_dim (int): The input dimension of the MLP.
            hidden_dims (Union[int, List[int]]): The number of hidden neurons in the hidden layers.
            output_dim_per_session (Union[int, List[int]]): The output dimension of the MLP.
            id_per_sess: Defaults to None, the ids of the sessions corresponding to the
                                                various output layers.
            joint_decoder (bool): If True, uses a single output layer for all sessions. Default is False.

        Returns:
            None
        """

        super(linear_mlp, self).__init__()

        self.joint_decoder = joint_decoder

        layers = []

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        if joint_decoder:
            self.out_layers = nn.Linear(in_dim, int(output_dim_per_session))
        else:
            self.out_layers = nn.ModuleDict({str(sess_id): nn.Linear(in_dim, output_dim)
                                            for sess_id, output_dim in zip(id_per_session, output_dim_per_session)})

        self.layers = nn.Sequential(*layers)

    def forward(self, x, sess_id: int = 0) -> torch.Tensor:
        sess_id = str(sess_id)

        x = self.layers(x)

        if self.joint_decoder:
            return self.out_layers(x)
        else:
            return self.out_layers[sess_id](x)


class linear(nn.Module):
    def __init__(self, in_dim: int,
                 output_dim_per_session: Any,
                 id_per_session: Any = None,
                 joint_decoder: bool = False) -> None:

        """
        Initialize a  fully connected, or dense, layer, with either no activation function or log_softmax.

        If the model is trained to encode more than one session via unsupervised learning, the decoder is given
        one output layer for each to accommodate the varying output dimensions.

        Args:
            in_dim (int): The input dimension of the MLP.
            output_dim_per_session (Union[int, List[int]]): The output dimension of the MLP.
            id_per_sess (Optional[np.array]): Defaults to None, the ids of the sessions corresponding to the
                                                various output layers.
            joint_decoder (bool): If True, uses a single output layer for all sessions. Default is False.

        Returns:
            None
        """

        super(linear, self).__init__()

        self.joint_decoder = joint_decoder

        if joint_decoder:
            self.out_layers = nn.Linear(in_dim, int(output_dim_per_session))
        else:
            self.out_layers = nn.ModuleDict({str(sess_id): nn.Linear(in_dim, output_dim)
                                            for sess_id, output_dim in zip(id_per_session, output_dim_per_session)})

    def forward(self, x, sess_id: int = 0) -> torch.Tensor:
        sess_id = str(sess_id)

        if self.joint_decoder:
            return self.out_layers(x)
        else:
            return self.out_layers[sess_id](x)
        

class transformer(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 output_dim_per_session: Any,
                 n_layers: int = 1,
                 n_heads: int = 1,
                 dropout: float = 0.,
                 id_per_session: Any = None,
                 joint_decoder: bool = False,
                 zero_init: bool = False):
        super().__init__()

        self.attention_layers = nn.ModuleList([AttentionBlock(d_model=embed_dim, 
                                                              n_heads=n_heads, 
                                                              dropout=dropout) for _ in range(n_layers)])
        self.joint_decoder = joint_decoder

        if joint_decoder:
            self.out_layers = nn.Linear(embed_dim, int(output_dim_per_session))
        else:
            self.out_layers = nn.ModuleDict({str(sess_id): nn.Linear(embed_dim, output_dim)
                                            for sess_id, output_dim in zip(id_per_session, output_dim_per_session)})
        
        if zero_init:
            if self.joint_decoder:
                nn.init.zeros_(self.out_layers.weight)
                nn.init.zeros_(self.out_layers.bias)
            else:
                for sess_id in self.out_layers.keys():
                    nn.init.zeros_(self.out_layers[sess_id].weight)
                    nn.init.zeros_(self.out_layers[sess_id].bias)

    def forward(self, x, sess_id: int = 0, **kwargs) -> torch.Tensor:
        """
        Forward pass through the full transformer decoder block.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch, seq_len, n_neurons).
        Returns:
            torch.Tensor: Output tensor after passing through the attention layer, feedforward network, and perceiver block.
        """

        for attn_layer in self.attention_layers:
            x = attn_layer(x)

        if self.joint_decoder:
            return self.out_layers(x)
        else:
            return self.out_layers[str(sess_id)](x)