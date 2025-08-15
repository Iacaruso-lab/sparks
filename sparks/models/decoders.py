from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class mlp(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dims: int,
                 output_dim_per_session: Any,
                 id_per_sess: np.ndarray = np.array([0])) -> None:

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

        Returns:
            None
        """

        super(mlp, self).__init__()

        layers = []

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.out_layers = nn.ModuleDict({str(sess_id): nn.Linear(in_dim, output_dim)
                                         for sess_id, output_dim in zip(id_per_sess, output_dim_per_session)})

        self.layers = nn.Sequential(*layers)

    def forward(self, x, sess_id: int = 0) -> torch.Tensor:
        sess_id = str(sess_id)

        x = self.layers(x.flatten(1))

        return self.out_layers[sess_id](x)


class linear(nn.Module):
    def __init__(self, in_dim: int,
                 output_dim_per_session: Any,
                 id_per_sess: Any = None) -> None:

        """
        Initialize a  fully connected, or dense, layer, with either no activation function or log_softmax.

        If the model is trained to encode more than one session via unsupervised learning, the decoder is given
        one output layer for each to accommodate the varying output dimensions.

        Args:
            in_dim (int): The input dimension of the MLP.
            output_dim_per_session (Union[int, List[int]]): The output dimension of the MLP.
            id_per_sess (Optional[np.array]): Defaults to None, the ids of the sessions corresponding to the
                                                various output layers.
            softmax (bool): Defaults to False. If True, apply a log_softmax activation function to the neuron outputs.

        Returns:
            None
        """

        super(linear, self).__init__()

        self.out_layers = nn.ModuleDict({str(sess_id): nn.Linear(in_dim, output_dim)
                                         for sess_id, output_dim in zip(id_per_sess, output_dim_per_session)})

    def forward(self, x, sess_id: int = 0) -> torch.Tensor:
        sess_id = str(sess_id)

        return self.out_layers[sess_id](x.flatten(1))
