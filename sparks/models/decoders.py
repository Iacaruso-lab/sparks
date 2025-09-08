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
                 id_per_session: np.ndarray = np.array([0]),
                 joint_decoder: bool = False) -> None:

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
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        if joint_decoder:
            self.out_layers = nn.Linear(in_dim, int(output_dim_per_session))
        else:
            self.out_layers = nn.ModuleDict({str(sess_id): nn.Linear(in_dim, output_dim)
                                            for sess_id, output_dim in zip(id_per_session, output_dim_per_session)})

        self.layers = nn.Sequential(*layers)

    def forward(self, x, sess_id: int = 0) -> torch.Tensor:
        sess_id = str(sess_id)

        x = self.layers(x.flatten(1))

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
            return self.out_layers(x.flatten(1))
        else:
            return self.out_layers[sess_id](x.flatten(1))
