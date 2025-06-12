from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




def get_decoder(output_dim_per_session: Any,
                args: Any,
                id_per_sess: Any = None,
                softmax: bool = False,
                **kwargs):

    if not hasattr(output_dim_per_session, '__iter__'):
        output_dim_per_session = [output_dim_per_session]
    if id_per_sess is None:
        id_per_sess = np.arange(len(output_dim_per_session))

    n_inputs_decoder = args.latent_dim * args.tau_p
    hid_features = kwargs.get('hid_features', [int(np.mean([n_inputs_decoder, np.mean(output_dim_per_session)]))])
    dropout = kwargs.get('dropout', 0.)

    if args.dec_type == 'linear':
        return linear(in_dim=n_inputs_decoder,
                      output_dim_per_session=output_dim_per_session,
                      id_per_sess=id_per_sess,
                      softmax=softmax).to(args.device)
    elif args.dec_type == 'mlp-tiny':
        return mlp(in_dim=n_inputs_decoder,
                   hid_features=[np.max(output_dim_per_session) // 32, np.max(output_dim_per_session) // 16],
                   output_dim_per_session=output_dim_per_session,
                   id_per_sess=id_per_sess,
                   dropout=dropout,
                   softmax=softmax).to(args.device)
    elif args.dec_type == 'mlp-small':
        return mlp(in_dim=n_inputs_decoder,
                   hid_features=[np.max(output_dim_per_session) // 64, np.max(output_dim_per_session) // 32],
                   output_dim_per_session=output_dim_per_session,
                   id_per_sess=id_per_sess,
                   dropout=dropout,
                   softmax=softmax).to(args.device)
    elif args.dec_type == 'mlp-medium':
        return mlp(in_dim=n_inputs_decoder,
                   hid_features=[np.max(output_dim_per_session) // 128, np.max(output_dim_per_session) // 64],
                   output_dim_per_session=output_dim_per_session,
                   id_per_sess=id_per_sess,
                   dropout=dropout,
                   softmax=softmax).to(args.device)
    elif args.dec_type == 'mlp-large':
        return mlp(in_dim=n_inputs_decoder,
                   hid_features=[np.max(output_dim_per_session) // 256, np.max(output_dim_per_session) // 128],
                   output_dim_per_session=output_dim_per_session,
                   id_per_sess=id_per_sess,
                   dropout=dropout,
                   softmax=softmax).to(args.device)
    elif args.dec_type == 'mlp-xlarge':
        return mlp(in_dim=n_inputs_decoder,
                   hid_features=[np.max(output_dim_per_session) // 512, 
                                 np.max(output_dim_per_session) // 256],
                   output_dim_per_session=output_dim_per_session,
                   id_per_sess=id_per_sess,
                   dropout=dropout,
                   softmax=softmax).to(args.device)
    elif args.dec_type == 'mlp':
        return mlp(in_dim=n_inputs_decoder,
                   hid_features=hid_features,
                   output_dim_per_session=output_dim_per_session,
                   id_per_sess=id_per_sess,
                   dropout=dropout,
                   softmax=softmax).to(args.device)
    elif args.dec_type == 'rnn':
        return rnn(input_size=n_inputs_decoder,
                   hidden_size=hid_features,
                   output_size=output_dim_per_session[0]).to(args.device)
    else:
        raise ValueError("dec_type must be one of: linear, mlp, rnn")


class mlp(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_features: int,
                 output_dim_per_session: Any,
                 id_per_sess: Any = None,
                 dropout: float = 0.,
                 softmax: bool = False) -> None:

        """
        Initialize a Multi-Layer Perceptron (MLP).

        This MLP consists of an input layer, zero or more hidden layers, and an output layer.
        Each layer is a fully connected, or dense, layer, meaning each neuron in one layer is connected to all neurons
        in the previous layer. The last layer has either no activation function or log_softmax.

        If the model is trained to encode more than one session via unsupervised learning, the decoder is given
        one output layer for each to accommodate the varying output dimensions.

        Args:
            in_dim (int): The input dimension of the MLP.
            hid_features (Union[int, List[int]]): The number of hidden neurons in the hidden layers.
            output_dim_per_session (Union[int, List[int]]): The output dimension of the MLP.
            id_per_sess (Optional[np.array]): Defaults to None, the ids of the sessions corresponding to the
                                                various output layers.
            softmax (bool): Defaults to False. If True, apply a log_softmax activation function to the neuron outputs.

        Returns:
            None
        """

        super(mlp, self).__init__()

        if not hasattr(output_dim_per_session, '__iter__'):
            output_dim_per_session = [output_dim_per_session]
        if not hasattr(hid_features, '__iter__'):
            hid_features = [hid_features]
        if id_per_sess is None:
            id_per_sess = np.arange(len(output_dim_per_session))
        self.id_per_sess = id_per_sess

        self.layers = nn.ModuleList()
        self.in_layer = nn.Sequential(nn.Linear(in_dim, hid_features[0]),
                                      nn.GELU())

        for i in range(len(hid_features) - 1):
            self.layers.append(nn.Sequential(nn.Linear(hid_features[i], hid_features[i+1]),
                                             nn.GELU()))

        self.out = nn.ModuleList([nn.Linear(hid_features[-1], out_features)
                                        for out_features in output_dim_per_session])
        self.softmax = softmax

    def forward(self, x, sess_id=0) -> torch.Tensor:
        if len(self.id_per_sess) > 1:
            out_layer_idx = np.where(self.id_per_sess == sess_id)[0][0]
        else:
            out_layer_idx = 0

        # x = nn.functional.relu(self.in_layer(x.flatten(1)))
        x = self.in_layer(x.flatten(1))
        for layer in self.layers:
            # x = nn.functional.relu(layer(x))
            x = layer(x)

        if self.softmax:
            return torch.log_softmax(self.out[out_layer_idx](x), dim=-1)
        else:
            return self.out[out_layer_idx](x)


class linear(nn.Module):
    def __init__(self, in_dim: int,
                 output_dim_per_session: Any,
                 id_per_sess: Any = None,
                 softmax: bool = False) -> None:

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

        if not hasattr(output_dim_per_session, '__iter__'):
            output_dim_per_session = [output_dim_per_session]
        if id_per_sess is None:
            id_per_sess = np.arange(len(output_dim_per_session))
        self.id_per_sess = id_per_sess

        self.fc1 = nn.ModuleList([nn.Linear(in_dim, out_features)
                                        for out_features in output_dim_per_session])
        self.softmax = softmax

    def forward(self, z, sess_id=0) -> torch.Tensor:
        out_layer_idx = np.where(self.id_per_sess == sess_id)[0][0]
        z = self.fc1[out_layer_idx](z.flatten(1))

        if self.softmax:
            return torch.log_softmax(z, dim=-1)
        else:
            return z


class rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rnn, self).__init__()

        self.hidden_size = hidden_size[0]
        self.hidden = None

        self.i2h = nn.Sequential(nn.Linear(input_size, self.hidden_size),
                                 nn.GELU())
        self.h2h = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.GELU())
        self.h2o = nn.Linear(self.hidden_size, output_size)

    def forward(self, x_t, sess_id=None):
        if self.hidden is None:
            self.hidden = torch.zeros([len(x_t), self.hidden_size]).to(x_t.device)
        
        self.hidden = self.i2h(x_t.flatten(1)) + self.h2h(self.hidden)
        output = self.h2o(self.hidden)
        return output

    def init(self):
        self.hidden = None
