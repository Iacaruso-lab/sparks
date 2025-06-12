import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.allen.gratings_pseudomouse import make_gratings_dataset
from sparks.models.decoders import get_decoder
from sparks.models.encoders import HebbianTransformerEncoder
from sparks.utils.test import test_on_batch
import json

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.01, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Whether to use the online gradient descent algorithm')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=64, help='Size of the latent space')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=128, help='Size of attention embeddings')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--dec_type', type=str, default='mlp', choices=['linear', 'mlp'],
                        help='Type of decoder (one of linear or mlp)')
    parser.add_argument('--tau_p', type=int, default=10, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, default=0.5, help='STDP decay')
    parser.add_argument('--alpha', type=float, default=1., help='')

    # data
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--target_type', type=str, default='freq', choices=['freq', 'class', 'unsupervised'],
                    help='Type of target to predict: either spatial frequencies or class index')
    parser.add_argument('--p_train', type=float, default=0.8, help='Number of training example')
    parser.add_argument('--dt', type=float, default=0.001, help='time bins period')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    parser.add_argument('--save_period', type=int, default=50, help='Test period in number of time-steps')

    parser.add_argument('--weights_folder', type=str, default='')
    

    args = parser.parse_args()

    with open(os.path.join(args.weights_folder, 'commandline_args.txt'), 'r') as f:
        data_string = f.read()
    data_dict = json.loads(data_string)

    for arg in data_dict.keys():
        setattr(args, arg, data_dict[arg])

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps:0')
    else:
        args.device = torch.device('cpu')

    neuron_types = ['VISp', 'VISal', 'VISrl', 'VISpm', 'VISam', 'VISl']
    args.neuron_types = neuron_types

    correct_units_ids = np.load(os.path.join(args.weights_folder, 'good_units_ids.npy'))
    (_, train_dl, test_dataset, test_dl) = make_gratings_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                                 n_neurons=args.n_neurons,
                                                                 dt=args.dt,
                                                                 neuron_types=neuron_types,
                                                                 p_train=args.p_train,
                                                                 batch_size=args.batch_size,
                                                                 num_workers=args.num_workers,
                                                                 target_type=args.target_type,
                                                                 seed=args.seed,
                                                                 correct_units_ids=correct_units_ids)

    encoding_network = HebbianTransformerEncoder(n_neurons_per_sess=args.n_neurons * len(neuron_types),
                                                 embed_dim=args.embed_dim,
                                                 latent_dim=args.latent_dim,
                                                 tau_s_per_sess=args.tau_s,
                                                 dt_per_sess=args.dt,
                                                 n_layers=args.n_layers,
                                                 n_heads=args.n_heads,
                                                 alpha=args.alpha).to(args.device)

    if args.target_type == 'freq':
        output_size = 2
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.target_type == 'class':
        output_size = 5
        loss_fn = torch.nn.NLLLoss()
    elif args.target_type == 'unsupervised':
        output_size = args.n_neurons * len(neuron_types)
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError

    decoding_network = get_decoder(output_dim_per_session=output_size * args.tau_f, args=args)
    # Load pretrained network and add neural attention layers for additional sessions
    encoding_network.load_state_dict(torch.load(os.path.join(args.weights_folder, 'encoding_network.pt')))
    decoding_network.load_state_dict(torch.load(os.path.join(args.weights_folder, 'decoding_network.pt')))

    attention_coeffs = torch.Tensor()

    with torch.no_grad():
        for i in range(len(neuron_types)):
            test_iterator = iter(test_dl)

            encoder_outputs = torch.Tensor()
            decoder_outputs = torch.Tensor()

            for inputs, targets in test_iterator:
                targets = targets.unsqueeze(2).repeat_interleave(inputs.shape[-1], dim=-1)
                inputs[:, i * args.n_neurons:(i + 1) * args.n_neurons] = 0
                print(inputs.shape)
                print(torch.sum(inputs, dim=(0, -1)))
                _, encoder_outputs_batch, decoder_outputs_batch = test_on_batch(encoder=encoding_network,
                                                                                        decoder=decoding_network,
                                                                                        inputs=inputs,
                                                                                        targets=targets,
                                                                                        latent_dim=args.latent_dim,
                                                                                        tau_p=args.tau_p,
                                                                                        tau_f=args.tau_f,
                                                                                        loss_fn=loss_fn,
                                                                                        device=args.device,
                                                                                        act=torch.sigmoid)

                encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch), dim=0)
                decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch), dim=0)

            np.save(os.path.join(args.weights_folder, 'decoder_outputs_without_%s.npy' % neuron_types[i]),
                    decoder_outputs.cpu().numpy())