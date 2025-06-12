import argparse
import os

import numpy as np
import torch

from sparks.data.allen.movies_pseudomouse import make_pseudomouse_allen_movies_dataset
from sparks.models.decoders import get_decoder
from sparks.models.encoders import HebbianTransformerEncoder
from sparks.utils.test import test
import json

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=9, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Whether to use the online gradient descent algorithm')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=64, help='Size of the latent space')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=256, help='Size of attention embeddings')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--dec_type', type=str, default='mlp',
                        help='Type of decoder (one of linear, mlp or deconv)')
    parser.add_argument('--output_type', type=str, default='flatten',
                          help='Output architecture for the decoder')
    parser.add_argument('--tau_p', type=int, default=6, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, default=0.5, help='STDP decay')
    parser.add_argument('--alpha', type=float, default=1., help='')

    # Data parameters
    parser.add_argument('--mode', type=str, default='prediction',
                        choices=['prediction', 'reconstruction', 'unsupervised'],
                        help='Which type of task to perform')
    parser.add_argument('--data_type', type=str, default='ephys', choices=['ephys', 'calcium'],
                        help='Whether to use neuropixels or calcium data')
    parser.add_argument('--n_neurons', type=int, default=50)
    parser.add_argument('--dt', type=float, default=0.006, help='Time sampling period')
    parser.add_argument('--ds', type=int, default=2, help='Frame downsampling factor')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')

    # sliding
    parser.add_argument('--block_size', type=int, default=100, help='Dimension of the sliding attention blocks')
    parser.add_argument('--window_size', type=int, default=3, help='Size of the sliding window')
    parser.add_argument('--sliding', action='store_true', default=False, help='')

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

    neuron_types = ['VISp']
    correct_units_ids = np.load(os.path.join(args.weights_folder, 'good_units_ids.npy'))
    dataset, _, dl, _ = make_pseudomouse_allen_movies_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                           neuron_types=neuron_types,
                                                           n_neurons=args.n_neurons,
                                                           dt=args.dt,
                                                           block="all",
                                                           batch_size=args.batch_size,
                                                           num_workers=args.num_workers,
                                                           mode=args.mode,
                                                           ds=args.ds,
                                                           correct_units_ids=correct_units_ids)
    
    encoding_network = HebbianTransformerEncoder(n_neurons_per_sess=len(dataset.good_units_ids),
                                                 embed_dim=args.embed_dim,
                                                 latent_dim=args.latent_dim,
                                                 tau_s_per_sess=args.tau_s,
                                                 dt_per_sess=args.dt,
                                                 n_layers=args.n_layers,
                                                 n_heads=args.n_heads,
                                                 output_type=args.output_type,
                                                 sliding=args.sliding,
                                                 window_size=args.window_size,
                                                 block_size=args.block_size,
                                                 alpha=args.alpha).to(args.device)

    if args.mode == 'prediction':
        output_size = 900
    elif args.mode == 'reconstruction':
        output_size = np.prod(dataset.true_frames.shape[:-1])
    elif args.mode == 'unsupervised':
        output_size = args.n_neurons
    else:
        raise NotImplementedError

    decoding_network = get_decoder(output_dim_per_session=output_size * args.tau_f, args=args,
                                   n_neurons=args.n_neurons, softmax=True if args.mode == 'prediction' else False)

    # Load pretrained network and add neural attention layers for additional sessions
    encoding_network.load_state_dict(torch.load(os.path.join(args.weights_folder, 'encoding_network.pt')))
    decoding_network.load_state_dict(torch.load(os.path.join(args.weights_folder, 'decoding_network.pt')))

    if args.mode == 'prediction':
        loss_fn = torch.nn.NLLLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    best_test_acc = -np.inf

    test_acc, encoder_outputs, decoder_outputs = test(encoder=encoding_network,
                                                        decoder=decoding_network,
                                                        test_dls=[dl],
                                                        true_frames=dataset.true_frames,
                                                        mode=args.mode,
                                                        latent_dim=args.latent_dim,
                                                        tau_p=args.tau_p,
                                                        tau_f=args.tau_f,
                                                        dt=args.dt,
                                                        loss_fn=loss_fn,
                                                        device=args.device)

    np.save(args.weights_folder + '/test_dec_outputs_all.npy', decoder_outputs.cpu().numpy())
    np.save(args.weights_folder + '/test_enc_outputs_all.npy', encoder_outputs.cpu().numpy())

    if args.mode == 'prediction':
        print("test acc: %.3f" % test_acc)
    else:
        print("=test loss: %.3f" % -test_acc)
