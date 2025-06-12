import argparse
import os

import numpy as np
import torch
import json

from sparks.data.allen.gratings_pseudomouse_singlesess import make_gratings_dataset
from sparks.models.decoders import get_decoder
from sparks.models.encoders import HebbianTransformerEncoder
from sparks.utils.test import test_init
from sparks.utils.vae import skip, ae_forward
from sparks.utils.misc import LongCycler

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
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

    # data
    parser.add_argument('--n_skip_sessions', type=int, default=0, help='First session to consider')
    parser.add_argument('--n_sessions', type=int, default=1, help='How many sessions to use')
    parser.add_argument('--target_type', type=str, default='freq', choices=['freq', 'class', 'unsupervised'],
                        help='Type of target to predict: either spatial frequencies or class index')
    parser.add_argument('--p_train', type=float, default=0.8, help='Number of training example')
    parser.add_argument('--dt', type=float, default=0.001, help='time bins period')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')

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
    (train_datasets, train_dls, 
     test_datasets, test_dls) = make_gratings_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                      session_idxs=np.arange(args.n_skip_sessions,
                                                      args.n_sessions + args.n_skip_sessions),
                                                      dt=args.dt,
                                                      neuron_types=neuron_types,
                                                      p_train=args.p_train,
                                                      num_workers=args.num_workers,
                                                      batch_size=args.batch_size,
                                                      target_type=args.target_type,
                                                      seed=args.seed)

    input_sizes = [len(train_dataset.good_units_ids) for train_dataset in train_datasets]
    encoding_network = HebbianTransformerEncoder(n_neurons_per_sess=input_sizes,
                                                 embed_dim=args.embed_dim,
                                                 latent_dim=args.latent_dim,
                                                 tau_s_per_sess=args.tau_s,
                                                 dt_per_sess=args.dt,
                                                 n_layers=args.n_layers,
                                                 n_heads=args.n_heads).to(args.device)
    print(f"Input sizes per session: {input_sizes}")
    if args.target_type == 'freq':
        output_size = 2
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.target_type == 'class':
        output_size = 5
        loss_fn = torch.nn.NLLLoss()
    elif args.target_type == 'unsupervised':
        output_size = [input_size * args.tau_f for input_size in input_sizes]
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError

    decoding_network = get_decoder(output_dim_per_session=output_size * args.tau_f, args=args)

    if args.online:
        args.lr = args.lr / (0.25 * args.dt)
    optimizer = torch.optim.Adam(list(encoding_network.parameters())
                                 + list(decoding_network.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.9)

    loss_best = np.inf

    # Load pretrained network and add neural attention layers for additional sessions
    encoding_network.load_state_dict(torch.load(os.path.join(args.weights_folder, 'encoding_network.pt')))
    decoding_network.load_state_dict(torch.load(os.path.join(args.weights_folder, 'decoding_network.pt')))

    with torch.no_grad():
        attention_coeffs = torch.Tensor()
        train_iterator = LongCycler(train_dls)
        for inputs, targets in train_iterator:
            targets = targets.unsqueeze(2).repeat_interleave(inputs.shape[-1], dim=-1)
            encoder_outputs_batch, _, T = test_init(encoder=encoding_network,
                                                    inputs=inputs,
                                                    latent_dim=args.latent_dim,
                                                    tau_p=args.tau_p,
                                                    device=args.device)

            for t in range(T):
                encoder_outputs_batch, _, _, _ = ae_forward(encoder=encoding_network,
                                                            decoder=decoding_network,
                                                            inputs=inputs[..., t],
                                                            encoder_outputs=encoder_outputs_batch,
                                                            tau_p=args.tau_p,
                                                            device=args.device)

            attention_coeffs = torch.cat((attention_coeffs,
                                        encoding_network.hebbian_attn_blocks[0].attention_layer.heads[0].attention.data.cpu()), dim=0)
        np.save(os.path.join(args.weights_folder, 'train_attention_coeffs.npy'), attention_coeffs.numpy())

        attention_coeffs = torch.Tensor()
        test_iterator = LongCycler(test_dls)
        for inputs, targets in test_iterator:
            targets = targets.unsqueeze(2).repeat_interleave(inputs.shape[-1], dim=-1)
            encoder_outputs_batch, _, T = test_init(encoder=encoding_network,
                                                    inputs=inputs,
                                                    latent_dim=args.latent_dim,
                                                    tau_p=args.tau_p,
                                                    device=args.device)

            for t in range(T):
                encoder_outputs_batch, _, _, _ = ae_forward(encoder=encoding_network,
                                                            decoder=decoding_network,
                                                            inputs=inputs[..., t],
                                                            encoder_outputs=encoder_outputs_batch,
                                                            tau_p=args.tau_p,
                                                            device=args.device)

            attention_coeffs = torch.cat((attention_coeffs,
                                        encoding_network.hebbian_attn_blocks[0].attention_layer.heads[0].attention.data.cpu()), dim=0)

        np.save(os.path.join(args.weights_folder, 'test_attention_coeffs.npy'), attention_coeffs.numpy())
