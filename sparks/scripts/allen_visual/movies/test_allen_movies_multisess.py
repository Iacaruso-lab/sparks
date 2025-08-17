import argparse
import os

import numpy as np
import torch
import json

from sparks.data.allen.movies_multisess import make_allen_movies_dataset
from sparks.scripts.allen_visual.movies.utils.misc import make_network_and_optimizers
from sparks.scripts.allen_visual.movies.utils.test import test

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Whether to use the online gradient descent algorithm')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=64, help='Size of the latent space')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=128, help='Size of attention embeddings')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--tau_p', type=int, default=6, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, default=1., help='STDP decay')

    # Data parameters
    parser.add_argument('--n_skip_sessions', type=int, default=0, help='First session to consider')
    parser.add_argument('--n_sessions', type=int, default=1, help='How many sessions to use')
    parser.add_argument('--mode', type=str, default='prediction',
                        choices=['prediction', 'reconstruction', 'unsupervised'],
                        help='Which type of task to perform')
    parser.add_argument('--dt', type=float, default=0.006, help='Time sampling period')
    parser.add_argument('--ds', type=int, default=2, help='Frame downsampling factor')

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

    # Create datasets and dataloaders
    (_, test_datasets, _, test_dls) = make_allen_movies_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                                session_idxs=np.arange(args.n_skip_sessions,
                                                                                       args.n_sessions + args.n_skip_sessions),
                                                                                       dt=args.dt,
                                                                                       block='all',
                                                                                       batch_size=args.batch_size,
                                                                                       num_workers=args.num_workers,
                                                                                       mode=args.mode,
                                                                                       ds=args.ds)

    sparks, optimizer, scheduler, loss_fn = make_network_and_optimizers(args, datasets=test_datasets)

    # Load pretrained network and add neural attention layers for additional sessions
    sparks.load_state_dict(torch.load(os.path.join(args.weights_folder, 'sparks.pt')))

    best_test_acc = -np.inf

    test_acc, encoder_outputs, decoder_outputs = test(sparks=sparks,
                                                        test_dls=test_dls,
                                                        loss_fn=loss_fn,
                                                        mode=args.mode,
                                                        frames=test_datasets[0].true_frames,
                                                        dt=args.dt,
                                                        act=torch.sigmoid,
                                                        sess_ids=np.arange(args.n_sessions) + args.n_skip_sessions)
    
    np.save(args.weights_folder + '/test_dec_outputs_all.npy', decoder_outputs.cpu().numpy())
    np.save(args.weights_folder + '/test_enc_outputs_all.npy', encoder_outputs.cpu().numpy())

    if args.mode == 'prediction':
        print("test acc: %.3f" % test_acc)
    else:
        print("=test loss: %.3f" % -test_acc)
