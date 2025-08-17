import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.allen.movies_pseudomouse import make_pseudomouse_allen_movies_dataset
from sparks.scripts.allen_visual.movies.utils.misc import make_network_and_optimizers
from sparks.scripts.allen_visual.movies.utils.test import test
from sparks.scripts.allen_visual.movies.utils.train import train
from sparks.utils.misc import make_res_folder, save_results


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.000001, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=9, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Whether to use the online gradient descent algorithm')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=64, help='Size of the latent space')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=256, help='Size of attention embeddings')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--tau_p', type=int, default=6, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, default=1., help='STDP decay')

    # Data parameters
    parser.add_argument('--block', type=str, default='first',
                        choices=['first', 'second', 'across', 'both'], help='From which blocks to use')
    parser.add_argument('--mode', type=str, default='prediction',
                        choices=['prediction', 'reconstruction', 'unsupervised'],
                        help='Which type of task to perform')
    parser.add_argument('--data_type', type=str, default='ephys', choices=['ephys', 'calcium'],
                        help='Whether to use neuropixels or calcium data')
    parser.add_argument('--n_neurons', type=int, default=50)
    parser.add_argument('--dt', type=float, default=0.006, help='Time sampling period')
    parser.add_argument('--ds', type=int, default=2, help='Frame downsampling factor')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')

    # sliding-window attention parameters
    parser.add_argument('--block_size', type=int, default=100, help='Dimension of the sliding attention blocks')
    parser.add_argument('--window_size', type=int, default=3, help='Size of the sliding window')
    parser.add_argument('--sliding', action='store_true', default=False, help='')

    args = parser.parse_args()

    # Create folder to save results
    make_res_folder('allen_movies_pseudomouse_' + args.mode + '_nneurons_' + str(args.n_neurons), os.getcwd(), args)

    neuron_types = ['VISp']
    (train_dataset, test_dataset,
     train_dl, test_dl) = make_pseudomouse_allen_movies_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                                neuron_types=neuron_types,
                                                                n_neurons=args.n_neurons,
                                                                dt=args.dt,
                                                                block=args.block,
                                                                batch_size=args.batch_size,
                                                                num_workers=args.num_workers,
                                                                mode=args.mode,
                                                                ds=args.ds,
                                                                seed=args.seed)
    np.save(args.results_path + '/good_units_ids.npy', train_dataset.good_units_ids)

    sparks, optimizer, scheduler, loss_fn = make_network_and_optimizers(args, datasets=[train_dataset])

    best_test_acc = -np.inf

    # Training loop
    pbar = tqdm.tqdm(range(args.n_epochs))
    for epoch in pbar:
        train(sparks=sparks,
              train_dls=[train_dl],
              loss_fn=loss_fn,
              optimizer=optimizer,
              beta=args.beta,
              device=args.device,
              mode=args.mode,
              online=args.online,
              frames=train_dataset.true_frames,
              dt=args.dt)
        scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            test_acc, encoder_outputs, decoder_outputs = test(sparks=sparks,
                                                               test_dls=[test_dl],
                                                               loss_fn=loss_fn,
                                                               mode=args.mode,
                                                               frames=test_dataset.true_frames,
                                                               dt=args.dt,
                                                               act=torch.sigmoid)

            best_test_acc = save_results(args.results_path, test_acc, best_test_acc,
                                          encoder_outputs, decoder_outputs, sparks)

            if args.mode == 'prediction':
                pbar.set_description("Epoch %d, test acc: %.3f" % (epoch, test_acc))
            else:
                pbar.set_description("Epoch %d, test loss: %.3f" % (epoch, -test_acc))
