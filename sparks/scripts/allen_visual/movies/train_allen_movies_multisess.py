import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.allen.movies_multisess import make_allen_movies_dataset
from sparks.scripts.allen_visual.movies.utils.test import test
from sparks.scripts.allen_visual.movies.utils.train import train
from sparks.scripts.allen_visual.movies.utils.misc import make_network_and_optimizers
from sparks.utils.misc import make_res_folder, save_results

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--beta', type=float, default=0.000001, help='KLD regularisation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
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
    parser.add_argument('--block', type=str, default='first',
                        choices=['first', 'second', 'across', 'both'], help='From which blocks to use')
    parser.add_argument('--mode', type=str, default='prediction',
                        choices=['prediction', 'reconstruction', 'unsupervised', 'denoising'],
                        help='Which type of task to perform')
    parser.add_argument('--dt', type=float, default=0.006, help='Time sampling period')
    parser.add_argument('--ds', type=int, default=2, help='Frame downsampling factor')

    args = parser.parse_args()

    # Create folder to save results
    make_res_folder('allen_movies_multisess', os.getcwd(), args)

    # Create datasets and dataloaders
    (train_datasets, test_datasets,
     train_dls, test_dls) = make_allen_movies_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                      session_idxs=np.arange(args.n_skip_sessions,
                                                                             args.n_sessions + args.n_skip_sessions),
                                                      dt=args.dt,
                                                      block=args.block,
                                                      batch_size=args.batch_size,
                                                      num_workers=args.num_workers,
                                                      mode=args.mode,
                                                      ds=args.ds)

    sparks, optimizer, scheduler, loss_fn = make_network_and_optimizers(args, datasets=train_datasets)

    best_test_acc = -np.inf

    # Training loop
    pbar = tqdm.tqdm(range(args.n_epochs))
    for epoch in pbar:
        train(sparks=sparks,
              train_dls=train_dls,
              loss_fn=loss_fn,
              optimizer=optimizer,
              beta=args.beta,
              device=args.device,
              mode=args.mode,
              frames=train_datasets[0].true_frames,
              online=args.online,
              dt=args.dt)
        scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            test_acc, encoder_outputs, decoder_outputs = test(sparks=sparks,
                                                               test_dls=test_dls,
                                                               loss_fn=loss_fn,
                                                               mode=args.mode,
                                                               frames=test_datasets[0].true_frames,
                                                               dt=args.dt,
                                                               act=torch.sigmoid)

            best_test_acc = save_results(args.results_path, test_acc, best_test_acc,
                                          encoder_outputs, decoder_outputs, sparks)

            if args.mode == 'prediction':
                pbar.set_description("Epoch %d, test acc: %.3f" % (epoch, test_acc))
            else:
                pbar.set_description("Epoch %d, test loss: %.3f" % (epoch, -test_acc))
