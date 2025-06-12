import argparse
import os

import numpy as np
import torch

from sparks.data.mec.mec_ca import make_mec_ca_dataset
from sparks.models.decoders import get_decoder
from sparks.models.encoders import HebbianTransformerEncoder
from sparks.utils.misc import make_res_folder, save_results
from sparks.utils.test import test
from sparks.utils.train import train

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Whether to use the online gradient descent algorithm')

    parser.add_argument('--latent_dim', type=int, default=2, help='Size of the latent space')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=64, help='Size of attention embeddings')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--dec_type', type=str, default='mlp', choices=['linear', 'mlp'],
                        help='Type of decoder (one of linear or mlp)')
    parser.add_argument('--tau_p', type=int, default=50, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, default=1., help='STDP decay')
    parser.add_argument('--alpha', type=float, default=0.2, help='')

    # Data parameters
    parser.add_argument('--ds', type=int, default=4, help='downsampling factor')
    parser.add_argument('--mode', type=str, default='prediction', choices=['prediction', 'unsupervised'],
                        help='Which type of task to perform')

    args = parser.parse_args()

    # Create folder to save results
    args.dt = args.ds / 7.73
    make_res_folder('mec_' + args.mode + '_alpha_' + str(args.alpha) + '_ca', os.getcwd(), args)

    data_path = os.path.join(args.home, 'datasets/mec/calcium_activity_matrix_60584_session17.mat')
    if args.mode == 'prediction':
        start_stop_times_train = np.array([(600 + i * 500, 600 + (i + 1) * 500) for i in range(5)])
        start_stop_times_test = np.array([[3100, 3600]])
    elif args.mode == 'unsupervised':
        start_stop_times_train = np.array([(600 + i * 500, 600 + (i + 1) * 500) for i in range(6)])
        start_stop_times_test = np.array([[600, 3600]])
    else:
        raise NotImplementedError

    train_dataset, train_dl = make_mec_ca_dataset(data_path,
                                                  start_stop_times=start_stop_times_train,
                                                  downsampling_factor=1,
                                                  batch_size=args.batch_size,
                                                  train=True,
                                                  num_workers=args.num_workers)

    test_dataset, test_dl = make_mec_ca_dataset(data_path,
                                                start_stop_times=start_stop_times_test,
                                                downsampling_factor=args.ds,
                                                batch_size=args.batch_size,
                                                train=False,
                                                num_workers=args.num_workers)

    input_size = len(train_dataset.spikes)
    encoding_network = HebbianTransformerEncoder(n_neurons_per_sess=input_size,
                                                 embed_dim=args.embed_dim,
                                                 latent_dim=args.latent_dim,
                                                 tau_s_per_sess=args.tau_s,
                                                 dt_per_sess=args.dt,
                                                 n_layers=args.n_layers,
                                                 n_heads=args.n_heads,
                                                 alpha=args.alpha).to(args.device)

    decoding_network = get_decoder(output_dim_per_session=input_size * args.tau_f, args=args)

    if args.online:
        args.lr = args.lr / (train_dataset[0][0].shape[-1])
    optimizer = torch.optim.Adam(list(encoding_network.parameters())
                                 + list(decoding_network.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_test_acc = -np.inf

    for epoch in range(args.n_epochs):
        train(encoder=encoding_network,
              decoder=decoding_network,
              train_dls=[train_dl],
              loss_fn=loss_fn,
              optimizer=optimizer,
              latent_dim=args.latent_dim,
              tau_p=args.tau_p,
              tau_f=args.tau_f,
              beta=args.beta,
              online=args.online,
              device=args.device)
        scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            test_loss, encoder_outputs, decoder_outputs = test(encoder=encoding_network,
                                                               decoder=decoding_network,
                                                               test_dls=[test_dl],
                                                               latent_dim=args.latent_dim,
                                                               tau_p=args.tau_p,
                                                               tau_f=args.tau_f,
                                                               loss_fn=loss_fn,
                                                               device=args.device)

            print("Epoch %d, test loss: %.3f" % (epoch, test_loss.cpu().numpy()))
            best_test_acc = save_results(args.results_path, -test_loss, best_test_acc, encoder_outputs,
                                         decoder_outputs, encoding_network, decoding_network)
