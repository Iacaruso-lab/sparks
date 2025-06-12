import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.allen.gratings_pseudomouse import make_gratings_dataset
from sparks.models.decoders import get_decoder
from sparks.models.encoders import HebbianTransformerEncoder
from sparks.utils.misc import make_res_folder, save_results, LongCycler
from sparks.utils.test import test_on_batch
from sparks.utils.train import train_on_batch

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
    parser.add_argument('--alpha', type=float, default=1., help='')

    # data
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--target_type', type=str, default='freq', choices=['freq', 'class', 'unsupervised'],
                    help='Type of target to predict: either spatial frequencies or class index')
    parser.add_argument('--p_train', type=float, default=0.8, help='Number of training example')
    parser.add_argument('--dt', type=float, default=0.001, help='time bins period')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')

    args = parser.parse_args()
    neuron_types = ['VISp', 'VISal', 'VISrl', 'VISpm', 'VISam', 'VISl']
    args.neuron_types = neuron_types

    make_res_folder('allen_gratings_pseudomouse_multiheaded_' + args.target_type, os.getcwd(), args)

    train_datasets = []
    test_datasets = []
    train_dls = []
    test_dls = []

    for neuron_type in neuron_types:
        (train_dataset, train_dl,
        test_dataset, test_dl) = make_gratings_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                        n_neurons=args.n_neurons,
                                                        dt=args.dt,
                                                        neuron_types=[neuron_type],
                                                        p_train=args.p_train,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.num_workers,
                                                        target_type=args.target_type,
                                                        seed=args.seed)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        train_dls.append(train_dl)
        test_dls.append(test_dl)
        np.save(args.results_path + '/good_units_ids_%s.npy' % neuron_type, train_dataset.good_units_ids)

    encoding_network = HebbianTransformerEncoder(n_neurons_per_sess=[args.n_neurons] * len(train_dataset),
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
        output_size = args.n_neurons
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError

    decoding_network = get_decoder(output_dim_per_session=output_size * args.tau_f, args=args)

    if args.online:
        args.lr = args.lr / (0.25 * args.dt)
    optimizer = torch.optim.Adam(list(encoding_network.parameters())
                                 + list(decoding_network.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    loss_best = np.inf

    for epoch in tqdm.tqdm(range(args.n_epochs)):
        random_order = np.random.choice(np.arange(len(train_dls)), size=len(train_dls), replace=False)
        train_iterator = LongCycler([train_dls[i] for i in random_order])
        for i, (inputs, targets) in enumerate(train_iterator):
            if args.target_type == 'unsupervised':
                targets = inputs
            else:
                targets = targets.unsqueeze(2).repeat_interleave(inputs.shape[-1], dim=-1)
            train_on_batch(encoder=encoding_network,
                           decoder=decoding_network,
                           inputs=inputs,
                           targets=targets,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           latent_dim=args.latent_dim,
                           tau_p=args.tau_p,
                           tau_f=args.tau_f,
                           device=args.device,
                           online=args.online,
                           sess_id=random_order[i % len(train_dls)],
                           beta=args.beta)

        if (epoch + 1) % args.test_period == 0:
            test_loss = 0
            all_encoder_outputs = []
            all_decoder_outputs = []

            test_iterators = [iter(test_dl) for test_dl in test_dls]

            for i, test_iterator in enumerate(test_iterators):
                encoder_outputs = torch.Tensor()
                decoder_outputs = torch.Tensor()

                for inputs, targets in test_iterator:
                    if args.target_type == 'unsupervised':
                        targets = inputs
                    else:
                        targets = targets.unsqueeze(2).repeat_interleave(inputs.shape[-1], dim=-1)
                    test_loss, encoder_outputs_batch, decoder_outputs_batch = test_on_batch(encoder=encoding_network,
                                                                                            decoder=decoding_network,
                                                                                            inputs=inputs,
                                                                                            targets=targets,
                                                                                            latent_dim=args.latent_dim,
                                                                                            tau_p=args.tau_p,
                                                                                            tau_f=args.tau_f,
                                                                                            test_loss=test_loss,
                                                                                            loss_fn=loss_fn,
                                                                                            sess_id=i,
                                                                                            device=args.device,
                                                                                            act=torch.sigmoid)

                    encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch.cpu()), dim=0)
                    decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch.cpu()), dim=0)

                all_encoder_outputs.append(encoder_outputs)
                all_decoder_outputs.append(decoder_outputs)

            print('Avg test loss: ', test_loss)
            if test_loss < loss_best:
                loss_best = test_loss
                np.save(args.results_path + '/test_loss.npy', test_loss)
                torch.save(encoding_network.state_dict(), args.results_path + '/encoding_network.pt')
                torch.save(decoding_network.state_dict(), args.results_path + '/decoding_network.pt')
                for i, neuron_type in enumerate(neuron_types):
                    np.save(args.results_path + '/test_enc_outputs_best_%s.npy' % neuron_type, 
                            all_encoder_outputs[i].numpy())
                    np.save(args.results_path + '/test_dec_outputs_best_%s.npy' % neuron_type,
                            all_decoder_outputs[i].numpy())
            else:
                for i, neuron_type in enumerate(neuron_types):
                    np.save(args.results_path + '/test_enc_outputs_last_%s.npy' % neuron_type, 
                            all_encoder_outputs[i].numpy())
                    np.save(args.results_path + '/test_dec_outputs_last_%s.npy' % neuron_type,
                            all_decoder_outputs[i].numpy())
