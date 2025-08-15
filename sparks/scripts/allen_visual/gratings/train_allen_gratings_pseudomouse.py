import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.allen.gratings_pseudomouse import make_gratings_dataset
from sparks.models.sparks import SPARKS
from sparks.models.dataclasses import HebbianAttentionConfig, AttentionConfig
from sparks.utils.misc import make_res_folder, save_results
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

    # data
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--neuron_type', type=str, default='VISp')

    parser.add_argument('--target_type', type=str, default='freq', choices=['freq', 'class', 'unsupervised'],
                        help='Type of target to predict: either spatial frequencies or class index')
    parser.add_argument('--p_train', type=float, default=0.8, help='Number of training example')
    parser.add_argument('--dt', type=float, default=0.001, help='time bins period')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')

    args = parser.parse_args()

    make_res_folder('allen_gratings_pseudomouse_' + args.target_type + '_' + args.neuron_type, os.getcwd(), args)

    neuron_types = [args.neuron_type]
    (train_dataset, train_dl,
     test_dataset, test_dl) = make_gratings_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                    n_neurons=args.n_neurons,
                                                    dt=args.dt,
                                                    neuron_types=neuron_types,
                                                    p_train=args.p_train,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    target_type=args.target_type,
                                                    seed=args.seed)
    np.save(args.results_path + '/good_units_ids.npy', train_dataset.good_units_ids)

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

    hebbian_config = HebbianAttentionConfig(tau_s=args.tau_s, dt=args.dt, n_heads=args.n_heads)
    attention_config = AttentionConfig(n_layers=args.n_layers, n_heads=args.n_heads)
    sparks = SPARKS(n_neurons_per_sess=len(train_dataset.good_units_ids),
                    embed_dim=args.embed_dim,
                    latent_dim=args.latent_dim,
                    tau_p=args.tau_p,
                    tau_f=args.tau_f,
                    hebbian_config=hebbian_config,
                    attention_config=attention_config,
                    output_dim_per_session=output_size,
                    device=args.device)

    if args.online:
        args.lr = args.lr / (0.25 * args.dt)
    optimizer = torch.optim.Adam(sparks.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.9)

    loss_best = -np.inf

    for epoch in tqdm.tqdm(range(args.n_epochs)):
        train_iterator = iter(train_dl)
        for inputs, targets in train_iterator:
            if args.target_type == 'unsupervised':
                targets = inputs
            else:
                targets = targets.unsqueeze(2).repeat_interleave(inputs.shape[-1], dim=-1)
            train_on_batch(sparks,
                           inputs=inputs,
                           targets=targets,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           device=args.device,
                           online=args.online,
                           beta=args.beta)

        if (epoch + 1) % args.test_period == 0:
            test_iterator = iter(test_dl)
            encoder_outputs = torch.Tensor()
            decoder_outputs = torch.Tensor()
            test_loss = 0

            for inputs, targets in test_iterator:
                if args.target_type == 'unsupervised':
                    targets = inputs
                else:
                    targets = targets.unsqueeze(2).repeat_interleave(inputs.shape[-1], dim=-1)
                test_loss, encoder_outputs_batch, decoder_outputs_batch = test_on_batch(sparks,
                                                                                        inputs=inputs,
                                                                                        targets=targets,
                                                                                        test_loss=test_loss,
                                                                                        loss_fn=loss_fn,
                                                                                        device=args.device,
                                                                                        act=torch.sigmoid)

                encoder_outputs = torch.cat((encoder_outputs, encoder_outputs_batch), dim=0)
                decoder_outputs = torch.cat((decoder_outputs, decoder_outputs_batch), dim=0)

            print('Avg test loss: ', test_loss)
            loss_best = save_results(args.results_path, -test_loss.cpu(), loss_best, encoder_outputs,
                                     decoder_outputs, sparks)
