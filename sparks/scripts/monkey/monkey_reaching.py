import argparse
import os

import numpy as np
import torch
import tqdm

from sparks.data.nlb.monkey_reaching import make_monkey_reaching_dataset
from sparks.models.sparks import SPARKS
from sparks.models.dataclasses import HebbianAttentionConfig, AttentionConfig
from sparks.scripts.monkey.utils import get_accuracy
from sparks.utils.misc import make_res_folder, save_results
from sparks.utils.test import test
from sparks.utils.train import train

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.000001, help='KLD regularisation')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='For dataloading')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Whether to use the online gradient descent algorithm')

    # Encoder parameters
    parser.add_argument('--latent_dim', type=int, default=3, help='Size of the latent space')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=64, help='Size of attention embeddings')
    parser.add_argument('--n_layers', type=int, default=0, help='Number of conventional attention layers')
    parser.add_argument('--output_type', type=str, default='flatten',
                          help='Output architecture for the decoder')
    parser.add_argument('--dec_type', type=str, default='mlp', choices=['linear', 'mlp', 'rnn'],
                        help='Type of decoder (one of linear or mlp)')
    parser.add_argument('--tau_p', type=int, default=10, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, default=0.5, help='STDP decay')

    # Data parameters
    parser.add_argument('--target_type', type=str, default='hand_pos', choices=['hand_pos', 'hand_vel',
                                                                                'force', 'muscle_len',
                                                                                 'direction', 'joint_ang'])
    parser.add_argument('--mode', type=str, default='prediction', choices=['prediction', 'unsupervised', 'spikes_pred'],
                        help='Which type of task to perform')
    parser.add_argument('--dt', type=float, default=0.001, help='time bins period')

    args = parser.parse_args()

    # Create folder to save results
    if args.mode == 'prediction':
        make_res_folder('monkey_reaching' + '_' + args.target_type, os.getcwd(), args)
    else:
        make_res_folder('monkey_reaching' + '_' + args.mode, os.getcwd(), args)

    # Create dataloaders
    (train_dataset, test_dataset,
     train_dl, test_dl) = make_monkey_reaching_dataset(os.path.join(args.home, "datasets/000127/sub-Han/"),
                                                       mode=args.mode,
                                                       y_keys=args.target_type,
                                                       batch_size=args.batch_size)

    # Make network
    if args.mode == 'prediction':
        if args.target_type == 'direction':
            output_size = len(np.unique(train_dataset.y_trial_data))
        else:
            output_size = train_dataset.y_shape
    elif args.mode in ['unsupervised', 'spikes_pred']:
        output_size = train_dataset.x_shape
    else:
        raise NotImplementedError

    hebbian_config = HebbianAttentionConfig(tau_s=args.tau_s, dt=args.dt, n_heads=args.n_heads)
    attention_config = AttentionConfig(n_layers=args.n_layers, n_heads=args.n_heads)
    sparks = SPARKS(n_neurons_per_sess=train_dataset.x_shape,
                    embed_dim=args.embed_dim,
                    latent_dim=args.latent_dim,
                    tau_p=args.tau_p,
                    tau_f=args.tau_f,
                    hebbian_config=hebbian_config,
                    attention_config=attention_config,
                    output_dim_per_session=output_size,
                    device=args.device)

    # Training parameters
    optimizer = torch.optim.Adam(sparks.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    if args.target_type == 'direction':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    best_test_acc = -np.inf

    # Training loop
    pbar = tqdm.tqdm(range(args.n_epochs))
    for epoch in pbar:
        train(sparks=sparks,
              train_dls=[train_dl],
              loss_fn=loss_fn,
              optimizer=optimizer,
              beta=args.beta,
              device=args.device)
        scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            test_loss, encoder_outputs, decoder_outputs = test(sparks=sparks,
                                                               test_dls=[test_dl],
                                                               loss_fn=loss_fn,
                                                               act=torch.sigmoid)

            test_acc = get_accuracy(decoder_outputs, test_dataset, test_loss, args.mode, args.target_type)
            pbar.set_description("Epoch %d, acc: %.3f" % (epoch, test_acc))

            best_test_acc = save_results(args.results_path, test_acc, best_test_acc,
                                          encoder_outputs, decoder_outputs, sparks)
