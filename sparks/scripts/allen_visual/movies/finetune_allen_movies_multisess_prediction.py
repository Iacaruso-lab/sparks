import argparse
import os

import numpy as np
import torch

from sparks.data.allen.movies_multisess import make_allen_movies_dataset
from sparks.scripts.allen_visual.movies.utils.test import test, calculate_test_acc
from sparks.scripts.allen_visual.movies.utils.train import train
from sparks.scripts.allen_visual.movies.utils.misc import make_network_and_optimizers
from sparks.models.dataclasses import HebbianAttentionConfig
from sparks.utils.misc import save_results_finetuning

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
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
    parser.add_argument('--n_layers', type=int, default=1, help='Number of conventional attention layers')
    parser.add_argument('--dec_type', type=str, default='mlp', choices=['linear', 'mlp'],
                        help='Type of decoder (one of linear or mlp)')
    parser.add_argument('--tau_p', type=int, default=6, help='Past window size')
    parser.add_argument('--tau_f', type=int, default=1, help='Future window size')
    parser.add_argument('--tau_s', type=float, default=1., help='STDP decay')

    # Data parameters
    parser.add_argument('--n_skip_sessions', type=int, default=0, help='First session to consider')
    parser.add_argument('--n_sessions', type=int, default=1, help='How many sessions to use')
    parser.add_argument('--block', type=str, default='first',
                        choices=['first', 'second', 'across', 'both'], help='From which blocks to use')
    parser.add_argument('--dt', type=float, default=0.006, help='Time sampling period')
    parser.add_argument('--ds', type=int, default=2, help='Frame downsampling factor')

    parser.add_argument('--weights_folder', type=str, default='')

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # Create datasets and dataloaders
    (pretrain_datasets, _,
     _, pretrain_test_dls) = make_allen_movies_dataset(os.path.join(args.home, "datasets/allen_visual/"),
                                                       session_idxs=np.arange(args.n_skip_sessions),
                                                       dt=args.dt,
                                                       block=args.block,
                                                       batch_size=args.batch_size,
                                                       num_workers=args.num_workers,
                                                       mode=args.mode,
                                                       ds=args.ds)

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

    sparks, optimizer, scheduler, loss_fn = make_network_and_optimizers(args, datasets=pretrain_datasets)

    # Load pretrained network and add neural attention layers for additional sessions
    sparks.load_state_dict(torch.load(os.path.join(args.weights_folder, 'sparks.pt')))

    for i, train_dataset in enumerate(train_datasets):
        hebbian_config = HebbianAttentionConfig(tau_s=args.tau_s, dt=args.dt, n_heads=args.n_heads)
        sparks.encoder.add_neural_block(n_neurons=len(train_dataset.good_units_ids),
                                        hebbian_config=hebbian_config)

    #
    # Test accuracy on both datasets before finetuning
    new_datasets_acc_evolution = []
    pretrain_datasets_acc_evolution = []
    best_test_acc = 0

    _, encoder_outputs, decoder_outputs = test(sparks=sparks,
                                               test_dls=pretrain_test_dls,
                                               loss_fn=loss_fn,
                                               mode=args.mode,
                                               frames=test_datasets[0].true_frames,
                                               dt=args.dt,
                                               act=torch.sigmoid)
    pretrain_test_acc = calculate_test_acc(decoder_outputs, test_datasets[0].true_frames)
    pretrain_datasets_acc_evolution.append(pretrain_test_acc)

    _, encoder_outputs, decoder_outputs = test(sparks=sparks,
                                               test_dls=pretrain_test_dls,
                                               loss_fn=loss_fn,
                                               mode=args.mode,
                                               frames=test_datasets[0].true_frames,
                                               dt=args.dt,
                                               act=torch.sigmoid,
                                               sess_ids=np.arange(len(test_dls)) + args.n_skip_sessions)
    
    new_test_acc = calculate_test_acc(decoder_outputs, test_datasets[0].true_frames)
    new_datasets_acc_evolution.append(new_test_acc)

    tot_test_acc = (pretrain_test_acc * len(pretrain_test_dls) + new_test_acc * len(test_dls)) \
                    / (len(pretrain_test_dls) + len(test_dls))

    print("Start, pretrained test acc: %.3f" % pretrain_test_acc)
    print("Start, finetune test acc: %.3f" % new_test_acc)
    best_test_acc = save_results_finetuning(os.path.join(os.getcwd(), 'results', args.weigts_folder),
                                            tot_test_acc,  best_test_acc, encoder_outputs, decoder_outputs,
                                            sparks, pretrain_datasets_acc_evolution, new_datasets_acc_evolution)

    ### Finetuning
    for epoch in range(args.n_epochs):
        train(sparks=sparks,
              train_dls=train_dls,
              loss_fn=loss_fn,
              optimizer=optimizer,
              beta=args.beta,
              device=args.device,
              mode=args.mode,
              frames=train_datasets[0].true_frames,
              dt=args.dt,
              online=args.online,
              start_idx=args.n_skip_sessions)

        scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            _, encoder_outputs, decoder_outputs = test(sparks=sparks,
                                                    test_dls=pretrain_test_dls,
                                                    loss_fn=loss_fn,
                                                    mode=args.mode,
                                                    frames=test_datasets[0].true_frames,
                                                    dt=args.dt,
                                                    act=torch.sigmoid)
            pretrain_test_acc = calculate_test_acc(decoder_outputs, test_datasets[0].true_frames)
            pretrain_datasets_acc_evolution.append(pretrain_test_acc)

            _, encoder_outputs, decoder_outputs = test(sparks=sparks,
                                                    test_dls=pretrain_test_dls,
                                                    loss_fn=loss_fn,
                                                    mode=args.mode,
                                                    frames=test_datasets[0].true_frames,
                                                    dt=args.dt,
                                                    act=torch.sigmoid,
                                                    sess_ids=np.arange(len(test_dls)) + args.n_skip_sessions)
            
            new_test_acc = calculate_test_acc(decoder_outputs, test_datasets[0].true_frames)
            new_datasets_acc_evolution.append(new_test_acc)

            tot_test_acc = (pretrain_test_acc * len(pretrain_test_dls) + new_test_acc * len(test_dls)) \
                            / (len(pretrain_test_dls) + len(test_dls))

            print("Epoch %d, pretrained test acc: %.3f" % (epoch, pretrain_test_acc))
            print("Epoch %d, finetune test acc: %.3f" % (epoch, new_test_acc))
            best_test_acc = save_results_finetuning(os.path.join(os.getcwd(), 'results', args.weigts_folder),
                                                    tot_test_acc, best_test_acc, encoder_outputs, decoder_outputs,
                                                    sparks,
                                                    pretrain_datasets_acc_evolution, new_datasets_acc_evolution)

