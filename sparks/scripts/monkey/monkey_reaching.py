import argparse
import os

import numpy as np
import torch
import tqdm
from sklearn.metrics import r2_score

from sparks.data.nlb.monkey_reaching import make_monkey_reaching_dataset
from sparks.models.decoders import get_decoder
from sparks.models.encoders import HebbianTransformerEncoder
from sparks.utils.misc import make_res_folder, save_results
from sparks.utils.test import test
from sparks.utils.train import train
from sparks.utils.cobps import bits_per_spike

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
    parser.add_argument('--w_start', type=float, default=1., help='')
    parser.add_argument('--alpha', type=float, default=1., help='')

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
        make_res_folder('monkey_reaching_alpha_' + str(args.alpha) + '_' + args.target_type, os.getcwd(), args)
    else:
        make_res_folder('monkey_reaching_alpha_' + str(args.alpha) + '_' + args.mode, os.getcwd(), args)

    # Create dataloaders
    (train_dataset, test_dataset,
     train_dl, test_dl) = make_monkey_reaching_dataset(os.path.join(args.home, "datasets/000127/sub-Han/"),
                                                       mode=args.mode,
                                                       y_keys=args.target_type,
                                                       batch_size=args.batch_size)

    # Make networks
    encoding_network = HebbianTransformerEncoder(n_neurons_per_sess=train_dataset.x_shape,
                                                 output_type=args.output_type,
                                                 embed_dim=args.embed_dim,
                                                 latent_dim=args.latent_dim,
                                                 tau_s_per_sess=args.tau_s,
                                                 dt_per_sess=args.dt,
                                                 n_layers=args.n_layers,
                                                 w_start=args.w_start,
                                                 alpha=args.alpha).to(args.device)

    if args.mode == 'prediction':
        if args.target_type == 'direction':
            output_size = len(np.unique(train_dataset.y_trial_data))
        else:
            output_size = train_dataset.y_shape
    elif args.mode in ['unsupervised', 'spikes_pred']:
        output_size = train_dataset.x_shape
    else:
        raise NotImplementedError

    decoding_network = get_decoder(output_dim_per_session=output_size * args.tau_f,
                                   args=args, softmax=True if args.target_type == 'direction' else False,
                                   hid_features = [args.latent_dim * args.tau_p,
                                                   args.latent_dim * args.tau_p,
                                                   int(np.mean([args.latent_dim * args.tau_p,
                                                                 np.mean(output_size * args.tau_f)]))])

    optimizer = torch.optim.Adam(list(encoding_network.parameters())
                                 + list(decoding_network.parameters()), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    if args.target_type == 'direction':
        loss_fn = torch.nn.NLLLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    best_test_acc = -np.inf

    for epoch in tqdm.tqdm(range(args.n_epochs)):
        train(encoder=encoding_network,
              decoder=decoding_network,
              train_dls=[train_dl],
              loss_fn=loss_fn,
              optimizer=optimizer,
              latent_dim=args.latent_dim,
              tau_p=args.tau_p,
              tau_f=args.tau_f,
              beta=args.beta,
              device=args.device)
        scheduler.step()

        if (epoch + 1) % args.test_period == 0:
            test_loss, encoder_outputs, decoder_outputs = test(encoding_network,
                                                               decoding_network,
                                                               [test_dl],
                                                               latent_dim=args.latent_dim,
                                                               tau_p=args.tau_p,
                                                               tau_f=args.tau_f,
                                                               loss_fn=loss_fn,
                                                               device=args.device,
                                                               act=torch.sigmoid)

            if args.mode == 'prediction':
                if args.target_type == 'direction':
                    test_targets = test_dataset.y_trial_data[..., 100:]
                    preds = decoder_outputs[..., 100:].cpu().numpy().transpose(0, 2, 1)
                    test_acc = np.mean(test_targets == preds.argmax(-1, keepdims=True)[..., 0])
                else:
                    test_targets = test_dataset.y_trial_data[..., 100:].transpose(0, 2, 1)
                    preds = decoder_outputs[..., 100:].cpu().numpy().transpose(0, 2, 1).reshape([test_targets.shape[0], 
                                                                                                test_targets.shape[1],
                                                                                                test_targets.shape[2], 
                                                                                                -1])
                    test_acc = r2_score(test_targets.reshape(-1, test_dataset.y_trial_data.shape[-2]),
                                        preds[..., 0].reshape(-1, test_dataset.y_trial_data.shape[-2]),
                                        multioutput='variance_weighted')
                print("Epoch %d, acc: %.3f" % (epoch, test_acc))
            elif args.mode == 'spikes_pred':
                test_targets = test_dataset.x_trial_data
                preds = decoder_outputs.cpu().numpy().reshape([test_targets.shape[0], 
                                                               test_targets.shape[1],
                                                               -1,
                                                               test_targets.shape[2]])[:, :, 0]
                test_acc = bits_per_spike(preds, test_targets)
                print("Epoch %d, acc: %.3f" % (epoch, test_acc))
            else:
                test_acc = -test_loss
                print("Epoch %d, loss: %.3f" % (epoch, test_loss))

            best_test_acc = save_results(args.results_path, test_acc, best_test_acc, encoder_outputs,
                                         decoder_outputs, encoding_network, decoding_network)
            # np.save(args.results_path + '/test_enc_outputs_%d.npy' % epoch, encoder_outputs.cpu().numpy())

