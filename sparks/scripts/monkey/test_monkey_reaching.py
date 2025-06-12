import argparse
import os

import numpy as np
import torch

from sparks.data.nlb.monkey_reaching import make_monkey_reaching_dataset
from sparks.models.decoders import get_decoder
from sparks.models.encoders import HebbianTransformerEncoder
from sparks.utils.test import test_init
from sparks.utils.vae import ae_forward

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='')

    # Training arguments
    parser.add_argument('--home', default=r"/home")
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--test_period', type=int, default=5, help='Test period in number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='KLD regularisation')
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
    parser.add_argument('--w_start', type=float, default=0., help='')
    parser.add_argument('--alpha', type=float, default=0.5, help='')

    # Data parameters
    parser.add_argument('--target_type', type=str, default='hand_pos', choices=['hand_pos', 'hand_vel',
                                                                                'force', 'muscle_len',
                                                                                 'direction', 'joint_ang'])
    parser.add_argument('--mode', type=str, default='prediction', choices=['prediction', 
                                                                           'unsupervised',
                                                                           'unsupervised_pred'],
                        help='Which type of task to perform')
    parser.add_argument('--dt', type=float, default=0.006, help='Time sampling period')

    parser.add_argument('--weights_folder', type=str, default='')

    args = parser.parse_args()

    # Create folder to save results
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps:0')
    else:
        args.device = torch.device('cpu')

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
    elif args.mode in ['unsupervised', 'unsupervised_pred']:
        output_size = train_dataset.x_shape
    else:
        raise NotImplementedError

    decoding_network = get_decoder(output_dim_per_session=output_size * args.tau_f,
                                   args=args, softmax=True if args.target_type == 'direction' else False)
    
    # Load pretrained network and add neural attention layers for additional sessions
    encoding_network.load_state_dict(torch.load(os.path.join(os.getcwd(), 'results',
                                                             args.weights_folder, 'encoding_network.pt')))
    decoding_network.load_state_dict(torch.load(os.path.join(os.getcwd(), 'results',
                                                             args.weights_folder, 'decoding_network.pt')))
    
    with torch.no_grad():
        encoding_network.eval()
        decoding_network.eval()

        encoder_outputs = torch.Tensor()
        decoder_outputs = torch.Tensor()
        attn_coeffs = torch.Tensor()
        pre_trace = torch.Tensor()
        post_trace = torch.Tensor()

        test_acc = 0

        test_iterator = iter(test_dl)

        for inputs, targets in test_iterator:
            attn_coeffs_batch = torch.Tensor()
            pre_trace_batch = torch.Tensor()
            post_trace_batch = torch.Tensor()

            encoder_outputs_batch, decoder_outputs_batch, T = test_init(encoding_network, inputs,
                                                                        args.latent_dim, args.tau_p, args.device)

            for t in range(T):
                encoder_outputs_batch, decoder_outputs, _, _ = ae_forward(encoding_network, decoding_network,
                                                                          inputs[..., t], encoder_outputs_batch,
                                                                          args.tau_p, args.device)
                attn_coeffs_batch = torch.cat((attn_coeffs_batch,
                                               encoding_network.hebbian_attn_blocks[0].attention_layer.heads[0].attention.unsqueeze(3).cpu()), dim=-1)
                pre_trace_batch = torch.cat((pre_trace_batch,
                                             encoding_network.hebbian_attn_blocks[0].attention_layer.heads[0].pre_trace.unsqueeze(3).cpu()), dim=-1)
                post_trace_batch = torch.cat((post_trace_batch,
                                             encoding_network.hebbian_attn_blocks[0].attention_layer.heads[0].post_trace.unsqueeze(3).cpu()), dim=-1)

                decoder_outputs_batch = torch.cat((decoder_outputs_batch, decoder_outputs.unsqueeze(2)), dim=-1)

        attn_coeffs = torch.cat((attn_coeffs, attn_coeffs_batch), dim=0)
        np.save(os.path.join(os.getcwd(), 'results', args.weights_folder, 'attn_coeffs.npy'),
                attn_coeffs.numpy())
        
        pre_trace = torch.cat((pre_trace, pre_trace_batch), dim=0)
        np.save(os.path.join(os.getcwd(), 'results', args.weights_folder, 'pre_trace.npy'),
                pre_trace.numpy())
        
        post_trace = torch.cat((post_trace, post_trace_batch), dim=0)
        np.save(os.path.join(os.getcwd(), 'results', args.weights_folder, 'post_trace.npy'),
                post_trace.numpy())


