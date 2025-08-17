import argparse
import fnmatch
import json
import os
import time
from typing import List

import numpy as np
import torch

from sparks.models.sparks import SPARKS


def make_res_folder(name: str, path_to_res: str, args: argparse.Namespace):
    """
    Constructs a folder to store experimental results with a unique id based on the given name and path.

    If the argument `args` contains the attribute `online`, '_online' is appended to the folder name.

    Creates a `commandline_args.txt` file that stores the command line arguments with which the script was run.

    Sets device for torch in `args` if cuda is available, else sets it to cpu.

    Args:
        name (str): The base name for the results directory.
        path_to_res (str): The path to the main directory where the result directories are stored.
        args (argparse.Namespace): A namespace containing arguments to the script.

    Side effects:
        This function modifies `args` to have two additional attributes:
            - results_path (str): The path to the created directory for results of this run.
            - device (torch.device): The device used for computations.

    Returns:
        None
    """

    prelist = np.sort(fnmatch.filter(os.listdir(os.path.join(path_to_res, r"results")), '[0-9][0-9][0-9]__*'))
    if len(prelist) == 0:
        expDirN = "001"
    else:
        expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

    online_flag = '_online' if args.online else ''
    try:
        args.results_path = time.strftime(path_to_res + '/results/' + expDirN + "__" + "%d-%m-%Y_"
                                          + name + online_flag + '_taus_' + str(args.tau_s) + '_taup_' + str(args.tau_p)
                                          + '_tauf_' + str(args.tau_f) + '_embed_dim_' + str(args.embed_dim)
                                          + "_latent_dim_" + str(args.latent_dim) + "_beta_" + str(args.beta)
                                          + "_lr_" + str(args.lr) + "_n_layers_" + str(args.n_layers), time.localtime())
    except AttributeError:
        args.results_path = time.strftime(path_to_res + '/results/' + expDirN + "__" + "%d-%m-%Y_"
                                          + name + online_flag, time.localtime())
    os.makedirs(args.results_path)

    with open(os.path.join(args.results_path, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps:0')
    else:
        args.device = torch.device('cpu')


def save_results(results_path: str,
                 test_acc: float,
                 best_test_acc: float,
                 encoder_outputs: torch.Tensor,
                 decoder_outputs: torch.Tensor,
                 sparks: SPARKS):
    """
    Saves test results, and the current encoder and decoder states if the test_acc is greater than the best seen so far.

    If test_acc is greater than best_test_acc, updates best_test_acc and saves the test accuracy, encoder outputs,
    decoder outputs, and state dictionaries of encoder and decoder to disk under results_path.
    Otherwise, only saves the encoder and decoder outputs.

    Args:
        results_path (str): The path where to save the results.
        test_acc (float): The achieved test accuracy.
        best_test_acc (float): The best test accuracy seen so far.
        sparks (SPARKS): The SPARKS instance.

    Returns:
        best_test_acc (float): The updated best test accuracy.
    """

    if test_acc >= best_test_acc:
        best_test_acc = test_acc
        np.save(results_path + '/test_acc.npy', test_acc)
        np.save(results_path + '/test_enc_outputs_best.npy', encoder_outputs.cpu().numpy())
        torch.save(sparks.state_dict(), results_path + '/sparks.pt')
        np.save(results_path + '/test_dec_outputs_best.npy', decoder_outputs.cpu().numpy())
    else:
        np.save(results_path + '/test_dec_outputs_last.npy', decoder_outputs.cpu().numpy())
        np.save(results_path + '/test_enc_outputs_last.npy', encoder_outputs.cpu().numpy())

    return best_test_acc


def save_results_finetuning(results_path: str,
                            test_acc: float,
                            best_test_acc: float,
                            encoder_outputs: torch.Tensor,
                            decoder_outputs: torch.Tensor,
                            sparks: SPARKS,
                            pretrain_datasets_acc_evolution: List[float],
                            new_datasets_acc_evolution: List[float]):
    """
    Saves test results, and the current encoder and decoder states if the test_acc is greater than best_test_acc.

    If test_acc is greater than best_test_acc, saves the test accuracy, encoder outputs, decoder outputs,
    state dictionaries of encoder and decoder, accuracy evolution of pretraining datasets,
    and accuracy evolution of the new datasets to disk under results_path.

    Args:
        results_path (str): The path where to save the results.
        test_acc (float): The achieved test accuracy.
        best_test_acc (float): The best test accuracy achieved so far.
        encoder_outputs (torch.Tensor): The tensor of encoder outputs.
        decoder_outputs (torch.Tensor): The tensor of decoder outputs.
        encoder (torch.nn.Module): The encoder model of the VAE.
        decoder (torch.nn.Module): The decoder model of the VAE.
        pretrain_datasets_acc_evolution (list of float): The list of accuracy values for the pretraining datasets over time.
        new_datasets_acc_evolution (list of float): The list of accuracy values for the new datasets over time.

    Returns:
        best_test_acc (float): The updated best test accuracy.
    """

    if test_acc > best_test_acc:
        np.save(os.path.join(results_path, 'finetune_test_enc_outputs_best.npy'),
                encoder_outputs.cpu().numpy())
        torch.save(sparks.state_dict(),
                   os.path.join(results_path, 'sparks.pt'))
        np.save(os.path.join(results_path, 'finetune_test_dec_outputs_best.npy'),
                decoder_outputs.cpu().numpy())

        np.save(os.path.join(results_path, 'pretrain_datasets_acc_evolution.npy'),
                np.array(pretrain_datasets_acc_evolution))
        np.save(os.path.join(results_path, 'new_datasets_acc_evolution.npy'),
                np.array(new_datasets_acc_evolution))

        return best_test_acc


def identity(x):
    return x
