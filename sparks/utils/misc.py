import argparse
import fnmatch
import json
import os
import time
from typing import List
import random

import numpy as np
import torch


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
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module):
    """
    Saves test results, and the current encoder and decoder states if the test_acc is greater than the best seen so far.

    If test_acc is greater than best_test_acc, updates best_test_acc and saves the test accuracy, encoder outputs,
    decoder outputs, and state dictionaries of encoder and decoder to disk under results_path.
    Otherwise, only saves the encoder and decoder outputs.

    Args:
        results_path (str): The path where to save the results.
        test_acc (float): The achieved test accuracy.
        best_test_acc (float): The best test accuracy seen so far.
        encoder_outputs (torch.Tensor): The tensor of encoder outputs.
        decoder_outputs (torch.Tensor): The tensor of decoder outputs.
        encoder (torch.nn.Module): The encoder model of the VAE.
        decoder (torch.nn.Module): The decoder model of the VAE.

    Returns:
        best_test_acc (float): The updated best test accuracy.
    """

    if test_acc >= best_test_acc:
        best_test_acc = test_acc
        np.save(results_path + '/test_acc.npy', test_acc)
        np.save(results_path + '/test_enc_outputs_best.npy', encoder_outputs.cpu().numpy())
        torch.save(encoder.state_dict(), results_path + '/encoding_network.pt')
        torch.save(decoder.state_dict(), results_path + '/decoding_network.pt')
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
                            encoder: torch.nn.Module,
                            decoder: torch.nn.Module,
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
        torch.save(encoder.state_dict(),
                   os.path.join(results_path, 'finetune_encoding_network.pt'))
        torch.save(decoder.state_dict(),
                   os.path.join(results_path, 'finetune_decoding_network.pt'))
        np.save(os.path.join(results_path, 'finetune_test_dec_outputs_best.npy'),
                decoder_outputs.cpu().numpy())

        np.save(os.path.join(results_path, 'pretrain_datasets_acc_evolution.npy'),
                np.array(pretrain_datasets_acc_evolution))
        np.save(os.path.join(results_path, 'new_datasets_acc_evolution.npy'),
                np.array(new_datasets_acc_evolution))

        return best_test_acc


def identity(x):
    return x


"""
=======================================================================================================================

Adapted from https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/training/cyclers.py
"""


def cycle(iterable):
    # see https://github.com/pytorch/pytorch/issues/23900
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def alternate(*args):
    """
    Given multiple iterators, returns a generator that alternatively visit one element from each iterator at a time.

    Examples:
        >>> list(alternate(['a', 'b', 'c'], [1, 2, 3], ['Mon', 'Tue', 'Wed']))
        ['a', 1, 'Mon', 'b', 2, 'Tue', 'c', 3, 'Wed']

    Args:
        *args: one or more iterables (e.g. tuples, list, iterators) separated by commas

    Returns:
        A generator that alternatively visits one element at a time from the list of iterables
    """
    for row in zip(*args):
        yield from row


def cycle_datasets(loaders):
    """
    Given a dictionary mapping data_key into dataloader objects, returns a generator that alternately yields
    output from the loaders in the dictionary. The order of data_key traversal is determined by the first invocation to `.keys()`.
    To obtain deterministic behavior of key traversal, recommended to use OrderedDict.

    The generator terminates as soon as any one of the constituent loaders is exhausted.

    Args:
        loaders (dict): Dict mapping a data_key to a dataloader object.

    Yields:
        string, Any: data_key  and and the next output from the data loader corresponding to the data_key
    """
    keys = list(loaders.keys())
    # establish a consistent ordering across loaders
    ordered_loaders = [loaders[k] for k in keys]
    for data_key, outputs in zip(cycle(loaders.keys()), alternate(*ordered_loaders)):
        yield data_key, outputs


class Exhauster:
    """
    Given a dictionary of data loaders, mapping data_key into a data loader, steps through each data loader, moving onto the next data loader
    only upon exhausing the content of the current data loader.
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        for data_key, loader in self.loaders.items():
            for batch in loader:
                yield data_key, batch

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])


class LongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders]
        for loader, _ in zip((cycle(cycles)), range(len(self.loaders) * self.max_batches), ):
            yield next(loader)

    def __len__(self):
        return len(self.loaders) * self.max_batches


class ShortCycler:
    """
    Cycles through trainloaders until the loader with smallest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.min_batches = min([len(loader) for loader in self.loaders])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders]
        for loader, _ in zip((cycle(cycles)), range(len(self.loaders) * self.min_batches), ):
            yield next(loader)

    def __len__(self):
        return len(self.loaders) * self.min_batches


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class MultiSessionDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)


class RandomCycler:
    """
    Randomly selects a dataloader and yields its next item.
    Aware not to select a dataloader that has already been exhausted.
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.iterators = [iter(loader) for loader in loaders]
        self.active_loaders = list(range(len(self.loaders)))

    def __iter__(self):
        while self.active_loaders:
            loader_idx = random.choice(self.active_loaders)
            try:
                yield loader_idx, next(self.iterators[loader_idx])
            except StopIteration:
                self.active_loaders.remove(loader_idx)
                self.__iter__()
        
    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

"""
=======================================================================================================================
"""
