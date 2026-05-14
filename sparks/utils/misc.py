import argparse
import fnmatch
import json
import os
import time
from typing import List, Union

import numpy as np
import torch

from sparks.models.sparks import SPARKS
from sparks.models.transformer import HebbianTransformer


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

    if not os.path.exists(path_to_res):
        os.makedirs(path_to_res)

    prelist = np.sort(fnmatch.filter(os.listdir(path_to_res), '[0-9][0-9][0-9]__*'))
    if len(prelist) == 0:
        expDirN = "001"
    else:
        expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

    args.results_path = time.strftime(path_to_res + '/' + expDirN + "__" + "%d-%m-%Y_" + name, time.localtime())
    os.makedirs(args.results_path)

    with open(os.path.join(args.results_path, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps:0')
    else:
        args.device = torch.device('cpu')


def identity(x):
    return x


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_acc = -np.inf

    def early_stop(self, test_acc):
        if test_acc > self.max_acc:
            self.max_acc = test_acc
            self.counter = 0
        elif test_acc < (self.max_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
