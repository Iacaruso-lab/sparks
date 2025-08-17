import torch
import numpy as np

from sparks.models.sparks import SPARKS
from sparks.models.dataclasses import HebbianAttentionConfig, AttentionConfig


def make_network_and_optimizers(args, datasets):
    """
    This function creates the SPARKS network and the optimizers for training.
    It is used in both single-session and multi-session training scripts.
    """

    input_sizes = [len(train_dataset.good_units_ids) for train_dataset in datasets]
    if args.mode == 'prediction':
        output_sizes = 900
    elif args.mode == 'reconstruction':
        output_sizes = [np.prod(dataset.targets.shape[:-1]) for dataset in datasets]
    elif args.mode == 'unsupervised':
        output_sizes = input_sizes
    else:
        raise NotImplementedError

    hebbian_config = HebbianAttentionConfig(tau_s=args.tau_s, dt=args.dt, n_heads=args.n_heads)
    attention_config = AttentionConfig(n_layers=args.n_layers, n_heads=args.n_heads)
    
    sparks = SPARKS(n_neurons_per_sess=input_sizes,
                    embed_dim=args.embed_dim,
                    latent_dim=args.latent_dim,
                    tau_p=args.tau_p,
                    tau_f=args.tau_f,
                    hebbian_config=hebbian_config,
                    attention_config=attention_config,
                    output_dim_per_session=output_sizes,
                    device=args.device)

    if args.online:
        args.lr = args.lr / 900
    optimizer = torch.optim.Adam(sparks.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    if args.mode == 'prediction':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    return sparks, optimizer, scheduler, loss_fn