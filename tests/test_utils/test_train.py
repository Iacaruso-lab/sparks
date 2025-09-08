import torch


def test_update_and_reset():
    """Test gradient update and reset function"""
    from sparks.utils.train import update_and_reset
    from sparks.models.sparks import SPARKS
    
    model = SPARKS(
        n_neurons_per_session=10,
        embed_dim=64,
        latent_dim=32,
        output_dim_per_session=10
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = torch.tensor(1.0, requires_grad=True)

    spikes = torch.bernoulli(torch.ones(4, 10, 20) * 0.1)
    encoder_outputs, decoder_outputs, mu, logvar = model(spikes[..., 0], 
                                                         torch.zeros([spikes.shape[0], 32, 1]), 
                                                         session_id=0)
    targets = torch.rand(decoder_outputs.shape)
    loss = loss_fn(decoder_outputs, targets)
    update_and_reset(model, loss, optimizer)
    
    # Check that gradients are reset
    for p in model.parameters():
        assert p.grad is None or p.grad.sum() == 0


def test_train_on_batch():
    """Test training on a single batch"""
    from sparks.utils.train import train_on_batch
    from sparks.models.sparks import SPARKS
    from sparks.data.providers import StandardTargetProvider
    
    batch_size = 8
    n_neurons = 20
    timesteps = 30
    latent_dim = 32
    tau_p = 3
    tau_f = 2
    output_dim = 10
    
    model = SPARKS(
        n_neurons_per_session=n_neurons,
        embed_dim=64,
        latent_dim=latent_dim,
        tau_p=tau_p,
        tau_f=tau_f,
        output_dim_per_session=output_dim
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    
    inputs = torch.randn(batch_size, n_neurons, timesteps)
    targets = torch.randn(batch_size, output_dim, timesteps)
    target_provider = StandardTargetProvider(targets)
    
    # Test offline training
    train_on_batch(
        sparks=model,
        inputs=inputs,
        target_provider=target_provider,
        loss_fn=loss_fn,
        optimizer=optimizer,
        beta=0.1,
        device='cpu',
        online=False,
        burnin=5
    )
    
    # Test online training
    train_on_batch(
        sparks=model,
        inputs=inputs,
        target_provider=target_provider,
        loss_fn=loss_fn,
        optimizer=optimizer,
        beta=0.1,
        device='cpu',
        online=True,
        burnin=5
    )
