import torch

def test_test_on_batch():
    """Test evaluation on a single batch"""
    from sparks.utils.test import test_on_batch
    from sparks.models.sparks import SPARKS
    from sparks.data.providers import StandardTargetProvider
    
    batch_size = 4
    n_neurons = 20
    timesteps = 30
    latent_dim = 32
    tau_p = 2
    tau_f = 1
    output_dim = 10
    
    model = SPARKS(
        n_neurons_per_session=n_neurons,
        embed_dim=64,
        latent_dim=latent_dim,
        tau_p=tau_p,
        tau_f=tau_f,
        output_dim_per_session=output_dim
    )
    
    model.eval()
    
    inputs = torch.randn(batch_size, n_neurons, timesteps)
    targets = torch.randn(batch_size, output_dim, timesteps)
    target_provider = StandardTargetProvider(targets)
    loss_fn = torch.nn.MSELoss()
    
    test_loss, encoder_outputs, decoder_outputs = test_on_batch(
        sparks=model,
        inputs=inputs,
        target_provider=target_provider,
        loss_fn=loss_fn,
        test_loss=0,
        burnin=5
    )
    
    assert isinstance(test_loss.item(), float) or test_loss is None
    assert encoder_outputs.shape == (batch_size, latent_dim, tau_p + timesteps)
    assert decoder_outputs.shape[0] == batch_size


def test_test_dataloader():
    """Test evaluation on dataloader"""
    from sparks.utils.test import test
    from sparks.models.sparks import SPARKS
    from torch.utils.data import DataLoader, TensorDataset
    
    batch_size = 4
    n_neurons = 20
    timesteps = 30
    latent_dim = 32
    tau_p = 2
    tau_f = 1
    output_dim = 10
    n_samples = 12
    
    model = SPARKS(
        n_neurons_per_session=n_neurons,
        embed_dim=64,
        latent_dim=latent_dim,
        tau_p=tau_p,
        tau_f=tau_f,
        output_dim_per_session=output_dim
    )
    
    # Create mock dataset
    inputs = torch.randn(n_samples, n_neurons, timesteps)
    targets = torch.randn(n_samples, output_dim, timesteps)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    loss_fn = torch.nn.MSELoss()
    
    test_loss, encoder_outputs, decoder_outputs = test(
        sparks=model,
        test_dls=[dataloader],
        loss_fn=loss_fn,
        device='cpu',
        session_ids=[0],
        burnin=5
    )
    
    assert isinstance(test_loss.item(), float) or test_loss.item() == 0
    assert encoder_outputs.shape[0] == n_samples
    assert decoder_outputs.shape[0] == n_samples