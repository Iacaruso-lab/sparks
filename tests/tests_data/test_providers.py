import torch


def test_standard_target_provider():
    """Test StandardTargetProvider"""
    from sparks.data.providers import StandardTargetProvider
    
    # Create mock targets
    targets = torch.randn(10, 5, 100)  # batch_size=10, features=5, timesteps=100
    provider = StandardTargetProvider(targets)
    
    # Test get_target
    batch_idxs = [0, 1, 2]
    timestep = 10
    tau_f = 5
    device = 'cpu'
    
    result = provider.get_target(batch_idxs, timestep, tau_f, device)
    
    assert result.shape == (10, 25)  # 5 features * 5 timesteps
    assert result.device == torch.device('cpu')


def test_allen_movies_target_provider_prediction():
    """Test AllenMoviesTargetProvider"""
    from sparks.data.providers import AllenMoviesTargetProvider
    
    # Create mock frames
    frames = None
    dt = 0.01
    mode = 'prediction'

    provider = AllenMoviesTargetProvider(frames, mode, dt)
    
    batch_idxs = [0, 1]
    timestep = 100
    tau_f = 10
    device = 'cpu'
    
    result = provider.get_target(batch_idxs, timestep, tau_f, device)
    
    assert result.shape == (2,)  # batch size x tau_f
    assert result.device == torch.device('cpu')


def test_allen_movies_target_provider_reconstruction():
    """Test AllenMoviesTargetProvider"""
    from sparks.data.providers import AllenMoviesTargetProvider
    
    # Create mock frames
    frames = torch.randn(10, 15, 20)  # height, width, num_frames
    dt = 0.01
    mode = 'reconstruction'

    provider = AllenMoviesTargetProvider(frames, mode, dt)
    
    batch_idxs = [0, 1]
    timestep = 5
    tau_f = 10
    device = 'cpu'
    
    result = provider.get_target(batch_idxs, timestep, tau_f, device)
    
    assert result.shape == (2, tau_f * frames.shape[0] * frames.shape[1])  # batch size x tau_f
    assert result.device == torch.device('cpu')


def test_denoising_target_provider():
    """Test ShuffleTargetProvider"""
    from sparks.data.providers import ShuffleTargetProvider
    
    targets = torch.randn(10, 5, 100)
    provider = ShuffleTargetProvider(targets)
    
    batch_idxs = list(range(10))
    timestep = 10
    tau_f = 5
    device = 'cpu'
    
    result = provider.get_target(batch_idxs, timestep, tau_f, device)
    
    assert result.shape == (10, 25)
    assert result.device == torch.device('cpu')


