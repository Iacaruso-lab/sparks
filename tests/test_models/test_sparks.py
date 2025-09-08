import torch


def test_sparks_model():
    """Test SPARKS model"""
    from sparks.models.sparks import SPARKS
    
    n_neurons_per_sess = 100
    embed_dim = 64
    latent_dim = 32
    tau_p = 3
    tau_f = 2
    output_dim_per_session = 10
    session_id = 23
    batch_size = 8
    timesteps = 50
    
    model = SPARKS(
        n_neurons_per_session=n_neurons_per_sess,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        tau_p=tau_p,
        tau_f=tau_f,
        id_per_session=[session_id],
        output_dim_per_session=output_dim_per_session
    )
    
    # Test forward pass
    x = torch.randn(batch_size, n_neurons_per_sess, timesteps)
    encoder_outputs = torch.zeros(batch_size, latent_dim, tau_p)
    
    # Run through multiple timesteps
    for t in range(timesteps - tau_f + 1):
        encoder_outputs, decoder_outputs, mu, logvar = model(
            x[..., t],
            encoder_outputs=encoder_outputs,
            session_id=session_id
        )
        
        assert encoder_outputs.shape == (batch_size, latent_dim, tau_p + 1)
        assert decoder_outputs.shape == (batch_size, output_dim_per_session * tau_f)
        assert mu.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        
        # Keep only last tau_p timesteps
        encoder_outputs = encoder_outputs[..., -tau_p:]
    
    # Test generate method
    z = torch.randn(batch_size, latent_dim * tau_p)
    generated = model.generate(z, session_id=session_id)
    assert generated.shape == (batch_size, output_dim_per_session * tau_f)


def test_sparks_multi_session():
    """Test SPARKS with multiple sessions"""
    from sparks.models.sparks import SPARKS
    
    n_neurons_per_sess = [100, 150]
    embed_dim = 64
    latent_dim = 32
    tau_p = 2
    tau_f = 1
    output_dim_per_session = [10, 20]
    id_per_sess = [0, 1]
    batch_size = 4
    
    model = SPARKS(
        n_neurons_per_session=n_neurons_per_sess,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        id_per_session=id_per_sess,
        tau_p=tau_p,
        tau_f=tau_f,
        output_dim_per_session=output_dim_per_session
    )
    
    # Test each session
    for sess_id, n_neurons, out_dim in zip(id_per_sess, n_neurons_per_sess, output_dim_per_session):
        x = torch.randn(batch_size, n_neurons)
        encoder_outputs = torch.zeros(batch_size, latent_dim, tau_p)
        
        encoder_outputs, decoder_outputs, mu, logvar = model(
            x,
            encoder_outputs=encoder_outputs,
            session_id=sess_id
        )
        
        assert decoder_outputs.shape == (batch_size, out_dim * tau_f)


def test_add_session():
    """Test adding a new session to SPARKS model"""
    from sparks.models.sparks import SPARKS
    
    n_neurons_per_sess = [50]
    embed_dim = 64
    latent_dim = 32
    tau_p = 2
    tau_f = 1
    output_dim_per_session = 10
    id_per_sess = [0]
    batch_size = 4
    
    model = SPARKS(
        n_neurons_per_session=n_neurons_per_sess,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        id_per_session=id_per_sess,
        tau_p=tau_p,
        tau_f=tau_f,
        output_dim_per_session=output_dim_per_session
    )
    
    # Add a new session
    new_n_neurons = 30
    new_out_dim = 20
    new_sess_id = 5
    model.add_session(new_n_neurons, new_sess_id, new_output_dim=new_out_dim)
    
    assert model.encoder.session_ids == [str(0), str(5)]
    
    # Test the new session
    x = torch.randn(batch_size, new_n_neurons)
    encoder_outputs = torch.zeros(batch_size, latent_dim, tau_p)
    
    encoder_outputs, decoder_outputs, mu, logvar = model(
        x,
        encoder_outputs=encoder_outputs,
        session_id=new_sess_id
    )
    
    assert decoder_outputs.shape == (batch_size, new_out_dim * tau_f)


def test_add_session_with_joint_decoder():
    """Test adding a new session with joint decoder to SPARKS model"""
    from sparks.models.sparks import SPARKS
    
    n_neurons_per_sess = [20, 30]
    embed_dim = 64
    latent_dim = 32
    tau_p = 2
    tau_f = 1
    output_dim_per_session = 10
    id_per_sess = [0, 1]
    batch_size = 4
    
    model = SPARKS(
        n_neurons_per_session=n_neurons_per_sess,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        id_per_session=id_per_sess,
        tau_p=tau_p,
        tau_f=tau_f,
        output_dim_per_session=output_dim_per_session,
        joint_decoder=True
    )
    
    # Add a new session
    new_n_neurons = 150
    new_sess_id = 2
    model.add_session(new_n_neurons, new_sess_id)
    
    assert model.encoder.session_ids == [str(0), str(1), str(2)]
    
    # Test the new session
    x = torch.randn(batch_size, new_n_neurons)
    encoder_outputs = torch.zeros(batch_size, latent_dim, tau_p)
    
    encoder_outputs, decoder_outputs, mu, logvar = model(
        x,
        encoder_outputs=encoder_outputs,
        session_id=new_sess_id
    )
    
    assert decoder_outputs.shape == (batch_size, output_dim_per_session * tau_f)

def test_custom_decoder():
    """Test SPARKS model with a custom decoder"""
    from sparks.models.sparks import SPARKS
    from sparks.models.decoders import linear
    
    n_neurons_per_sess = 50
    embed_dim = 64
    latent_dim = 32
    tau_p = 2
    tau_f = 1
    output_dim_per_session = 10
    session_id = 0
    batch_size = 4
    
    custom_decoder = linear(in_dim=latent_dim * tau_p,
        output_dim_per_session=[output_dim_per_session],
        id_per_session=[session_id]
    )
    
    model = SPARKS(
        n_neurons_per_session=n_neurons_per_sess,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        tau_p=tau_p,
        tau_f=tau_f,
        id_per_session=[session_id],
        output_dim_per_session=output_dim_per_session,
        decoder=custom_decoder
    )
    
    # Test forward pass
    x = torch.randn(batch_size, n_neurons_per_sess)
    encoder_outputs = torch.zeros(batch_size, latent_dim, tau_p)
    
    encoder_outputs, decoder_outputs, mu, logvar = model(
        x,
        encoder_outputs=encoder_outputs,
        session_id=session_id
    )
    
    assert decoder_outputs.shape == (batch_size, output_dim_per_session * tau_f)