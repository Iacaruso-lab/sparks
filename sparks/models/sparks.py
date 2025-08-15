from typing import Union, List, Optional, Tuple
import torch
import numpy as np

from sparks.models.encoders import HebbianTransformer
from sparks.models.dataclasses import HebbianAttentionConfig, AttentionConfig, ProjectionConfig
from sparks.models.decoders import mlp


class SPARKS(torch.nn.Module):
    """
    SPARKS model class.
    SPARKS is a VAE-based model designed for neural data.
    It consists of an encoder encompassing a Hebbian attention layer, and a decoder that can generally
    be any neural network architecture.
    """

    def __init__(self,
                 n_neurons_per_sess: Union[int, List[int]],
                 embed_dim: int,
                 latent_dim: int,
                 id_per_sess: Optional[List[Union[str, int]]] = None,
                 tau_p: int = 1,
                 tau_f: Optional[int] = 1,
                 hebbian_config: Union[HebbianAttentionConfig, List[HebbianAttentionConfig]] = HebbianAttentionConfig(),
                 attention_config: AttentionConfig = AttentionConfig(),
                 projection_config: ProjectionConfig = ProjectionConfig(),
                 share_projection_head: bool = False,
                 decoder: Optional[torch.nn.Module] = None,
                 output_dim_per_session: Optional[Union[int, List[int]]] = None,
                 device: torch.device = torch.device('cpu')):

        super().__init__()

        self.tau_p = tau_p
        self.tau_f = tau_f
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.device = device

        if isinstance(n_neurons_per_sess, int):
            n_neurons_per_sess = [n_neurons_per_sess]

        if id_per_sess is None:
            id_per_sess = [str(i) for i in range(len(n_neurons_per_sess))]
        elif len(id_per_sess) != len(n_neurons_per_sess):
            raise ValueError("`id_per_sess` must have the same length as `n_neurons_per_sess`.")

        self.encoder = HebbianTransformer(n_neurons_per_sess=n_neurons_per_sess,
                                          embed_dim=embed_dim,
                                          latent_dim=latent_dim,
                                          id_per_sess=id_per_sess,
                                          hebbian_config=hebbian_config,
                                          attention_config=attention_config,
                                          projection_config=projection_config,
                                          share_projection_head=share_projection_head,
                                          device=device)
        
        if decoder is None:
            n_inputs_decoder = latent_dim * tau_p
            hid_dims = [int(np.mean([n_inputs_decoder, np.mean(output_dim_per_session)]))]
            if output_dim_per_session is None:
                raise ValueError("`output_dim_per_session` must be provided if `decoder` is None.")
            if isinstance(output_dim_per_session, int):
                output_dim_per_session = [output_dim_per_session] * len(n_neurons_per_sess)
            output_dim_per_session = [dim * tau_f for dim in output_dim_per_session]

            decoder = mlp(in_dim=n_inputs_decoder, hidden_dims=hid_dims,
                          output_dim_per_session=output_dim_per_session,
                          id_per_sess=id_per_sess).to(device)

        self.decoder = decoder

    def forward(self, x: torch.Tensor, encoder_outputs, sess_id: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                           torch.Tensor, torch.Tensor]:
        """
        The forward pass of the autoencoder.

        The steps of the forward pass are:
        1. Compute the latent representation of the input.
        2. Stack the latent representation in the encoder_outputs buffer of length tau_p.
        3. Obtain the prediction of the decoder.

        Args:
            inputs (torch.tensor): The input data tensor.
            encoder_outputs (torch.tensor): The collection of past encoder output tensors.
            sess_id (int, optional): The session id. Default is 0.

        Returns:
            encoder_outputs (torch.tensor): The encoder outputs tensor with the newly computed encoder output appended.
            decoder_outputs (torch.tensor): The output tensor of the decoder.
            mu (torch.tensor): The mean of the latent distribution computed by the encoder.
            logvar (torch.tensor): The log-variance of the latent distribution computed by the encoder.
        """

        mu, logvar = self.encoder(x.view(x.shape[0], -1).float().to(self.device), sess_id)
        enc_outputs = self.encoder.reparametrize(mu, logvar)

        encoder_outputs = torch.cat((encoder_outputs, enc_outputs.unsqueeze(2)), dim=2)
        decoder_outputs = self.decoder(encoder_outputs[..., -self.tau_p:], sess_id)

        return encoder_outputs, decoder_outputs, mu, logvar
        

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generates an output from a given latent vector `z`.

        This method is useful for exploring the latent space by bypassing the encoder.

        Args:
            z (torch.Tensor): A tensor representing a point (or batch of points)
                              in the latent space.

        Returns:
            The output generated by the decoder.
        """
        return self.decoder(z)