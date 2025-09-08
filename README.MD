<div align="center">


<p align="center">
<img src="logo.png" width="95%">
</p>

[📚Documentation](https://franciscrickinstitute.github.io/sparks-ai/docs#) |
[💡DEMOS](https://franciscrickinstitute.github.io/sparks-ai//demos#) |
[🛠️ Installation](https://franciscrickinstitute.github.io/sparks-ai/docs#) |
[🌎 Home Page](https://franciscrickinstitute.github.io/sparks-ai/) |
[🪲 Reporting Issues](https://github.com/FrancisCrickInstitute/sparks/issues)



</div>

# SPARKS 🎆

This is the official repository for **SPARKS**: a **S**equential **P**redictive **A**utoencoder for the **A**nalysis of spi**K**ing **S**ignals. 


SPARKS includes a novel self-attention mechanism using Hebbian learning to generate reliable latent representations from single spike timings. SPARKS trains a variational autoencoder with a novel criterion inspired by predictive coding for temporal coherence.

``sparks`` is implemented in [PyTorch](https://pytorch.org/) and includes [demos](https://github.com/FrancisCrickInstitute/SPARKS/sparks/scripts) for a quickstart.
It can perform supervised or unsupervised to produce low-dimensional latent embeddings which allows to gain biological insights from neural data.

Make sure to 👀 watch or ⭐️ star this repository to keep updated!

# New release v0.2.0!
- New API: no need to separately define the encoder and decoder anymore
- Easier prototyping without needing to dig into the implementation details
- Better support of custom architectures for the decoder and the projection layer of the encoder
- More flexible definition of configurations for different Hebbian attention layers
- Softly bounded STDP provides much more stable results for long recordings
- New demo notebook for training a single model with several recordings
- Beta: sliding-window attention to scale the model for modern large recordings

# Reference
- 📄 **Preprint**:

Available now on [BiorXiv](https://www.biorxiv.org/content/10.1101/2024.08.13.607787v2)!

- Scripts to reproduce paper results [here](https://github.com/Iacaruso-lab/sparks-examples)

# License
- SPARKS is an open source software under a GLPv3 license.
