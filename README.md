<div align="center">

<img src="https://raw.githubusercontent.com/FrancisCrickInstitute/sparks/main/logo.png" width="95%" alt="SPARKS">

[📚 Documentation](https://franciscrickinstitute.github.io/sparks-ai/docs#) |
[💡 Demos](https://franciscrickinstitute.github.io/sparks-ai/demos#) |
[🌎 Home page](https://franciscrickinstitute.github.io/sparks-ai/) |
[🪲 Report an issue](https://github.com/FrancisCrickInstitute/sparks/issues)

</div>

# SPARKS 🎆

**SPARKS** (**S**equential **P**redictive **A**utoencoder for the **R**epresentation of
spi**K**ing **S**ignals) learns low-dimensional latent representations of neural recordings.

Its encoder is built around a Hebbian attention mechanism: the attention between two neurons
follows spike-timing-dependent plasticity (STDP), so that it tracks their correlations as they
unfold, spike by spike. The latents are stochastic, and the model is trained to predict the next
time steps of a target from the latest ones:

- **Unsupervised:** with the recording itself as the target, SPARKS is a variational
  autoencoder.
- **Supervised:** with a behavioural variable or a stimulus as the target, it is a variational
  information bottleneck.

One model can embed several recording sessions, with different numbers of neurons, in a single
latent space. It works with electrophysiology (spikes) and calcium imaging data.

`sparks-ai` is implemented in [PyTorch](https://pytorch.org/).

> **0.3.0 is a breaking release.** See the [CHANGELOG](https://github.com/FrancisCrickInstitute/sparks/blob/main/CHANGELOG.md)
> for what's new and how to update 0.2.x code.

## Installation

```bash
pip install sparks-ai
```

This installs SPARKS with PyTorch. To use a GPU, install the PyTorch build that matches your CUDA
version first, following [pytorch.org](https://pytorch.org/get-started/locally/).

Optional extras add the dependencies of the data loaders and the examples:

| Extra | For |
| --- | --- |
| `nlb` | Neural Latents Benchmark loaders (`sparks.data.nlb`) |
| `allen` | Allen Brain Observatory preprocessing and per-session loaders (`sparks.data.allen`) |
| `cebra` | Allen calcium-imaging movie data, fetched through [CEBRA](https://cebra.ai/) |
| `demos` | Running the example notebooks |
| `all` | `nlb`, `allen` and `cebra` |

For example, `pip install "sparks-ai[nlb,demos]"`.

NEW: Conventional attention can be replace with [Mamba](https://github.com/state-spaces/mamba) blocks (unpublished) - (`sparks.models.MambaBlock`) need the
`mamba-ssm` package, which is compiled against your PyTorch and CUDA. Install `ninja`, then run
`pip install causal-conv1d mamba-ssm --no-build-isolation`.

## Quickstart

SPARKS takes spikes as `[trials, time, neurons]` tensors. Here it learns to decode a (random)
2-dimensional behaviour from 50 neurons:

```python
import torch
import sparks

# 32 trials of 200 time bins (1 ms) from 50 neurons: [trials, time, neurons]
spikes = (torch.rand(32, 200, 50) < 0.05).float()
behaviour = torch.rand(32, 200, 2)  # e.g. a 2D position in [0, 1] at each time bin
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(spikes, behaviour), batch_size=8, shuffle=True)

model = sparks.SPARKS(n_neurons_per_session=50,
                      embed_dim=32,          # each neuron's embedding
                      latent_dim=2,          # the latent space
                      bottleneck_dim=4,      # latent slots the neurons are compressed into
                      hebbian_config=sparks.HebbianAttentionConfig(tau_s=0.5, dt=0.001),  # STDP timescale and bin size, in s
                      output_dim_per_session=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(10):
    sparks.train(model, [loader], loss_fn, optimizer, beta=1e-3)  # beta weighs the KL term

_, latents, predictions = sparks.evaluate(model, [loader], act=torch.sigmoid)
print(latents[0].shape, predictions[0].shape)  # torch.Size([32, 200, 2]) twice: one per trial and time bin
```

To train on a GPU, pass `device=torch.device('cuda')` to `SPARKS`. To train without labels, pass
`unsupervised=True` to `train` and `evaluate`, and set `output_dim_per_session` to the number of
neurons. The [examples](#examples) show complete workflows.

## Data loaders

`sparks.data` has loaders for the datasets of the paper. Each returns a `DataSplit` with the train
and test datasets and their dataloaders.

| Data | Loader | Extra |
| --- | --- | --- |
| Medial entorhinal cortex calcium imaging ([Gonzalo Cogno et al., 2024](https://doi.org/10.1038/s41586-023-06864-1)) | `sparks.data.mec.make_mec_data` | |
| Area2_Bump: monkey reaching ([Neural Latents Benchmark](https://neurallatents.github.io/)) | `sparks.data.nlb.area2_bump.make_area2_bump_data` | `nlb` |
| Allen Visual Coding Neuropixels: gratings and flashes, spontaneous activity, natural movie | `sparks.data.allen` (`make_pseudomouse_*_data`, `make_session_*_data`) | `allen` |
| Allen calcium imaging, natural movie | `sparks.data.allen.make_calcium_movies_data` | `cebra` |

The Allen loaders read files prepared from an AllenSDK cache by
`python scripts/preprocess_allen.py`. See the `sparks.data.allen` module documentation for
details.

To use your own data, subclass `sparks.data.BaseDataset`, or pass any dataloader that yields
`(spikes, targets)` batches.

## Examples

The [`examples/`](https://github.com/FrancisCrickInstitute/sparks/tree/main/examples) notebooks
train SPARKS on real data:

- [`mec.ipynb`](https://github.com/FrancisCrickInstitute/sparks/blob/main/examples/mec.ipynb):
  unsupervised learning on medial entorhinal cortex recordings, recovering their slow oscillatory
  sequences.
- [`area2_bump.ipynb`](https://github.com/FrancisCrickInstitute/sparks/blob/main/examples/area2_bump.ipynb):
  decoding a monkey's hand position from somatosensory cortex.
- [`allen_multisession.ipynb`](https://github.com/FrancisCrickInstitute/sparks/blob/main/examples/allen_multisession.ipynb):
  one model for several mice, decoding which movie frame they see, then adding a new mouse.

## Reproducing the paper

The scripts that reproduce the results of the paper are in
[sparks-examples](https://github.com/Iacaruso-lab/sparks-examples), and those of its ablation controls in
[sparks-controls](https://github.com/Iacaruso-lab/sparks-controls).

## Citation

If you use SPARKS, please cite:

```bibtex
@article{skatchkovsky2024sparks,
  title   = {A Biologically Inspired Attention Model for Neural Signal Analysis},
  author  = {Skatchkovsky, Nicolas and Glazman, Natalia and Egea-Weiss, Alexander and Sadeh, Sadra and Iacaruso, Florencia},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {10.1101/2024.08.13.607787}
}
```

## Contributing

Contributions are welcome! See
[CONTRIBUTING.md](https://github.com/FrancisCrickInstitute/sparks/blob/main/CONTRIBUTING.md) to
set up a development environment and run the tests.

## License

SPARKS is free software, released under the
[GNU General Public License v3.0 or later](https://github.com/FrancisCrickInstitute/sparks/blob/main/LICENSE.md).
