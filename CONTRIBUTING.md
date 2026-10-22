# Contributing to sparks-ai

Thanks for helping improve SPARKS! This page covers setting up a development environment, the
layout of the code, the conventions we follow, and how a release is made. For bugs and feature
requests, please [open an issue](https://github.com/FrancisCrickInstitute/sparks/issues).

## Development setup

```bash
git clone https://github.com/FrancisCrickInstitute/sparks.git
cd sparks
pip install -e ".[dev,all]"   # `all` adds the data loaders' optional dependencies
pre-commit install            # runs the linter and strips notebook outputs on every commit
```

PyTorch is a regular dependency. To install a specific CUDA or CPU build, install `torch` first,
following [pytorch.org](https://pytorch.org/get-started/locally/).

## Running the tests

```bash
pytest                          # the whole suite, about a minute on a CPU
pytest tests/test_models        # one directory
pre-commit run --all-files      # the same checks as CI's lint job
```

CI runs the tests on Python 3.9 to 3.13 with a CPU build of PyTorch and the `dev` and `nlb`
extras. Tests that need an optional package (e.g. pandas) are skipped when it is not installed;
the data-loader tests use small fake datasets, not the real data.

**Golden tests.** `tests/test_golden.py` checks that small reference models still produce the
outputs stored in `tests/golden_outputs.pt`. Each reference model's weights are loaded with
`strict=True`, so the golden tests also fail if a parameter is renamed or reshaped, which would
stop existing checkpoints from loading. If you change the model's outputs on purpose (e.g. a bug
fix), regenerate the reference with `python tests/golden_cases.py` and say why in the commit
message. Never regenerate it to make a failing test pass without understanding why it fails.

**Known bugs.** A test for a bug that is not fixed yet can be marked
`@pytest.mark.xfail(strict=True, reason=...)`. Because xfail is strict, the test fails as soon as
the bug is fixed, as a reminder to remove the marker.

## Code layout

```
sparks/
├── models/          # the model
│   ├── sparks.py        # SPARKS: encoder + decoder, sessions, carried state
│   ├── encoders.py      # HebbianEncoder: front end -> shared blocks -> Gaussian latents
│   ├── hebbian_block.py # HebbianAttentionBlock: per-session Hebbian layers + shared readout
│   ├── front_end.py     # FrontEndBlock interface and the shared Perceiver readout
│   ├── perceiver.py     # PerceiverBlock: neurons -> bottleneck_dim latent slots
│   ├── attention/       # Hebbian attention layers: ephys (STDP) and calcium
│   ├── layers.py        # generic blocks: attention, Mamba, feed-forward
│   ├── decoders.py      # MLP, linear and transformer decoders
│   ├── configs.py       # HebbianAttentionConfig, SharedConfig
│   ├── scan.py          # parallel affine scan used by the recurrent layers
│   └── state.py         # CarriedState: recurrent state carried across chunks
├── training/        # train/evaluate loops, losses, metrics, optimiser helpers
└── data/            # BaseDataset, DataSplit, and loaders for MEC, NLB and Allen data
scripts/             # command-line preprocessing (Allen)
examples/            # example notebooks
tests/               # mirrors the package layout
```

## Conventions

- **Docstrings** use the Google style (`Args:`, `Returns:`), in the present tense, and give tensor
  shapes, e.g. `[B, T, N]` for batch, time and neurons. Spikes are always `[B, T, N]`.
- **Comments** explain why the code is the way it is, not how it came to be: leave history to git
  and the CHANGELOG.
- **Parameter names** are part of the checkpoint format. Renaming an `nn.Module` attribute
  breaks loading existing checkpoints, and the golden tests will tell you.
- **Optional dependencies** are imported through `sparks._optional.import_optional`, so a missing
  package gives an error naming the extra to install.
- **Linting** is ruff's pyflakes rules (`F`), run by pre-commit and CI. `sparks/data/nlb/nwb_interface.py`
  is vendored from `nlb_tools` and is kept as upstream wrote it.
- **Notebooks** are committed without outputs; the nbstripout pre-commit hook takes care of it.

### Adding a dataset

Subclass `sparks.data.BaseDataset` and implement `__len__`, `n_neurons`, `get_spikes(index)`
(a `[T, N]` tensor) and `get_target(index)`, and set `dt` (in seconds). Write a `make_*_data`
function that returns `sparks.data.make_split(train_dataset, test_dataset, ...)`, so that every
loader returns the same `DataSplit`.

### Adding a front end

A front end replaces the Hebbian attention block in the encoder. Subclass
`sparks.models.FrontEndBlock` and select it with `HebbianAttentionConfig(block_class=...)`.
`tests/front_ends.py` has a minimal example, and the `controls` package of the `sparks-controls`
repository (the ablations of the paper) has complete ones.

## Making a release

1. Update the version in `sparks/__init__.py` (`__version__`), which `pyproject.toml` reads.
2. Move the CHANGELOG's *Unreleased* entries under the new version and date.
3. Check that CI passes, then build and check the package:
   ```bash
   python -m build
   twine check --strict dist/*
   ```
4. Upload to TestPyPI first and install from there in a fresh environment:
   ```bash
   twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sparks-ai
   ```
5. Upload to PyPI (`twine upload dist/*`), then tag the release (`git tag v0.3.0`) and push the
   tag.
