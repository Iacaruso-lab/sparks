# Changelog

All notable changes to sparks-ai are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). While the version is below 1.0, a minor
version (0.x.0) can break the API.

## [0.3.0] - 03/11/2026

0.3.0 is a breaking release. The model now processes whole sequences instead of one timestep at a
time, and the package has been reorganised. Models trained with 0.2.x cannot be loaded, and code
written for 0.2.x has to be updated as described in [Migrating from 0.2.x](#migrating-from-02x).
To keep running 0.2.x code, pin `sparks-ai==0.2.1`.

### Added

- **Whole sequences in one pass.** `SPARKS.forward` takes `[batch, time, neurons]` spikes and
  returns the latents and predictions at every timestep. The Hebbian (STDP) traces are computed
  with a parallel scan (see paper Methods) rather than a Python loop over time.
- **Long recordings in chunks.** `chunk_length` (in `train`, `evaluate` and their per-batch
  versions) processes a recording in consecutive chunks. The recurrent state carries over from one
  chunk to the next (truncated backpropagation through time), so memory use stays flat. The
  lower-level API is `carry_state=True` and `reset_state()`.
- **Sequential KL.** With `kl_type='sequential'`, the latents are regularised toward a learned
  autoregressive prior p(z_t | z_{t-1}) (`GaussianTransition`), instead of an i.i.d. standard
  Gaussian. It has a free-bits floor (`free_bits`).
- **Data:**
  - Every loader returns a `DataSplit(train_dataset, test_dataset, train_loader, test_loader)`.
  - Every dataset subclasses `BaseDataset`, which has `n_neurons` and `dt` (in seconds).
  - `scripts/preprocess_allen.py` builds the files the Allen loaders read.
- **Packaging:**
  - Optional extras: `allen`, `nlb`, `cebra`, `all`, `demos` and `dev`. A missing optional
    dependency gives an error naming the extra to install.
  - `import sparks` gives `SPARKS`, `HebbianAttentionConfig`, `SharedConfig`, `train`,
    `evaluate` and `__version__`.
  - A test suite and continuous integration (Python 3.9 to 3.13).
- **Example notebooks** in `examples/`: MEC (unsupervised), Area2_Bump (behaviour
  decoding) and Allen multi-session (movie frame decoding, adding a session).

### Changed

- Spikes are `[batch, time, neurons]` (time first) everywhere: datasets, model and training. They
  used to be `[batch, neurons, time]`.
- `evaluate` always returns one tensor per session, even for a single session.
- `SharedConfig` (formerly `AttentionConfig`) defaults to 4 heads.
- New sessions' parameters, including decoder heads, are created on the model's device and in its
  dtype, and initialised like the ones built with the model.

### Removed
- `ProjectionConfig`, `projection_config` and `share_projection_head`.
- Target providers (`sparks.data.providers`): datasets return their targets directly.

### Fixed
- Allen gratings preprocessing: the trials of each condition are found correctly when there are
  several stimulus types. Condition ids used to be overwritten from one stimulus type to the next.
- MEC: `spikes_downsample` keeps the spikes of the first time bin.

### Migrating from 0.2.x

**Model**

| 0.2.x | 0.3.0 |
| --- | --- |
| `SPARKS(..., attention_config=AttentionConfig())` | `SPARKS(..., shared_config=SharedConfig())` |
| — | `SPARKS(..., bottleneck_dim=...)` (required) |
| `hebbian_config` as a list, one per session | One `HebbianAttentionConfig`. Per-session settings go through `add_session(..., tau_s=...)`. |
| `HebbianAttentionConfig(w_start=...)` | Removed. `w_plus` sets the scale of the STDP trace weights. |
| `model(x_t, encoder_outputs, session_id)`, called at each timestep on `[B, N]` spikes | `model(x, session_id)`, called once on `[B, T, N]` spikes. Returns `(z, decoder_outputs, mu, logvar)`, each `[B, T, ...]`. |
| `add_session(n_neurons, session_id, hebbian_config=..., new_output_dim=...)` | `add_session(n_neurons, session_id, new_output_dim=None, **hebbian_overrides)` |
| `encoder.add_neural_block(...)` | `model.add_session(...)` |
| `sparks.models.decoders.mlp` / `linear` | `MLPDecoder` / `LinearDecoder` (also `TransformerDecoder`) |
| Decoder `forward(x, sess_id)` | Decoder `forward(x, session_id)` |
| `DenseHebbianAttentionLayer` | `HebbianAttentionLayer` |

**Modules**

| 0.2.x | 0.3.0 |
| --- | --- |
| `sparks.models.dataclasses` | `sparks.models.configs` (or `import sparks`) |
| `sparks.models.blocks` | `sparks.models.hebbian_block`, `sparks.models.perceiver`, `sparks.models.layers` |
| `sparks.models.attention` (module) | `sparks.models.attention` (package: `base`, `ephys`, `calcium`) |
| `sparks.utils.train.train`, `train_on_batch` | `sparks.train`, `sparks.training.train_on_batch` |
| `sparks.utils.test.test`, `test_on_batch` | `sparks.evaluate`, `sparks.training.evaluate_on_batch` |
| `sparks.utils.losses` | `sparks.training.losses` (losses) and `sparks.training.metrics` (`bits_per_spike`) |
| `sparks.data.misc.LongCycler` | `sparks.training.cyclers.LongCycler`, which yields `(loader_idx, batch)` |

**Training**

| 0.2.x | 0.3.0 |
| --- | --- |
| `train(sparks=model, train_dls=..., loss_fn=..., optimizer=..., beta=..., device=...)` | `train(model, train_dls, loss_fn, optimizer, beta=..., scheduler=...)` |
| `test(sparks=model, test_dls=..., loss_fn=..., act=...)` | `evaluate(model, test_dls, loss_fn, act=...)`. `act` is applied before the loss, so pair `act=torch.sigmoid` with a loss on probabilities, or leave `loss_fn` out. |
| `test_on_batch(..., test_loss=...)` (accumulates) | `evaluate_on_batch(...)`, which returns this batch's loss |
| `update_and_reset(model, loss, optimizer)` | `loss.backward()`, then `clip_and_step(model, optimizer)` |
| `tau_f=...` passed to the training functions | Read from the model |

**Data**

| 0.2.x | 0.3.0 |
| --- | --- |
| `make_mec_ca_dataset` | `sparks.data.mec.make_mec_data` (held-out neurons: `dataset.left_out_neurons`) |
| `sparks.data.nlb.monkey_reaching.make_monkey_reaching_dataset` | `sparks.data.nlb.area2_bump.make_area2_bump_data` |
| `MonkeyReachingDataset` | `sparks.data.nlb.area2_bump.Area2BumpDataset` |
| `gratings_pseudomouse.make_gratings_dataset` | `sparks.data.allen.make_pseudomouse_gratings_data` |
| `gratings_multisess.make_gratings_dataset` | `make_session_gratings_data`, `make_session_spontaneous_data` (one `DataSplit` per session) |
| `make_pseudomouse_allen_movies_dataset_and_dls` | `make_pseudomouse_movies_data`, `make_calcium_movies_data` |
| `make_allen_movies_dataset(session_idxs=...)` | `make_session_movies_data(session_ids=...)`: Allen session ids, see `get_session_ids_with_data` |
| `neuron_types`, `correct_units_ids` | `areas`, `units_ids` |
| `data_dir`, `dataset_path`, `manifest_dir`, ... | `data_path` |
| `dataset.x_shape`, `num_neurons` / `y_shape` | `dataset.n_neurons` / `dataset.target_dim` |
| `units_per_neuron_type` | `dataset.units_per_area` |
| `sparks/data/allen/preprocess/scripts/*.py` | `python scripts/preprocess_allen.py data_path {gratings_flashes,spontaneous,natural_movie_one}` |

The Allen movie datasets' `mode` defaults to `'prediction'` (the frame index at each timestep). The
pseudo-mouse loaders' `block` defaults to `'first'`. Gratings `'freq'` and `'unsupervised'`
targets are `[T, F]`.

#### From the pre-release development code

Code written against the unreleased development branch (e.g. `sparks-controls`) also needs these
renames:

| Development code | 0.3.0 |
| --- | --- |
| `ConvConfig`, `conv_config` | `SharedConfig`, `shared_config` |
| `sparks.models.controls` | Moved out of the package: the `controls` package of the [sparks-controls](https://github.com/Iacaruso-lab/sparks-controls) repository |
| `ControlBlockBase` | `FrontEndBlock` |
| `sparks.models.attention.scan` | `sparks.models.scan` |
| `FullCalciumAttentionLayer`, `LightCalciumAttentionLayer` | `CalciumAttentionLayer` (the calcium rule under *Changed*), with no `n_timescales` |
| `add_hebbian_layer` | `add_session` |
| `train_on_batch_and_record_grad_norm(...)` | `train_on_batch(..., return_grad_norms=True)` |
| `sparks.utils.losses.bits_per_spike`, `evaluate_cosmoothing` | `sparks.training.metrics` |
| `train_on_batch(..., tau_f=..., device=..., mode=...)` | Removed: `tau_f` is read from the model, and the model trains on its own device |


## [0.2.1] - 2025-09-19

### Fixed

- `HebbianAttentionConfig` and `AttentionConfig` fields (e.g. `tau_s`, `dt`) are passed to the
  blocks. They used to be ignored unless given through `params`.
- The Hebbian attention layer raises an error on NaN coefficients.

## [0.2.0] - 2025-09-08

### Changed

- New API: the encoder and decoder no longer have to be defined separately.
- Custom decoders, and custom projection layers in the encoder, are easier to plug in.
- Configurations of the Hebbian attention layers are more flexible.
- Softly bounded STDP, which is much more stable on long recordings.

### Added

- A demo notebook for training one model on several recordings.

[0.3.0]: https://pypi.org/project/sparks-ai/0.3.0/
[0.2.1]: https://pypi.org/project/sparks-ai/0.2.1/
[0.2.0]: https://pypi.org/project/sparks-ai/0.2.0/
