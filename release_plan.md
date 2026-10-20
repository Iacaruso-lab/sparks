# Release review: SPARKS

The core model code is in good shape, but the release around it is not. The new shared Hebbian block works, and running chunked with `carry_state` gives the same result as a full pass (I checked this numerically). What's broken is what users actually touch:

- **Packaging:** the 0.2.1 PyPI page has an empty description, and installing the wheel adds stray top-level `tests`, `conda` and `dist` folders to `site-packages`.
- **Notebooks:** all three demos fail at their first cell.
- **Tests:** 19 of 29 fail. 17 of those are written against an old API; 2 hit real bugs.
- **Code:** a few code paths are broken outright.

I checked the high-severity items below by running code; the rest come from reading it. I didn't change anything during this review.

## Decisions you need to make first

1. **Version:** release as **0.3.0**. It already breaks the 0.2.1 API: `bottleneck_dim` is now required, providers are gone, and `AttentionConfig` became `ConvConfig`. That needs a changelog with a migration section.
2. **`HebbianTransformer`** ([transformer.py](sparks/models/transformer.py)): it can't be constructed at all, even with its own arguments (`TypeError: n_neurons`). But `sparks-examples` imports it 14 times, and the README points to that repo for reproducing the paper. I recommend deleting it and pinning `sparks-examples` to `sparks-ai==0.2.1`; that repo also still imports `AttentionConfig` and `sparks.scripts.*`.
3. **Sliding-window ("sparse") attention:** it's broken in every path (TypeError through the config, AttributeError and a shape mismatch when called directly), yet the README advertises it as Beta. I recommend removing it together with the `sliding`, `window_size` and `block_size` settings.
4. **Renames:** I'd do a clean break for things that are already broken, and keep deprecated aliases for one release for renames of things that work today.

---

## Part A: Findings

### A1. Bugs that must be fixed before release (all verified)

| Where | Bug |
|---|---|
| [train.py:234](sparks/utils/train.py#L234) | With `kl_type='sequential'`, the `GaussianTransition` gradients are **never zeroed**, because `model.zero_grad()` doesn't reach a module outside `model`. They grow every step (0.10 → 0.19 → 0.26 on the same batch) and are never clipped. Fix: use `optimizer.zero_grad(set_to_none=True)` and clip the transition too. |
| [encoders.py:139-142](sparks/models/encoders.py#L139-L142) | `add_session(hebbian_config=HebbianAttentionConfig(tau_s=0.5))` resets every per-session field to its default, because `__post_init__` fills them all in. A calcium/full model gets an ephys/light layer for the new session. |
| [sparks.py:248-254](sparks/models/sparks.py#L248-L254) | `add_session`: `new_output_dim` isn't multiplied by `tau_f`, although the decoder heads built in `__init__` are. A missing `new_output_dim` gives `nn.Linear(in, None)`, and a custom decoder without a `joint_decoder` attribute raises AttributeError. |
| [data/allen/movies.py:55-56](sparks/data/allen/movies.py#L55-L56) | The `'prediction'` and `'reconstruction'` targets return the trial index, not per-timestep frame indices. With providers removed, nothing in the package produces frame targets anymore. |
| [data/allen/movies_multisess.py:60-65](sparks/data/allen/movies_multisess.py#L60-L65) | `mode` is never passed to the dataset, so it only changes the train/test split. |
| [data/allen/movies_pseudomouse.py:168](sparks/data/allen/movies_pseudomouse.py#L168) | The default `block='within'` raises `NotImplementedError`. |
| [data/allen/utils.py:65](sparks/data/allen/utils.py#L65) | `hasattr(x, '__iter__')` is true for strings, so `neuron_types='VISp'` is iterated letter by letter. |
| `data/nlb/monkey_reaching.py:157` | `smooth=True` raises TypeError (`torch.concatenate` on numpy arrays) and would smooth across neurons rather than time. |
| `data/misc.py` | `RandomCycler` raises NameError (`random` isn't imported). `Exhauster.__len__` and `MultiSessionDataset` are wrong. `normalize` *returns* `NotImplementedError` instead of raising it. |
| [data/base.py:7](sparks/data/base.py#L7) | `BaseDataset` doesn't inherit from `ABC`, so `@abstractmethod` is never enforced. `super(BaseDataset).__init__()` here and in 4 subclasses does nothing. |
| Calcium attention | Calcium layers never read `tau_s`, so a calcium model with several `tau_s` heads gets heads that differ only in their random init. Ephys silently drops `min/max_attn_value` and `n_timescales` (`**kwargs` swallowed in `attention/base.py`). |

### A2. Packaging and licensing

- **The PyPI page is blank.** [pyproject.toml](pyproject.toml) points to `README.md` and `LICENSE.md`, but the files are `README.MD` and `LICENSE.MD`, and this filesystem is case-sensitive. The 0.2.1 PyPI page has an empty description and no licence.
- **The wheel ships things it shouldn't:**
  - Top-level `conda`, `dist` and `tests` packages.
  - 1.5 MB of notebooks; without them the wheel is 122 KB.
  - The sdist also contains `.pypirc`. It holds only URLs, but it's tracked even though it's in `.gitignore`.
  - Cause: `find = {}` with namespace discovery, plus `setuptools-scm` listed as a build requirement while the version is hard-coded.
  - `sparks/models/attention/` has no `__init__.py` and only ships thanks to that namespace discovery.
- **Dependencies are wrong:**
  - `torch` is unpinned, but `nn.RMSNorm` needs `>=2.4`.
  - `allensdk`, `cebra` and `mamba-ssm` aren't declared at all.
  - `pynwb` and `pandas` are core dependencies but only `nlb/` uses them.
  - `allensdk` is imported at module level, so any `sparks.data.allen.*` import fails without it.
  - `mamba_ssm` is imported eagerly, which costs about 16 s per test run and crashes on a broken CUDA build (OSError isn't caught).
- [conda/meta.yaml](conda/meta.yaml) is malformed: it says version 0.1.0, references a `setup.py` that doesn't exist, and has the wrong name. Delete it, or regenerate it with grayskull.
- **Licensing:** [nlb/nwb_interface.py](sparks/data/nlb/nwb_interface.py) is a verbatim copy of `nlb_tools` 0.0.4, which is MIT-licensed, with no copyright notice. `data/misc.py:25` ("adapted from neuralpredictors") has the same problem.
- **README:** the demos link returns 404, "GLPv3" is a typo, the "v0.2.0" release notes are stale, there are no install instructions, and the acronym doesn't match the one in pyproject. It also says "variational autoencoder" where your notes say VIB.
- **Missing files:** CI, CHANGELOG, CONTRIBUTING, CITATION.cff, a pre-commit/ruff config and a pytest config.
- **`.gitignore`:** the global `*.sh` and `*.html` patterns will silently swallow CI and docs files.
- **Git history:** the pack is 78 MB, mostly old wheels, `.npy` files and notebooks.
- `docs/methods_technical_report.*` is deleted in your working tree but still tracked; decide which way you want it.

### A3. Organisation

- **No public API.**
  - `sparks/__init__.py` and `sparks/models/__init__.py` are empty, and there's no `__version__`.
  - Users have to know deep paths like `sparks.models.dataclasses`.
- **[blocks.py](sparks/models/blocks.py) does two jobs:**
  - the session-aware Hebbian front end (`HebbianAttentionBlock`, `PerceiverBlock`, `CONFIG_DICT`);
  - generic backbone layers (`AttentionBlock`, `MambaBlock`, RoPE attention).

  Meanwhile `FeedForward`, `DropPath` and `SwiGLU` sit in `models/utils.py`.
- **Confusing names:**
  - `models/dataclasses.py` echoes the standard-library module name; call it `configs.py`.
  - `ConvConfig` configures *conventional* attention but reads as *convolution*; call it `BackboneConfig`.
  - **`ProjectionConfig` does nothing.** `projection_config` is stored but never read, even though the README advertises custom projection heads.
- **`sparks/utils/test.py` is a problem:**
  - It defines `test` and `test_on_batch`, which pytest collects as tests whenever a test module imports them.
  - `test()` is also a confusing name; rename the module to `evaluate.py`.
- **`sparks/utils/losses.py` mixes training losses with evaluation metrics** (`bits_per_spike`, `evaluate_cosmoothing`, which also pull in sklearn). Split it into `losses.py` and `metrics.py`.
- `sparks/utils/misc.py` imports `SPARKS` and `HebbianTransformer` without using them.
- **The data layer is tangled:**
  - Training imports `LongCycler` from `sparks/data/misc.py`, and depends on the order it yields batches in (`random_order[i % n]` in `train()`).
  - `misc.py` also mixes array helpers with mostly dead loader cyclers.
- **Things that shouldn't be inside the package:**
  - `sparks/examples/*.ipynb`.
  - `data/allen/preprocess/scripts/*.py`: bare scripts that run on import and hardcode `/nemo/lab/...`.
- **Allen naming:** `gratings_pseudomouse.py` and `gratings_multisess.py` both define `make_gratings_dataset`, with different signatures and return lengths. The "gratings" loader in the multisession file defaults to spontaneous activity.

### A4. Function design and maintainability

**Training and evaluation ([train.py](sparks/utils/train.py), [test.py](sparks/utils/test.py))**

- **Options come in through `**kwargs` + `kwargs.get(...)`, so a typo is silently ignored:**
  - `online` is documented but doesn't exist.
  - `cd_mask`, `scheduler` and `unsupervised` exist but aren't documented.
  - `device` is accepted and never used.
  - `burnin` is documented in evaluation but never applied.
  - The `max_grad_norm` default is 5.0 in one place, 1.0 in another, and "1.0" in the docs.

  Make these explicit keyword-only arguments.
- **Duplication:**
  - `train_on_batch_and_record_grad_norm` copies `train_on_batch` plus an inlined copy of `update_and_reset`, which already returns the gradient norm. Replace it with a `return_grad_norm` flag.
  - There are three learning-rate schedule functions; `sparks-examples` only uses `get_lr_multiplier`.
  - The `tau_f` unfolding and the masked-loss code are each copied 3–4 times. Extract two helpers.
- **`test()` problems:**
  - It returns a tensor for one session but a list for several.
  - It builds outputs with repeated `torch.cat`, which is quadratic.
  - `if encoder_outputs is not None` is always true.
  - The docstring for the chunked path describes truncation that the code doesn't do.

**Model API ([sparks.py](sparks/models/sparks.py), [encoders.py](sparks/models/encoders.py))**

- `self.device` is a plain attribute, so it goes stale after `.to()`.
- The config defaults (`HebbianAttentionConfig()` in function signatures) are created once and shared between calls.
- A list of `hebbian_config` is accepted, but only the first item is used, silently.
- The `forward` and `HebbianEncoder` docstrings describe old arguments (`encoder_outputs`, `attention_config`).
- `HebbianEncoder(id_per_session=None)` crashes, although `None` is the documented default.
- The `TransformerEncoder` class is unused.
- In `pad_z` the comment says the window is reordered to time-major, but the permute is commented out. Fix the comment, not the code: changing the code would change behaviour and break checkpoints.

**Model internals**

- **Carried state is stored on the modules (`_carried_state`) in about 6 places:**
  - It's written on every forward, even with `carry_state=False`, which keeps the last batch's B×N×N tensors in memory.
  - It isn't in `state_dict`.
  - `.to()` doesn't move it.
  - It isn't safe under DataParallel.

  Move it into one `StatefulMixin` that only stores state when `carry_state=True`.
- **Duplication:**
  - Ephys Full and Light repeat both the sequential and the parallel paths almost line for line.
  - Calcium Full is literally `pass`, and calcium has a sequential path that duplicates `clamped_affine_scan`.
  - `CoefficientAttentionBlock` re-implements the head mixing, feed-forward and Perceiver path of `HebbianAttentionBlock`, so the ablation can drift away from the model it controls.
- **Inconsistent names and defaults:**
  - The method that adds a session has three names: `add_hebbian_layer`, `add_session` and `add_neural_block`.
  - `d_model` vs `embed_dim`, and `sess_id` vs `session_id`.
  - `config` defaults to `'full'` in the block but `'light'` in the config.
  - The decoder classes are lowercase: `mlp`, `linear`, `transformer`.
  - The ephys clamp `(-0.5, 1.5)` is hardcoded 4 times.
- **No shared interface:** control blocks have no `hebbian_forward`, so `SPARKS.hebbian_forward` and `get_hebbian_coeffs_on_batch` fail with them. A `Protocol` both block types implement would fix this.

**Data**

- **The data factories have no common interface:**
  - They return 4 or 5 values, in two different orders.
  - The path argument has four names (`data_dir`, `output_dir`, `spikes_file`, `dataset_path`).
  - Layouts differ: calcium is `[N,T]` while the rest are `[T,N]`, and gratings `freq` targets are `(F,T)`.
  - Time units differ: `dt` in seconds, `fs` in Hz with a default of 0.01, and fixed 1 ms bins.
- **Constructors have hidden side effects:**
  - Some download multi-GB NWB files from inside dataset factories.
  - Some call `np.random.seed` globally.
- **The on-disk file layout the loaders expect isn't documented anywhere.** Nothing in the repo produces the gratings pickles; the code that would is commented out.

**Docstrings**

- There are three styles (Google, NumPy, reST `:param`).
- Many docstrings tell the history ("as before", "unchanged behavior", "used to sum…") instead of describing current behaviour.
- About a dozen are stale after recent changes, for example the PerceiverBlock comments about "cross-attention", and ephys attributes that no longer exist.

### A5. Tests and examples

- **Tests:**
  - 17 tests are stale against the old API and 2 fail on the real sparse bug.
  - `tests/test_models/test_blocks` has no `.py` extension, so it's never collected.
  - There's no `conftest.py`, no fixtures and no markers.
  - The Allen tests error instead of skipping when `allensdk` is missing.
- **Behaviour that works but has no tests:**
  - parallel scan vs a sequential loop;
  - chunked `carry_state` vs a full pass;
  - `add_session`;
  - the control blocks;
  - `build_param_groups` and gradient clipping;
  - `TemporalCrossEntropyLoss`.

  The review agent's probe scripts already cover most of these and can be reused (paths in Phase 2 below).
- **Notebooks: all three break at cell 0.**
  - Shared causes: `AttentionConfig`, a missing `bottleneck_dim`, the `sparks=` keyword (now `model=`), and the `[B,N,T]` layout.
  - `demo_monkey` re-implements the NLB dataset inline instead of using the packaged one.
  - Their saved outputs come from older code.

---

## Part B: Plan

The order matters. Write the regression tests **before** restructuring, so every move after that is checked.

**Phase 0: Set up (½ day)**
- Commit the current WIP. Branch `release/0.3.0`. Settle the four decisions above.
- Agree on two rules up front:
  - Parameter names inside modules stay stable, so existing checkpoints keep loading.
  - Every rename keeps a deprecated alias for one release.

**Phase 1: Packaging and hygiene (½–1 day)**
- `git mv` README and LICENSE to lowercase `.md`. Use `license = "GPL-3.0-or-later"`.
- Set `[tool.setuptools.packages.find] include=["sparks*"]`. Add the missing `__init__.py` files.
- Either drop `setuptools-scm` or make the version dynamic from git tags. Add `sparks.__version__`.
- Dependencies:
  - Core: `torch>=2.4`, numpy, scipy, scikit-learn.
  - Extras: `allen`, `nlb`, `calcium`, `mamba`, `demos`, `dev`.
- Make the `allensdk` and `mamba_ssm` imports lazy, with an error message that names the right extra.
- `git rm --cached .pypirc`. Delete or regenerate `conda/`. Fix `.gitignore`.
- Add the MIT notice for `nlb_tools` (and the neuralpredictors attribution) and a `THIRD_PARTY_NOTICES` file.

**Phase 2: Safety-net tests (1 day)**
- Add `tests/conftest.py` with a seeded tiny-SPARKS fixture (N=12, E=8, L=4, K=2, T=20), plus `allen` and `slow` markers.
- Add regression tests for:
  - scan vs a sequential loop;
  - the ephys and calcium parallel vs sequential paths;
  - chunked == full, for the layers and for all of SPARKS;
  - `get_hebbian_coeffs_on_batch` with `last_only`;
  - `add_session`, both with per-session heads and with `joint_decoder`;
  - forward, `add_session` and `reset_state` for all 5 control blocks;
  - `train()` + `test()` on 2 sessions.
- Rewrite the 17 stale tests. Delete `test_providers.py`. Rename `test_blocks` to `.py`.
- The probe scripts in `/tmp/claude-30048506/-nemo-lab-iacarusof-home-shared-Nicolas-sparks/1292e35d-8042-4c86-b25c-7374c10f6891/scratchpad/` already cover most of these cases.
- Add CI (GitHub Actions): CPU pytest on Python 3.10–3.12, `python -m build`, `twine check`.

**Phase 3: Remove dead and broken code (½ day)**
- Delete:
  - `transformer.py` and its type-hint references;
  - `sparse.py` with the `sliding`, `window_size` and `block_size` settings, the `'sparse'` entries in `CONFIG_DICT`, and the README Beta line;
  - `TransformerEncoder`;
  - `ProjectionConfig`, or actually wire it up;
  - the dead cyclers and `MultiSessionDataset`;
  - the unused helpers in mec and movies;
  - the stub methods;
  - unused imports, via `ruff --fix`.

**Phase 4: Fix the A1 bugs (1–2 days)**
- Each fix gets its own regression test.
- `add_session` should only forward the config fields the caller explicitly set.
- Calcium should either use `tau_s` or reject a list of `tau_s`.
- Unknown kwargs should raise instead of being swallowed.
- Restore the Allen frame-index targets.

**Phase 5: Restructure and public API (2–3 days)**

Proposed layout:

```
sparks/
  __init__.py          # SPARKS, configs, train, evaluate, __version__
  models/
    __init__.py
    sparks.py  encoders.py  decoders.py (MLPDecoder, LinearDecoder, TransformerDecoder + old aliases)
    configs.py         # was dataclasses.py (shim left behind)
    hebbian_block.py   perceiver.py   layers.py (AttentionBlock, MambaBlock, FeedForward, DropPath)
    scan.py
    attention/  (__init__, base, ephys, calcium)
    controls/   (base, coefficient_blocks, sequence_blocks)
  training/            # was utils/
    train.py  evaluate.py (was test.py)  losses.py  metrics.py  optim.py (param groups, clipping, lr)
    cyclers.py         # LongCycler, yielding (session_idx, batch)
  data/
    base.py  (real ABC: [T,N] float32 spikes, dt in seconds, n_neurons)
    allen/ nlb/ mec.py
examples/              # notebooks, outputs stripped (outside the wheel)
scripts/preprocess_allen_{gratings,movies}.py   # argparse + main()
```

- In `train` and `evaluate`, replace `**kwargs` with explicit keyword-only arguments, or with a small `TrainConfig`.
- Merge the grad-norm variant into a flag. Keep one LR schedule.
- `evaluate()` always returns lists.
- Replace `self.device` with the device of the module's parameters.
- Use `default_factory` for the config defaults.

**Phase 6: Model internals (2 days)**
- Add a `StatefulMixin` for carried state.
- Merge the duplicated ephys Full/Light paths. Collapse calcium Full/Light into one class.
- Give `CoefficientAttentionBlock` and `HebbianAttentionBlock` a shared base class.
- Add a `HebbianFrontEnd` Protocol.
- One method name for adding a session (`add_session`) and one set of argument names (`embed_dim`, `session_id`).

**Phase 7: Data package (2–3 days)**
- Every factory returns one named result with the same arguments (`data_path`, `batch_size`, `seed`, …).
- One layout and one set of units everywhere.
- No network access during dataset construction.
- Use `np.random.default_rng`.
- Separate the gratings, spontaneous and movie loaders explicitly.
- Turn the preprocessing scripts into parametrised CLIs that regenerate every file the loaders read, and document that file layout.

**Phase 8: Docs and examples (1–2 days)**
- Standardise docstrings on Google style and enforce it with ruff's `D` rules. Rewrite history-narrating docstrings in the present tense. Settle VAE vs VIB wording everywhere.
- README: install instructions with extras, a minimal quickstart, working links, the GPLv3 fix, and a citation.
- Add CHANGELOG.md for 0.3.0 with a migration table (old call → new call), CONTRIBUTING.md and CITATION.cff.
- Port the 3 notebooks to `examples/` and strip their outputs (nbstripout in pre-commit). Smoke-run them in CI with tiny configs.
- Update the docs site (`franciscrickinstitute.github.io/sparks-ai`). Its source isn't in this repo.

**Phase 9: Release (½ day)**
- Build, `twine check`, upload to TestPyPI.
- Install into a fresh environment with `[all]` and run the tests and notebooks there.
- Tag `v0.3.0`, publish.
- Pin `sparks-examples` to `0.2.1`, or port it.
- Optional: `git filter-repo` to shrink history before collaborators clone.

Total: roughly 2–3 weeks of focused work. If you only have time for a minimal release, do Phases 1–4 plus the README and CHANGELOG from Phase 8: that fixes everything broken without restructuring.

I can start with Phase 1 now, or publish this plan as a shareable page for collaborators.