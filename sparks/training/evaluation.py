from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

from sparks.models.sparks import SPARKS
from sparks.training.training_loop import stack_future_targets


@torch.no_grad()
def evaluate_on_batch(model: SPARKS,
                      inputs: torch.Tensor,
                      targets: Optional[torch.Tensor] = None,
                      loss_fn: Optional[Callable] = None,
                      *,
                      session_id: Union[str, int] = 0,
                      act: Optional[Callable] = None,
                      burnin: int = 0,
                      chunk_length: Optional[int] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Runs the model, in eval mode, on a batch of inputs, and computes the loss if given targets and
    a loss function.

    The loss is computed as in `sparks.training.training_loop.train_on_batch`, without the KL
    term: over every timestep after `burnin`, against each timestep's window of the next tau_f
    targets.

    Args:
        model (SPARKS): The model instance.
        inputs (torch.Tensor): [B, T, N] input spikes.
        targets (torch.Tensor, optional): [B, T, ...] targets. Default is None (no loss).
        loss_fn (Callable, optional): The loss function, called as
            `loss_fn(decoder_outputs, targets)`. Default is None (no loss).
        session_id (Union[str, int]): The session the batch comes from. Default is 0.
        act (Callable, optional): Applied to the decoder outputs before the loss and before they
            are returned, e.g. torch.sigmoid for a decoder trained in logit space. Default is None.
        burnin (int): The number of initial timesteps left out of the loss. Default is 0.
        chunk_length (int, optional): If given, walks through `inputs` in consecutive,
            non-overlapping chunks of this many timesteps instead of one forward pass over the
            whole sequence, carrying the model's recurrent state across chunk boundaries via
            `carry_state=True` (see `SPARKS.forward`) -- use this when a recording is too long to
            process in one pass, e.g. when a control block's per-timestep [B, T, N, N]
            coefficient history (see `sparks.models.controls`) doesn't fit in memory. Each
            chunk's outputs are moved to CPU before being concatenated, so the accelerator only
            ever holds one chunk. Calls `model.reset_state(session_id)` once before the first
            chunk, so call this function once per trial/recording, not once per chunk. The
            returned outputs and loss are the same as the unchunked path's, up to floating-point
            error. Default is None (single pass over the whole sequence).

    Returns:
        loss (torch.Tensor or None): The loss, or None without targets or a loss function.
        encoder_outputs (torch.Tensor): [B, T, latent_dim] latents, on CPU.
        decoder_outputs (torch.Tensor): [B, T, ...] decoder outputs, on CPU.
    """
    model.eval()

    if chunk_length is None:
        encoder_outputs, decoder_outputs, _, _ = model(inputs, session_id=session_id)
        encoder_outputs, decoder_outputs = encoder_outputs.cpu(), decoder_outputs.cpu()
    else:
        model.reset_state(session_id)
        encoder_chunks, decoder_chunks = [], []
        for start in range(0, inputs.shape[1], chunk_length):
            encoder_chunk, decoder_chunk, _, _ = model(inputs[:, start:start + chunk_length],
                                                       session_id=session_id, carry_state=True)
            encoder_chunks.append(encoder_chunk.cpu())
            decoder_chunks.append(decoder_chunk.cpu())
        encoder_outputs, decoder_outputs = torch.cat(encoder_chunks, dim=1), torch.cat(decoder_chunks, dim=1)

    if act is not None:
        decoder_outputs = act(decoder_outputs)

    loss = None
    if loss_fn is not None and targets is not None:
        tau_f = model.tau_f
        predictions = decoder_outputs[:, :decoder_outputs.shape[1] - tau_f + 1]
        loss = loss_fn(predictions[:, burnin:], stack_future_targets(targets, tau_f)[:, burnin:].cpu())

    return loss, encoder_outputs, decoder_outputs


@torch.no_grad()
def evaluate(model: SPARKS,
             test_dls: List[torch.utils.data.DataLoader],
             loss_fn: Optional[Callable] = None,
             *,
             session_ids: Optional[Sequence[Union[str, int]]] = None,
             unsupervised: bool = False,
             **batch_kwargs) -> Tuple[Optional[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Evaluates the model on one or several sessions' dataloaders.

    Args:
        model (SPARKS): The model instance.
        test_dls (List[torch.utils.data.DataLoader]): One dataloader per session, each yielding
            (inputs, targets) batches.
        loss_fn (Callable, optional): The loss function, see `evaluate_on_batch`. Default is None
            (no loss).
        session_ids (Sequence[Union[str, int]], optional): The session of each dataloader.
            Default is None (0, 1, ...).
        unsupervised (bool): If True, compare the model's predictions with its inputs, ignoring
            the dataloaders' targets. Default is False.
        **batch_kwargs: Forwarded to `evaluate_on_batch` (e.g. act, burnin, chunk_length).

    Returns:
        loss (torch.Tensor or None): The loss summed over every batch of every session, or None
            without a loss function.
        encoder_outputs (List[torch.Tensor]): One [n_trials, T, latent_dim] tensor per session.
        decoder_outputs (List[torch.Tensor]): One [n_trials, T, ...] tensor per session.
    """
    if session_ids is None:
        session_ids = list(range(len(test_dls)))

    total_loss = None
    encoder_outputs, decoder_outputs = [], []
    for test_dl, session_id in zip(test_dls, session_ids):
        session_encoder_outputs, session_decoder_outputs = [], []
        for inputs, targets in test_dl:
            loss, encoder_batch, decoder_batch = evaluate_on_batch(model, inputs, inputs if unsupervised else targets,
                                                                   loss_fn, session_id=session_id, **batch_kwargs)
            if loss is not None:
                total_loss = loss if total_loss is None else total_loss + loss
            session_encoder_outputs.append(encoder_batch)
            session_decoder_outputs.append(decoder_batch)
        encoder_outputs.append(torch.cat(session_encoder_outputs, dim=0))
        decoder_outputs.append(torch.cat(session_decoder_outputs, dim=0))

    return total_loss, encoder_outputs, decoder_outputs


@torch.no_grad()
def get_hebbian_coeffs_on_batch(model: SPARKS,
                                inputs: torch.Tensor,
                                *,
                                session_id: Union[str, int] = 0,
                                chunk_length: Optional[int] = None,
                                last_only: bool = False) -> torch.Tensor:
    """
    Runs a batch of inputs through the Hebbian attention block only (`model.hebbian_forward`) and
    returns its STDP coefficients, without running the rest of the encoder or the decoder.

    Mirrors `evaluate_on_batch`, minus everything that concerns predictions: there is no loss, no
    target handling and no `act`, since the coefficients are the raw state of the Hebbian
    dynamics rather than an output to be scored.

    Args:
        model (SPARKS): The model instance.
        inputs (torch.Tensor): The input data for the batch, shape [B, T, N].
        session_id (Union[str, int]): The session the batch comes from. Default is 0.
        chunk_length (int, optional): If given, walks through `inputs` in consecutive,
            non-overlapping chunks of this many timesteps instead of one pass over the whole
            sequence, carrying the Hebbian block's STDP/trace state across chunk boundaries via
            `carry_state=True` (see `SPARKS.hebbian_forward`). Use this whenever the sequence is
            long: the [B, T, N, N] coefficient tensor is the *returned* value, so both the peak and
            the final footprint scale with T * N^2. Each chunk's coefficients are moved to CPU
            before being concatenated back together, which is what keeps the accelerator's peak
            at one chunk's worth. Calls `model.reset_state(session_id)` once before the first
            chunk, so call this function once per trial/recording, not once per chunk. Chunking
            changes nothing about the returned coefficients: the dynamics are continuous across
            boundaries, so the result is identical (up to floating-point) to the unchunked path.
            Default is None (single pass over the whole sequence).
        last_only (bool): If True, keep only the final timestep's coefficients instead of the
            whole history, returned with the time dimension kept at length 1 so that the usual
            `[..., -1, :, :]` indexing is unchanged. Combined with `chunk_length` this is what
            makes a long recording tractable: each chunk's history is released as soon as the
            next chunk is computed, so the retained footprint stops scaling with T and stays at
            one chunk's worth (chunk_length * N^2) rather than T * N^2. Use it whenever only the
            end state is wanted -- the coefficients accumulate each pair's history without leak,
            so the last timestep is the estimate over the full recording. On its own
            (chunk_length=None) it saves nothing during the pass, since the layer still builds
            the [B, T, N, N] history internally before it is sliced; pass both. Default is False.

    Returns:
        torch.Tensor: The STDP coefficients, on CPU. Shape [B, T, N, N] with a single Hebbian
            head, or [n_heads, B, T, N, N] with several (`tau_s` given as a list) -- in both cases
            time is dimension -3, and is length 1 when `last_only`.
    """
    model.eval()

    # Time is dim -3 for both the single-head [B, T, N, N] and multi-head [H, B, T, N, N]
    # layouts, so slicing it as [..., -1:, :, :] keeps the dimension at length 1 either way.
    if chunk_length is None:
        coeffs = model.hebbian_forward(inputs, session_id=session_id).cpu()
        return coeffs[..., -1:, :, :] if last_only else coeffs

    model.reset_state(session_id)
    coeff_chunks = []
    for start in range(0, inputs.shape[1], chunk_length):
        coeffs_chunk = model.hebbian_forward(inputs[:, start:start + chunk_length], session_id=session_id,
                                             carry_state=True)
        if last_only:
            # Replacing rather than appending is what bounds the memory: the previous chunk's
            # history is dropped here, leaving only the running last timestep. The dynamics
            # still run continuously, since the state carries across the boundary.
            coeff_chunks = [coeffs_chunk[..., -1:, :, :].cpu()]
        else:
            coeff_chunks.append(coeffs_chunk.cpu())
        del coeffs_chunk

    return torch.cat(coeff_chunks, dim=-3)
