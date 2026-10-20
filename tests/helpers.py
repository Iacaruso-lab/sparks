import torch

# Tiny dimensions shared across the suite: big enough to exercise every code path (several
# timesteps, several latents, a bottleneck > 1), small enough that each test runs in well under a
# second on CPU.
BATCH = 3
N_TIMESTEPS = 16
N_NEURONS = 10
EMBED_DIM = 8
LATENT_DIM = 4
BOTTLENECK_DIM = 2
OUTPUT_DIM = 5


@torch.no_grad()
def perturb_parameters(model: torch.nn.Module, scale: float = 0.05) -> torch.nn.Module:
    """
    Adds small Gaussian noise to every parameter of `model`, in place.

    Some layers are zero-initialized (e.g. the decoders' output layers), which makes their output
    independent of the input at initialization; comparing two such outputs would pass trivially.
    """
    for param in model.parameters():
        param.add_(scale * torch.randn_like(param))
    return model
