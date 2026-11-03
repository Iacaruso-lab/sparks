"""
SPARKS: Hebbian-attention models of spiking signals.

The main entry points are re-exported here; see `sparks.models`, `sparks.training` and
`sparks.data` for everything else.
"""
from sparks.models import HebbianAttentionConfig, SharedConfig, SPARKS
from sparks.training import evaluate, train

__version__ = "0.3.0"

__all__ = ['SPARKS', 'HebbianAttentionConfig', 'SharedConfig', 'train', 'evaluate', '__version__']
