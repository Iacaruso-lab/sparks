import pytest


def test_base_dataset():
    """Test BaseDataset abstract class"""
    from sparks.data.base import BaseDataset
    
    # Should not be instantiable directly
    with pytest.raises(TypeError):
        BaseDataset()