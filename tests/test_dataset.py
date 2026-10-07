"""Teste pentru dataset. Rulează cu: pytest"""
import pytest
import torch

from src.config import Config

cfg = Config()
pytestmark = pytest.mark.skipif(
    not cfg.data_path.exists(), reason="Rulează întâi prepare_data.py"
)


@pytest.fixture(scope="module")
def dataset():
    from src.dataset import TextDataset

    cfg.batch_size = 4  # valori mici -> teste rapide
    cfg.block_size = 8
    return TextDataset(cfg)


def test_split_sizes(dataset):
    total = len(dataset.train_data) + len(dataset.val_data)
    assert len(dataset.train_data) == int(cfg.train_split * total)


def test_batch_shape_and_type(dataset):
    x, y = dataset.get_batch("train")
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.long


def test_y_is_x_shifted_by_one(dataset):
    # Proprietatea esențială: y[i] este caracterul care urmează după x[i]
    x, y = dataset.get_batch("val")
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_invalid_split_raises(dataset):
    with pytest.raises(ValueError):
        dataset.get_batch("test")