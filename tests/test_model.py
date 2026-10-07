"""Teste pentru arhitectura modelului. Rulează cu: pytest"""
import math

import pytest
import torch

from src.config import Config
from src.model import MiniGPT

VOCAB = 65


@pytest.fixture
def model():
    # Model minuscul, pe CPU, fără dropout -> teste rapide și deterministe
    cfg = Config()
    cfg.device = "cpu"
    cfg.n_embd, cfg.n_head, cfg.n_layer = 32, 4, 2
    cfg.block_size, cfg.dropout = 16, 0.0
    torch.manual_seed(0)
    return MiniGPT(cfg, VOCAB).eval()


def test_output_shape(model):
    idx = torch.randint(VOCAB, (2, 10))
    logits, loss = model(idx)
    assert logits.shape == (2, 10, VOCAB)
    assert loss is None


def test_initial_loss_is_near_random_guess(model):
    # Model neantrenat = ghicește uniform -> loss ≈ ln(vocab_size) ≈ 4.17
    idx = torch.randint(VOCAB, (4, 16))
    targets = torch.randint(VOCAB, (4, 16))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(VOCAB)) < 0.5


def test_causal_mask_hides_future(model):
    # Schimbăm DOAR ultimul token. Predicțiile pentru pozițiile anterioare
    # trebuie să rămână identice -> modelul nu vede viitorul.
    idx = torch.randint(VOCAB, (1, 10))
    idx_changed = idx.clone()
    idx_changed[0, -1] = (idx[0, -1] + 1) % VOCAB
    out1, _ = model(idx)
    out2, _ = model(idx_changed)
    assert torch.allclose(out1[:, :-1], out2[:, :-1])


def test_generate_length(model):
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=30)  # depășește block_size=16 intenționat
    assert out.shape == (1, 31)


def test_too_long_input_raises(model):
    with pytest.raises(ValueError):
        model(torch.randint(VOCAB, (1, 17)))