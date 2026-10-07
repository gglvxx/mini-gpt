"""Teste automate pentru tokenizer. Rulează cu: pytest"""
import pytest

from src.config import config
from src.tokenizer import CharTokenizer


def test_vocab_is_sorted_and_unique():
    tok = CharTokenizer.from_text("banana")
    assert tok.chars == ["a", "b", "n"]
    assert tok.vocab_size == 3


def test_encode_decode_roundtrip():
    # Proprietatea esențială: decode(encode(x)) trebuie să returneze exact x
    text = "hello world"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_unknown_char_raises_error():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.encode("abz")  # 'z' nu e în vocabular


@pytest.mark.skipif(not config.data_path.exists(), reason="Rulează întâi prepare_data.py")
def test_real_dataset():
    text = config.data_path.read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    assert tok.vocab_size == 65
    sample = text[:1000]
    assert tok.decode(tok.encode(sample)) == sample