"""Tokenizer la nivel de caracter: fiecare caracter unic primește un ID numeric."""


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        # stoi = "string to int": ex. 'a' -> 39
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        # itos = "int to string": ex. 39 -> 'a'
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Construiește vocabularul din toate caracterele unice ale textului."""
        # sorted -> aceeași ordine (deci aceleași ID-uri) la fiecare rulare
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Text -> listă de ID-uri. Ex: 'hi' -> [46, 47]"""
        unknown = set(text) - self.stoi.keys()
        if unknown:
            raise ValueError(f"Caractere care nu există în vocabular: {sorted(unknown)}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        """Listă de ID-uri -> text. int() permite și elemente de tip tensor PyTorch."""
        return "".join(self.itos[int(i)] for i in ids)