"""Pregătește datele pentru antrenare: text -> tensor -> batch-uri (x, y)."""
import torch

from src.config import Config
from src.tokenizer import CharTokenizer


class TextDataset:
    def __init__(self, config: Config):
        self.config = config

        if not config.data_path.exists():
            raise FileNotFoundError(
                f"{config.data_path} lipsește. Rulează întâi: python prepare_data.py"
            )
        text = config.data_path.read_text(encoding="utf-8")
        self.tokenizer = CharTokenizer.from_text(text)

        # Tot textul devine UN singur tensor lung de ID-uri.
        # dtype=torch.long = numere întregi (necesar pentru embedding-uri mai târziu)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # Primele 90% pentru antrenare, ultimele 10% pentru validare.
        # Validarea = text pe care modelul NU îl vede la antrenare -> verificăm dacă
        # a învățat cu adevărat sau doar a memorat.
        n = int(config.train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Returnează un batch aleatoriu (x, y), fiecare de formă [batch_size, block_size]."""
        if split not in ("train", "val"):
            raise ValueError(f"split trebuie să fie 'train' sau 'val', nu '{split}'")

        data = self.train_data if split == "train" else self.val_data
        bs, block = self.config.batch_size, self.config.block_size

        # Alege batch_size poziții de start aleatorii în text
        ix = torch.randint(len(data) - block, (bs,))

        # Exemplu cu text "Hello!" și block_size=5:
        #   x = "Hello"  (intrarea)
        #   y = "ello!"  (ținta = x decalat cu o poziție)
        # La fiecare poziție, modelul învață: "după aceste caractere, urmează y[i]"
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + block + 1] for i in ix])

        # Mută datele pe același hardware ca modelul (CPU sau GPU)
        return x.to(self.config.device), y.to(self.config.device)