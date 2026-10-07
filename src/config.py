"""Configurație centrală: toți hiperparametrii proiectului într-un singur loc."""
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Folderul rădăcină al proiectului (mini-gpt/), calculat relativ la acest fișier.
# Așa căile funcționează indiferent din ce folder rulezi scripturile.
ROOT_DIR = Path(__file__).resolve().parent.parent


def get_device() -> str:
    """Alege automat cel mai rapid hardware disponibil."""
    if torch.cuda.is_available():
        return "cuda"  # GPU NVIDIA
    return "cpu"


# Setări mai mari, folosite automat doar dacă există GPU
GPU_PRESET = dict(
    block_size=256,
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.2,
    batch_size=64,
    learning_rate=3e-4,
    max_iters=5000,
)


@dataclass
class Config:
    # --- Hardware ---
    device: str = field(default_factory=get_device)
    seed: int = 1337  # fixează randomness-ul -> rezultate reproductibile

    # --- Căi (paths) ---
    data_dir: Path = ROOT_DIR / "data"
    checkpoint_dir: Path = ROOT_DIR / "checkpoints"
    dataset_url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt"
    )

    # --- Date ---
    train_split: float = 0.9  # 90% pentru antrenare, 10% pentru validare

    # --- Model (valorile implicite = preset mic, potrivit pentru CPU) ---
    block_size: int = 128   # câte caractere "vede" modelul deodată (fereastra de context)
    n_embd: int = 128       # lungimea vectorului care reprezintă fiecare caracter
    n_head: int = 4         # câte "capete" de atenție rulează în paralel
    n_layer: int = 4        # câte blocuri Transformer sunt stivuite
    dropout: float = 0.1    # % din neuroni opriți aleator la antrenare (anti-overfitting)

    # --- Antrenare ---
    batch_size: int = 32          # câte secvențe procesăm simultan
    learning_rate: float = 1e-3   # cât de mari sunt "pașii" de învățare
    max_iters: int = 3000         # câți pași de antrenare în total
    eval_interval: int = 250      # la câți pași măsurăm performanța
    eval_iters: int = 100         # pe câte batch-uri facem media la evaluare

    def __post_init__(self):
        # Dacă avem GPU, suprascriem cu preset-ul mare
        if self.device == "cuda":
            for key, value in GPU_PRESET.items():
                setattr(self, key, value)

        # Fiecare cap de atenție primește o felie egală din n_embd
        assert self.n_embd % self.n_head == 0, "n_embd trebuie să fie divizibil cu n_head"

    @property
    def data_path(self) -> Path:
        return self.data_dir / "input.txt"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "model.pt"


# Instanța globală: în restul proiectului facem doar `from src.config import config`
config = Config()