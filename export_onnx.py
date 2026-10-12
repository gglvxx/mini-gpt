"""Exportă modelele antrenate în format ONNX pentru demo-ul din browser: docs/models/<dataset>/"""
import json

import numpy as np
import onnxruntime as ort
import torch

from generate import load_model
from src.config import DATASETS, ROOT_DIR, Config

MODELS_DIR = ROOT_DIR / "docs" / "models"

# Textele afișate în pagina web pentru fiecare dataset
DEMO_INFO = {
    "shakespeare": {
        "label": "Shakespeare",
        "description": "Tiny Shakespeare (~1 MB of plays)",
        "examples": ["ROMEO:", "KING HENRY:", "First Citizen:"],
    },
    "romanian": {
        "label": "Română",
        "description": "Eminescu, Caragiale, Slavici (Project Gutenberg, public domain)",
        "examples": ["A fost odată", "Luceafărul", "Și era"],
    },
}


class LogitsOnly(torch.nn.Module):
    """Învelim modelul ca să returneze DOAR logits. Formatul ONNX nu acceptă ieșiri None."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx):
        logits, _ = self.model(idx)
        return logits


def export(dataset: str) -> None:
    cfg = Config(dataset=dataset)
    cfg.device = "cpu"
    if not cfg.checkpoint_path.exists():
        print(f"[{dataset}] Sărit: nu există {cfg.checkpoint_path}")
        return

    model, tokenizer, ck = load_model(cfg)
    T = cfg.block_size
    out_dir = MODELS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    # Formă fixă [1, block_size]; promptul scurt e completat cu zerouri DUPĂ el (vezi index.html)
    torch.onnx.export(
        LogitsOnly(model).eval(),
        (torch.zeros((1, T), dtype=torch.long),),
        str(onnx_path),
        input_names=["idx"],
        output_names=["logits"],
        dynamo=True,
        external_data=False,
    )

    # Păstrăm doar exemplele scrise cu caractere existente în vocabular
    info = DEMO_INFO[dataset]
    vocab = set(tokenizer.chars)
    examples = [e for e in info["examples"] if set(e) <= vocab]

    meta = {
        "label": info["label"],
        "description": info["description"],
        "examples": examples,
        "chars": tokenizer.chars,
        "block_size": T,
        "val_loss": round(ck["val_loss"], 4),
        "params": model.num_params(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # Verificare: ONNX + padding trebuie să dea aceleași predicții ca PyTorch
    ids = tokenizer.encode("".join(tokenizer.chars[10:22]))  # orice secvență validă din vocabular
    L = len(ids)
    with torch.no_grad():
        logits_pt, _ = model(torch.tensor([ids]))
    ref = logits_pt[0, -1].numpy()

    padded = np.zeros((1, T), dtype=np.int64)
    padded[0, :L] = ids
    session = ort.InferenceSession(str(onnx_path))
    out = session.run(["logits"], {"idx": padded})[0][0, L - 1]

    diff = float(np.abs(ref - out).max())
    assert diff < 1e-4, f"[{dataset}] Modelul ONNX NU se potrivește cu cel PyTorch (diff {diff})"
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"[{dataset}] OK: {onnx_path} ({size_mb:.1f} MB), diff {diff:.1e}, exemple: {examples}")


def main():
    for dataset in DATASETS:
        export(dataset)


if __name__ == "__main__":
    main()