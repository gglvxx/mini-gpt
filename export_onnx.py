"""Exportă modelul în format ONNX în folderul docs/, pentru rulare direct în browser."""
import json

import numpy as np
import onnxruntime as ort
import torch

from generate import load_model
from src.config import ROOT_DIR, Config

# GitHub Pages poate servi automat site-ul din folderul docs/ al repo-ului
OUT_DIR = ROOT_DIR / "docs"


class LogitsOnly(torch.nn.Module):
    """Învelim modelul ca să returneze DOAR logits. Formatul ONNX nu acceptă ieșiri None."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx):
        logits, _ = self.model(idx)
        return logits


def main():
    cfg = Config()
    cfg.device = "cpu"  # exportul se face pe CPU, indiferent de hardware
    model, tokenizer, ck = load_model(cfg)
    T = cfg.block_size

    OUT_DIR.mkdir(exist_ok=True)
    onnx_path = OUT_DIR / "model.onnx"

    # Exportăm cu formă FIXĂ [1, block_size]: cea mai simplă și mai robustă variantă.
    # Promptul mai scurt va fi completat (padding) cu zerouri DUPĂ el.
    # Masca cauzală garantează că padding-ul de după nu influențează predicțiile din prompt.
    dummy = torch.zeros((1, T), dtype=torch.long)
    torch.onnx.export(
        LogitsOnly(model).eval(),
        (dummy,),
        str(onnx_path),
        input_names=["idx"],
        output_names=["logits"],
        dynamo=True,
        external_data=False,  # greutățile în ACELAȘI fișier -> un singur fișier de încărcat în browser
    )

    # Informațiile de care pagina web are nevoie: vocabularul și dimensiunea contextului
    meta = {
        "chars": tokenizer.chars,
        "block_size": T,
        "val_loss": round(ck["val_loss"], 4),
        "params": model.num_params(),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # --- Verificare: ONNX + padding trebuie să dea EXACT aceleași predicții ca PyTorch ---
    ids = tokenizer.encode("ROMEO:")
    L = len(ids)

    with torch.no_grad():
        logits_pt, _ = model(torch.tensor([ids]))  # PyTorch: doar promptul, fără padding
    ref = logits_pt[0, -1].numpy()

    padded = np.zeros((1, T), dtype=np.int64)  # ONNX: promptul + zerouri până la block_size
    padded[0, :L] = ids
    session = ort.InferenceSession(str(onnx_path))
    out = session.run(["logits"], {"idx": padded})[0][0, L - 1]  # citim poziția ultimului caracter real

    diff = float(np.abs(ref - out).max())
    print(f"Diferență maximă PyTorch vs ONNX: {diff:.2e}")
    assert diff < 1e-4, "Modelul ONNX NU se potrivește cu cel PyTorch!"

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"OK. Exportat: {onnx_path} ({size_mb:.1f} MB) + {OUT_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()