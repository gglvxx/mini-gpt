"""Generează text cu modelul antrenat.

Exemple:
    python generate.py
    python generate.py --prompt "ROMEO:" --tokens 300
    python generate.py --temperature 0.5 --seed 42
"""
import argparse

import torch

from src.config import Config
from src.model import MiniGPT
from src.tokenizer import CharTokenizer


def load_model(cfg: Config):
    """Reconstruiește modelul și tokenizer-ul exclusiv din checkpoint (fără dataset)."""
    if not cfg.checkpoint_path.exists():
        raise SystemExit(f"Nu există {cfg.checkpoint_path}. Rulează întâi: python train.py")

    # weights_only=True -> încarcă doar date, nu cod arbitrar (practică de securitate)
    ck = torch.load(cfg.checkpoint_path, map_location=cfg.device, weights_only=True)

    # Arhitectura trebuie să fie IDENTICĂ cu cea de la antrenare,
    # altfel greutățile salvate nu "încap" în model
    for key, value in ck["model_args"].items():
        setattr(cfg, key, value)

    tokenizer = CharTokenizer(ck["chars"])
    model = MiniGPT(cfg, tokenizer.vocab_size).to(cfg.device)
    model.load_state_dict(ck["model_state"])
    model.eval()  # mod inferență: dropout dezactivat
    return model, tokenizer, ck


def main():
    parser = argparse.ArgumentParser(description="Generare text cu MiniGPT")
    parser.add_argument("--prompt", default="\n", help="Textul de start")
    parser.add_argument("--tokens", type=int, default=500, help="Câte caractere noi să genereze")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="<1 = mai sigur/repetitiv, >1 = mai creativ/haotic")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Alege doar dintre cele mai probabile K caractere")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixează rezultatul (același seed = același text)")
    args = parser.parse_args()

    if args.temperature <= 0:
        raise SystemExit("--temperature trebuie să fie > 0")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    cfg = Config()
    model, tokenizer, ck = load_model(cfg)
    print(f"[Model: pasul {ck['step']}, val loss {ck['val_loss']:.4f}, device {cfg.device}]\n")

    # Promptul poate conține doar caractere pe care modelul le-a văzut la antrenare
    try:
        ids = tokenizer.encode(args.prompt)
    except ValueError as e:
        raise SystemExit(f"Prompt invalid. {e}")

    idx = torch.tensor([ids], dtype=torch.long, device=cfg.device)  # [1, T]: batch de 1 secvență

    # Generare în flux (streaming): afișăm fiecare caracter imediat ce e generat
    print(args.prompt, end="", flush=True)
    for _ in range(args.tokens):
        idx = model.generate(idx, max_new_tokens=1, temperature=args.temperature, top_k=args.top_k)
        print(tokenizer.decode(idx[0, -1:]), end="", flush=True)
    print()


if __name__ == "__main__":
    main()