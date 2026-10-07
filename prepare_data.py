"""Descarcă dataset-ul Tiny Shakespeare în data/input.txt și afișează statistici."""
import requests

from src.config import config


def download_dataset() -> None:
    path = config.data_path
    if path.exists():
        print(f"Dataset-ul există deja: {path}")
        return

    print(f"Descarc din {config.dataset_url} ...")
    response = requests.get(config.dataset_url, timeout=30)
    response.raise_for_status()  # oprește scriptul cu eroare clară dacă descărcarea eșuează

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    print(f"Salvat în: {path}")


def show_stats() -> None:
    text = config.data_path.read_text(encoding="utf-8")
    chars = sorted(set(text))  # toate caracterele unice = vocabularul modelului

    print(f"Total caractere: {len(text):,}")
    print(f"Caractere unice (vocabular): {len(chars)}")
    print(f"Vocabular: {''.join(chars)!r}")
    print("\n--- Primele 300 de caractere ---")
    print(text[:300])


if __name__ == "__main__":
    download_dataset()
    show_stats()