"""Descarcă și pregătește dataset-ul ales în data/<dataset>.txt.

    Shakespeare (implicit):  python prepare_data.py
    Literatură română:       MINIGPT_DATASET=romanian python prepare_data.py
"""
import requests

from src.cleaning import clean_gutenberg_text, remove_rare_chars
from src.config import config

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

# Cărți în limba română de pe Project Gutenberg, domeniu public (autori decedați înainte de 1926)
ROMANIAN_BOOKS = {
    35323: "Mihai Eminescu - Poezii",
    64597: "I. L. Caragiale - Nuvele",
    62916: "Ioan Slavici - Povești",
}

HEADERS = {"User-Agent": "mini-gpt-educational-project"}


def download(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    response.encoding = "utf-8"  # forțăm UTF-8, altfel diacriticele pot ieși stricate
    return response.text


def build_shakespeare() -> str:
    print(f"Descarc Tiny Shakespeare ...")
    return download(SHAKESPEARE_URL)


def build_romanian() -> str:
    parts = []
    for book_id, title in ROMANIAN_BOOKS.items():
        print(f"Descarc: {title} ...")
        raw = download(f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8")
        cleaned = clean_gutenberg_text(raw)
        print(f"  {len(raw):,} caractere brute -> {len(cleaned):,} după curățare")
        parts.append(cleaned)

    text = "\n\n".join(parts)
    text, removed = remove_rare_chars(text, min_count=20)
    if removed:
        print(f"Caractere rare eliminate: {''.join(sorted(removed))!r}")
    return text


BUILDERS = {"shakespeare": build_shakespeare, "romanian": build_romanian}


def show_stats() -> None:
    text = config.data_path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    print(f"\nDataset: {config.dataset}")
    print(f"Total caractere: {len(text):,}")
    print(f"Caractere unice (vocabular): {len(chars)}")
    print(f"Vocabular: {''.join(chars)!r}")
    print("\n--- Primele 300 de caractere ---")
    print(text[:300])


def main():
    path = config.data_path
    if path.exists():
        print(f"Dataset-ul există deja: {path}")
    else:
        text = BUILDERS[config.dataset]()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Salvat în: {path}")
    show_stats()


if __name__ == "__main__":
    main()