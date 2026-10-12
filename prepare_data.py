"""Descarcă și pregătește dataset-ul ales în data/<dataset>.txt.

    Shakespeare (implicit):  python prepare_data.py
    Literatură română:       MINIGPT_DATASET=romanian python prepare_data.py
"""
import time

import requests

from src.cleaning import clean_gutenberg_text, diacritic_ratio, remove_rare_chars  # NOU
from src.config import config

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

# Cărți CANDIDATE în limba română de pe Project Gutenberg, domeniu public.
# Fiecare e verificată automat: cele transcrise fără diacritice sunt excluse.
ROMANIAN_BOOKS = {  # NOU
    35323: "Mihai Eminescu - Poezii",
    64597: "I. L. Caragiale - Nuvele",
    62916: "Ioan Slavici - Povești",
    65565: "Regina Maria (trad. N. Iorga) - Țara mea",
}
MIN_DIACRITIC_RATIO = 0.01  # NOU: sub 1% litere cu diacritice -> cartea e exclusă

HEADERS = {"User-Agent": "mini-gpt-educational-project"}
RAW_DIR = config.data_dir / "raw"  # cache pentru fișierele brute descărcate


def download(url: str, retries: int = 5) -> str:
    """Descarcă un text. La erori de rețea/server reîncearcă, cu pauze tot mai lungi."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=60)
            response.raise_for_status()
            response.encoding = "utf-8"  # forțăm UTF-8, altfel diacriticele pot ieși stricate
            return response.text
        except requests.RequestException as e:
            if attempt == retries:
                raise
            wait = 10 * attempt
            print(f"  Eroare: {e.__class__.__name__}. Reîncerc în {wait}s ({attempt}/{retries}) ...")
            time.sleep(wait)


def download_gutenberg_book(book_id: int) -> str:
    """Descarcă o carte o singură dată; la rulările următoare o citește din data/raw/."""
    cache = RAW_DIR / f"{book_id}.txt"
    if cache.exists():
        print("  (din cache local)")
        return cache.read_text(encoding="utf-8")

    text = download(f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def build_shakespeare() -> str:
    print("Descarc Tiny Shakespeare ...")
    return download(SHAKESPEARE_URL)


def build_romanian() -> str:
    parts = []
    for book_id, title in ROMANIAN_BOOKS.items():
        print(f"Descarc: {title} ...")
        raw = download_gutenberg_book(book_id)
        cleaned = clean_gutenberg_text(raw)

        # NOU: verificare la nivel de carte întreagă
        ratio = diacritic_ratio(cleaned)
        if ratio < MIN_DIACRITIC_RATIO:
            print(f"  EXCLUSĂ: doar {ratio:.2%} litere cu diacritice (transcrisă fără diacritice)")
            continue

        print(f"  {len(raw):,} caractere brute -> {len(cleaned):,} după curățare "
              f"({ratio:.2%} litere cu diacritice)")
        parts.append(cleaned)

    if not parts:
        raise SystemExit("Nicio carte nu a trecut verificarea diacriticelor.")

    text = "\n\n".join(parts)
    text, removed = remove_rare_chars(text, min_count=20)
    if removed:
        print(f"Simboluri rare eliminate: {''.join(sorted(removed))!r}")
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