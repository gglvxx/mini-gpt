"""Curățarea textelor brute descărcate de pe Project Gutenberg."""
import re
from collections import Counter

# Textele vechi folosesc des ş/ţ cu SEDILĂ (greșit). Forma corectă în română e cu VIRGULĂ: ș/ț.
# Fără unificare, modelul ar învăța două "litere" diferite pentru același sunet.
DIACRITICS_FIX = str.maketrans({"ş": "ș", "Ş": "Ș", "ţ": "ț", "Ţ": "Ț"})


def strip_gutenberg_boilerplate(text: str) -> str:
    """Păstrează doar opera: elimină antetul și licența adăugate de Project Gutenberg."""
    start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK.*?\*\*\*",
                      text, re.IGNORECASE | re.DOTALL)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG E-?BOOK", text, re.IGNORECASE)
    if end:
        text = text[:end.start()]
    return text


def clean_gutenberg_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # capete de rând Windows -> Linux
    text = strip_gutenberg_boilerplate(text)
    text = text.translate(DIACRITICS_FIX)
    text = re.sub(r"\[(?:Illustration|Ilustrație)[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+\]", "", text)  # trimiteri la note de subsol: [1], [23]
    text = text.replace("_", "")         # Gutenberg marchează italicul cu _text_
    text = "\n".join(line.rstrip() for line in text.split("\n"))  # spații inutile la final de rând
    text = re.sub(r"\n{3,}", "\n\n", text)  # maxim un rând gol între paragrafe
    return text.strip() + "\n"


def remove_rare_chars(text: str, min_count: int = 20) -> tuple[str, set[str]]:
    """Elimină caracterele care apar foarte rar (simboluri, greșeli de tipar).
    Modelul nu le poate învăța din câteva exemple, iar ele doar umflă vocabularul."""
    counts = Counter(text)
    rare = {ch for ch, n in counts.items() if n < min_count}
    if rare:
        text = "".join(ch for ch in text if ch not in rare)
    return text, rare