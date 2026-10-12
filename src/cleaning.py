"""Curățarea textelor brute descărcate de pe Project Gutenberg."""
import re
import string
from collections import Counter

# Textele vechi folosesc des ş/ţ cu SEDILĂ (greșit). Forma corectă în română e cu VIRGULĂ: ș/ț.
# Fără unificare, modelul ar învăța două "litere" diferite pentru același sunet.
DIACRITICS_FIX = str.maketrans({"ş": "ș", "Ş": "Ș", "ţ": "ț", "Ţ": "Ț"})

# Variante tipografice ale aceluiași semn -> o singură formă.
# Ele nu poartă nicio informație despre conținut, doar îngreunează predicția.
PUNCTUATION_FIX = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'",  # apostrofuri
    "„": '"', "“": '"', "”": '"', "«": '"', "»": '"',  # ghilimele
    "—": "–",       # linie lungă (em dash) -> linie de dialog (en dash)
    "\u00a0": " ",  # spațiu "non-breaking" (invizibil, dar diferit) -> spațiu normal
})

ROMANIAN_DIACRITICS = frozenset("ăâîșțĂÂÎȘȚ")

# Literele nu sunt eliminate niciodată: ștergerea lor ar strica cuvintele (ex: nume proprii cu K, X)
PROTECTED_CHARS = frozenset(string.ascii_letters) | ROMANIAN_DIACRITICS


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
    text = text.translate(PUNCTUATION_FIX)
    text = re.sub(r"\[(?:Illustration|Ilustrație)[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+\]", "", text)  # trimiteri la note de subsol: [1], [23]
    text = text.replace("_", "")         # Gutenberg marchează italicul cu _text_
    text = "\n".join(line.rstrip() for line in text.split("\n"))  # spații inutile la final de rând
    text = re.sub(r"\n{3,}", "\n\n", text)  # maxim un rând gol între paragrafe
    return text.strip() + "\n"


def diacritic_ratio(text: str) -> float:
    """Procentul de litere cu diacritice. Româna literară are în mod normal ~3-5%.
    O carte întreagă sub 1% a fost transcrisă fără diacritice și ar învăța modelul o ortografie greșită."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in ROMANIAN_DIACRITICS for c in letters) / len(letters)


def remove_rare_chars(text: str, min_count: int = 20,
                      protected: frozenset[str] = PROTECTED_CHARS) -> tuple[str, set[str]]:
    """Elimină simbolurile care apar foarte rar (greșeli de tipar, semne exotice).
    Caracterele din `protected` (literele) nu sunt eliminate niciodată."""
    counts = Counter(text)
    rare = {ch for ch, n in counts.items() if n < min_count and ch not in protected}
    if rare:
        text = "".join(ch for ch in text if ch not in rare)
    return text, rare