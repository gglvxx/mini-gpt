"""Teste pentru curățarea textelor. Rulează cu: pytest"""
from src.cleaning import clean_gutenberg_text, diacritic_ratio, remove_rare_chars


def test_cedilla_diacritics_are_normalized():
    assert clean_gutenberg_text("ş ţ Ş Ţ") == "ș ț Ș Ț\n"


def test_gutenberg_boilerplate_is_removed():
    raw = (
        "Licență și antet\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK POEZII ***\n"
        "Luceafărul\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK POEZII ***\n"
        "Text de licență la final"
    )
    assert clean_gutenberg_text(raw) == "Luceafărul\n"


def test_italics_and_footnotes_are_removed():
    assert clean_gutenberg_text("_Luceafărul_[12]") == "Luceafărul\n"


def test_extra_blank_lines_are_collapsed():
    assert clean_gutenberg_text("a\n\n\n\n\nb") == "a\n\nb\n"


def test_punctuation_variants_are_unified():
    assert clean_gutenberg_text("s’au „bine” — da") == "s'au \"bine\" – da\n"


def test_rare_symbols_are_removed():
    text, removed = remove_rare_chars("aaaa§", min_count=2)
    assert text == "aaaa"
    assert removed == {"§"}


def test_rare_letters_are_never_removed():
    text, removed = remove_rare_chars("aaaaĂKX", min_count=2)
    assert text == "aaaaĂKX"
    assert removed == set()


def test_diacritic_ratio():
    assert diacritic_ratio("Imbraca-te in doliu, frumoasa Bucovina") == 0.0
    assert diacritic_ratio("Îmbracă-te în doliu") > 0.1
    assert diacritic_ratio("") == 0.0