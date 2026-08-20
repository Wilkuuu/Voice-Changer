"""
Polish text normalization before TTS (numbers, abbreviations, punctuation).
"""

from __future__ import annotations

import re

_ONES = {
    0: "zero", 1: "jeden", 2: "dwa", 3: "trzy", 4: "cztery", 5: "pięć",
    6: "sześć", 7: "siedem", 8: "osiem", 9: "dziewięć", 10: "dziesięć",
    11: "jedenaście", 12: "dwanaście", 13: "trzynaście", 14: "czternaście",
    15: "piętnaście", 16: "szesnaście", 17: "siedemnaście", 18: "osiemnaście",
    19: "dziewiętnaście",
}
_TENS = {
    2: "dwadzieścia", 3: "trzydzieści", 4: "czterdzieści", 5: "pięćdziesiąt",
    6: "sześćdziesiąt", 7: "siedemdziesiąt", 8: "osiemdziesiąt", 9: "dziewięćdziesiąt",
}


def _int_to_words_pl(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_words_pl(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] if o == 0 else f"{_TENS[t]} {_ONES[o]}"
    if n < 1000:
        h, r = divmod(n, 100)
        hundreds = {
            1: "sto", 2: "dwieście", 3: "trzysta", 4: "czterysta", 5: "pięćset",
            6: "sześćset", 7: "siedemset", 8: "osiemset", 9: "dziewięćset",
        }
        head = hundreds.get(h, str(h))
        return head if r == 0 else f"{head} {_int_to_words_pl(r)}"
    return str(n)


_ABBREV = {
    r"\bdr\.": "doktor",
    r"\bprof\.": "profesor",
    r"\bnp\.": "na przykład",
    r"\bitd\.": "i tak dalej",
    r"\bitp\.": "i tym podobne",
    r"\bmln\.": "milionów",
    r"\bmlrd\.": "miliardów",
    r"\btys\.": "tysięcy",
    r"\bkm\.": "kilometrów",
    r"\bcm\.": "centymetrów",
    r"\bkg\.": "kilogramów",
}


def normalize_pl(text: str) -> str:
    """Light Polish normalization for more natural TTS prosody."""
    if not text or not text.strip():
        return text

    t = text.replace("\u00a0", " ")
    t = re.sub(r"\.{4,}", "...", t)
    t = re.sub(r"—", ", ", t)
    t = re.sub(r"–", ", ", t)

    for pat, repl in _ABBREV.items():
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)

    def _replace_num(m: re.Match) -> str:
        try:
            return _int_to_words_pl(int(m.group(0)))
        except ValueError:
            return m.group(0)

    t = re.sub(r"\b\d{1,4}\b", _replace_num, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_for_tts(text: str, lang_code: str) -> str:
    """Language-aware hook; Polish gets full normalization."""
    if (lang_code or "").strip().lower() == "pl":
        return normalize_pl(text)
    return text.strip()
