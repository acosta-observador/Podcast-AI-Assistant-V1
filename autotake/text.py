"""Normalização de texto partilhada pelo guião e pela transcrição."""
import re
import unicodedata

try:
    from num2words import num2words
except ImportError:  # pragma: no cover
    num2words = None

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
HESITATIONS = {"eh", "ehm", "hum", "hmm", "ahm", "ah", "uh", "um", "mm"}


def _num_to_words(match: re.Match) -> str:
    raw = match.group(0)
    if num2words is None:
        return raw
    try:
        value = float(raw.replace(",", ".")) if ("," in raw or "." in raw) else int(raw)
        return " " + num2words(value, lang="pt") + " "
    except Exception:
        return raw


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    """Minúsculas, números por extenso, sem pontuação nem acentos."""
    t = unicodedata.normalize("NFC", text).lower()
    t = t.replace("%", " por cento ").replace("€", " euros ").replace("&", " e ")
    t = _NUM.sub(_num_to_words, t)
    t = t.replace("-", " ").replace("’", "'").replace("'", " ")
    t = _NON_WORD.sub(" ", t)
    t = strip_accents(t)
    return [w for w in t.split() if w]
