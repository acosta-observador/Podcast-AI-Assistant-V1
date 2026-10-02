"""Guião → blocos (parágrafos) → frases."""
import re
from pathlib import Path

from .text import tokenize

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"«“(\wÀ-ÿ])")
MIN_SENT_TOKENS = 4


def _is_direction(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if (s[0] in "[(" and s[-1] in "])") or s.startswith("#") or s.startswith("//"):
        return True
    letters = [c for c in s if c.isalpha()]
    # Linhas curtas todas em maiúsculas costumam ser indicações (GENÉRICO, SOM, VOZ 2…)
    if letters and len(s) < 40 and all(c.isupper() for c in letters):
        return True
    return False


def read_paragraphs(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        import docx  # python-docx
        doc = docx.Document(str(path))
        lines = [p.text for p in doc.paragraphs]
        return [l.strip() for l in lines if not _is_direction(l)]
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if re.search(r"\n\s*\n", text):
        chunks = re.split(r"\n\s*\n", text)
        paras = [" ".join(c.split("\n")) for c in chunks]
    else:
        paras = text.split("\n")
    return [p.strip() for p in paras if not _is_direction(p)]


def split_sentences(paragraph: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(paragraph) if p.strip()]
    merged: list[str] = []
    for p in parts:
        # Frases muito curtas ("Sim.") não se alinham bem sozinhas: junta à vizinha
        if merged and (len(tokenize(p)) < MIN_SENT_TOKENS or len(tokenize(merged[-1])) < MIN_SENT_TOKENS):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def parse_script(path: Path) -> list[dict]:
    blocks = []
    for para in read_paragraphs(path):
        if not tokenize(para):
            continue
        sentences = []
        for s in split_sentences(para):
            toks = tokenize(s)
            if toks:
                sentences.append({"text": s, "tokens": toks})
        blocks.append({
            "id": f"b{len(blocks) + 1}",
            "index": len(blocks) + 1,
            "text": para,
            "sentences": sentences,
            "n_tokens": sum(len(s["tokens"]) for s in sentences),
        })
    return blocks
