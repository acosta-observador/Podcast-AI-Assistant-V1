"""Guião → blocos (parágrafos) → frases."""
import re
from pathlib import Path

from .text import tokenize

SCENE_RE = re.compile(r"^[\[(#\s]*(cena|scene)\s*\d+", re.I)
ELLIPSIS = re.compile(r"^\s*(…|\.{3})\s*")
INTERVIEW_MARK = re.compile(r"\bSOM\s*\.?\s*\d+|\bEP\.?\s*\d|ARQUIVO|\(\s*\d{1,2}:\d{2}", re.I)
QUOTE_START = ("\"", "“", "«")
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


def _raw_items(path: Path) -> list[str]:
    """Parágrafos do ficheiro tal como estão (inclui linhas vazias)."""
    if path.suffix.lower() == ".docx":
        import docx  # python-docx
        return [p.text for p in docx.Document(str(path)).paragraphs]
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if re.search(r"\n\s*\n", text):
        return [" ".join(c.split("\n")) for c in re.split(r"\n\s*\n", text)]
    return text.split("\n")


def classify_items(path: Path) -> list[dict]:
    """Guião → itens ordenados {id, type, text, speaker?}.

    narracao: o que o narrador lê. entrevista: excerto de entrevista ou arquivo (marcado por uma linha
    '… NOME EP4 SOM 1 (01:52)'). titulo: títulos e avisos de produção (DISCLAIMERS, TEASER, CENA 1...).
    nota: indicações da equipa. Só a narração tem takes no AutoTake.
    """
    items: list[dict] = []
    expecting, speaker = False, ""

    def add(kind: str, text: str, **kw):
        items.append({"id": len(items) + 1, "type": kind, "text": text, **kw})

    for raw in _raw_items(path):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("http") or (s[0] in "[(" and s[-1] in "])"):
            if s[0] in "[(" and not s.lstrip("[(").startswith("http"):
                add("nota", s)
            continue
        if SCENE_RE.match(s) and len(s) < 140:
            add("titulo", s)
            expecting = False
            continue
        if ELLIPSIS.match(s):
            body = ELLIPSIS.sub("", s).strip()
            if INTERVIEW_MARK.search(body):
                expecting, speaker = True, body
                add("nota", s, marker="entrevista")
            else:
                add("nota", s)
            continue
        letters = [c for c in s if c.isalpha()]
        words = [w for w in re.findall(r"[^\W\d_]+", s) if len(w) > 2][:2]
        if s.startswith(("#", "//")) or (len(s) < 90 and ((letters and all(c.isupper() for c in letters))
                                                          or (len(words) == 2 and all(w.isupper() for w in words)))):
            add("titulo", s)
            continue
        toks = tokenize(s)
        if not toks:
            continue
        if len(toks) <= 2 and not re.search(r"[.!?…\"”»]$", s):      # 'Episódio', '4': rótulos soltos, não se lêem
            add("titulo", s)
            continue
        if expecting:
            add("entrevista", s, speaker=speaker)
            expecting = False
        elif s.startswith(QUOTE_START) and items and items[-1]["type"] == "entrevista":
            add("entrevista", s, speaker=items[-1].get("speaker", ""))
        else:
            add("narracao", s)
    return items


def read_paragraphs(path: Path) -> list[str]:
    """Só a narração: títulos, notas e entrevistas não levam takes."""
    paras = [it["text"] for it in classify_items(path) if it["type"] == "narracao"]
    if paras:
        return paras
    return [p.strip() for p in _raw_items(path) if not _is_direction(p)]  # guião sem estrutura reconhecível


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
