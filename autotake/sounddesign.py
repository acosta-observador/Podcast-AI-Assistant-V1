"""Estrutura Plus: planeia e gera sound design (efeitos e música) a partir do guião.

As regras locais procuram no guião, já alinhado com os takes, as palavras que pedem som
(porta, tiro, comboio, mar...) e propõem uma cama de música. A ElevenLabs gera cada áudio.
Tudo fica numa linha do tempo medida na montagem final, não na gravação original.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import httpx
import numpy as np

from .pipeline import build_cuts
from .script_parser import _is_direction
from .text import tokenize

SFX_URL = "https://api.elevenlabs.io/v1/sound-generation"
MUSIC_URL = "https://api.elevenlabs.io/v1/music"
N_SFX_LANES = 4
MUSIC_LANE = N_SFX_LANES
LANE_NAMES = tuple(f"Efeitos {i + 1}" for i in range(N_SFX_LANES)) + ("Música",)
# pausas na linha do tempo da Estrutura Plus (a montagem do AutoTake continua com as suas)
BLOCK_GAP = 1.5
SCENE_GAP = 4.0
SCENE_RE = re.compile(r"^[\[(#\s]*(cena|scene)\s*\d+", re.I)
FALLBACK_SCENE_BLOCKS = 4

# ---------- cenas ----------
def _raw_items(path: Path) -> list[str]:
    if path.suffix.lower() == ".docx":
        import docx
        return [p.text for p in docx.Document(str(path)).paragraphs]
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if re.search(r"\n\s*\n", text):
        return [" ".join(c.split("\n")) for c in re.split(r"\n\s*\n", text)]
    return text.split("\n")


def scenes_for(script_path: Path, n_blocks: int) -> list[dict]:
    """Cenas do episódio: linhas 'CENA 1', 'CENA 2'... no guião; sem elas, grupos de 4 blocos."""
    starts, count = [], 0
    try:
        for item in _raw_items(script_path):
            s = item.strip()
            if SCENE_RE.match(s) and len(s) < 60:
                starts.append(count + 1)
            elif not _is_direction(s) and tokenize(s):
                count += 1
    except Exception:
        starts = []
    starts = sorted({x for x in starts if 1 <= x <= n_blocks})
    if not starts:
        starts = list(range(1, n_blocks + 1, FALLBACK_SCENE_BLOCKS))
    elif starts[0] != 1:
        starts.insert(0, 1)
    out = []
    for i, st in enumerate(starts):
        end = (starts[i + 1] - 1) if i + 1 < len(starts) else n_blocks
        out.append({"n": i + 1, "title": f"Cena {i + 1}", "blocks": list(range(st, end + 1))})
    return out


# ---------- linha do tempo da montagem ----------
def timeline(result: dict, selections: dict, scenes: list[dict] | None = None) -> dict:
    """Posição de cada bloco e de cada corte na linha do tempo (segundos).

    Entre blocos há uma pausa maior do que na montagem do AutoTake, e maior ainda entre cenas.
    """
    cuts = build_cuts(result, selections)
    texts = {b["index"]: b["text"] for b in result["blocks"]}
    scene_first = {sc["blocks"][0] for sc in (scenes or [])}
    pos, spans, segs, gaps = 0.0, {}, [], {}
    for k, c in enumerate(cuts):
        length = c["end"] - c["start"]
        sp = spans.setdefault(c["block"], [pos, pos])
        segs.append({"start": round(pos, 3), "end": round(pos + length, 3),
                     "src": round(c["start"], 3), "block": c["block"]})
        pos += length
        sp[1] = pos
        last = k == len(cuts) - 1 or cuts[k + 1]["block"] != c["block"]
        if last:
            nxt = cuts[k + 1]["block"] if k + 1 < len(cuts) else None
            gap = 0.0 if nxt is None else (SCENE_GAP if nxt in scene_first else BLOCK_GAP)
            gaps[c["block"]] = gap
            pos += gap
        else:
            pos += c["gap_after"]
    blocks = [{"block": i, "start": round(s, 3), "end": round(e, 3), "text": texts[i]}
              for i, (s, e) in sorted(spans.items())]
    by_block = {b["block"]: b for b in blocks}
    sc_out = []
    for sc in scenes or []:
        bs = [by_block[i] for i in sc["blocks"] if i in by_block]
        if bs:
            sc_out.append({"n": sc["n"], "title": sc["title"], "blocks": [b["block"] for b in bs],
                           "start": bs[0]["start"], "end": bs[-1]["end"]})
    return {"blocks": blocks, "segments": segs, "total": round(pos, 3), "gaps": gaps, "scenes": sc_out}


def narration_peaks(env: np.ndarray, blocks: list[dict], segments: list[dict], n_per_sec: float = 3.0) -> dict:
    """Alturas (0 a 100) para desenhar a onda de cada bloco, a partir do envelope de energia."""
    out = {}
    for b in blocks:
        vals = []
        for s in (x for x in segments if x["block"] == b["block"]):
            i0, i1 = int(s["src"] * 100), int((s["src"] + s["end"] - s["start"]) * 100)
            vals.append(_bins(env[i0:i1], max(4, round((s["end"] - s["start"]) * n_per_sec))))
        out[b["block"]] = [h for v in vals for h in v]
    return out


def _bins(db: np.ndarray, n: int) -> list[int]:
    if len(db) == 0:
        return [8] * n
    edges = np.linspace(0, len(db), n + 1, dtype=int)
    hs = []
    for a, b in zip(edges[:-1], edges[1:]):
        v = float(db[a:max(b, a + 1)].max())
        hs.append(int(max(8, min(100, (v + 60) / 60 * 100))))
    return hs


def audio_peaks(path: Path, n: int) -> list[int]:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                       capture_output=True)
    x = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) == 0:
        return [8] * n
    edges = np.linspace(0, len(x), n + 1, dtype=int)
    return [int(max(8, min(100, 100 * float(np.abs(x[a:max(b, a + 1)]).max())))) for a, b in zip(edges[:-1], edges[1:])]


# ---------- planeamento local (sem LLM): palavras-chave do guião ----------
# (padrão sobre o texto sem acentos e em minúsculas, nome, descrição para a ElevenLabs, duração em s)
RULES = [
    (r"\bporta(s)?\b", "Porta a abrir", "A heavy wooden door creaking open and then closing", 3),
    (r"\b(tiros?|disparos?|disparou|balas?)\b", "Tiro", "A single gunshot echoing in the distance", 2),
    (r"\b(comboios?|carruagens?|locomotiva)\b", "Comboio a passar", "A steam train passing by on the tracks with rhythmic clatter", 6),
    (r"\b(passos|caminhou|andou)\b", "Passos", "Slow footsteps on gravel", 3),
    (r"\b(sinos?|igrejas?)\b", "Sino de igreja", "A church bell tolling slowly", 5),
    (r"\b(mar|ondas?|maremoto|oceano|tsunami)\b", "Ondas do mar", "Large ocean waves crashing and receding", 6),
    (r"\b(fogo|incendios?|chamas|ardeu|arder|ardia)\b", "Fogo a crepitar", "A large fire crackling and roaring", 6),
    (r"\b(terramoto|sismo|abalo|tremer|tremor)\b", "Terra a tremer", "Deep low rumbling of an earthquake with debris falling", 6),
    (r"\b(multidao|gritos?|gritaram|povo|populacao)\b", "Multidão", "A frightened crowd shouting and murmuring in the distance", 5),
    (r"\b(chuva|trovoada|trovao|tempestade)\b", "Chuva e trovoada", "Heavy rain with distant thunder", 6),
    (r"\b(telefone|telefonou|chamada)\b", "Telefone", "An old telephone ringing", 3),
    (r"\b(carros?|automovel|viatura|viaturas)\b", "Carro a passar", "A car passing by on a road", 4),
    (r"\b(sirenes?|policia|ambulancia|ambulancias)\b", "Sirene", "A distant police siren", 5),
    (r"\bvento\b", "Vento", "Wind howling", 5),
    (r"\b(relogio|relogios)\b", "Relógio", "A ticking clock in a quiet room", 4),
    (r"\b(aviao|avioes|helicoptero)\b", "Avião", "An airplane passing overhead", 5),
    (r"\b(cao|caes|ladrar|ladrou)\b", "Cão a ladrar", "A dog barking in the distance", 3),
    (r"\b(explosao|explodiu|bomba|bombas)\b", "Explosão", "A loud explosion followed by a rumbling echo", 4),
    (r"\b(janela|vidros?)\b", "Vidro a partir", "Glass breaking", 2),
    (r"\b(sussurro|sussurrou|segredo)\b", "Sussurros", "Quiet indistinct whispering in a room", 4),
]
MUSIC_PROMPT = ("Subtle atmospheric instrumental underscore for a narrative documentary podcast, "
                "low intensity, slow, understated, no vocals, leaves room for a spoken voice")
MUSIC_PART = 120.0
MAX_PER_BLOCK = 3
MIN_REPEAT_GAP = 12.0


def _norm(t: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()


def plan_local(tl: dict, span: tuple | None = None) -> list[dict]:
    """Escolhe os sons sem LLM: procura palavras-chave no texto de cada bloco e põe uma cama de música."""
    cues, last = [], {}
    for b in tl["blocks"]:
        text, blen = _norm(b["text"]), max(0.1, b["end"] - b["start"])
        hits = []
        for k, (pat, label, prompt, dur) in enumerate(RULES):
            m = re.search(pat, text)
            if m:
                hits.append((m.start(), k, label, prompt, dur))
        for pos, k, label, prompt, dur in sorted(hits)[:MAX_PER_BLOCK]:
            start = b["start"] + blen * pos / max(1, len(text))
            start = max(b["start"], min(b["end"] - 0.5, start - 0.3))
            if start - last.get(k, -1e9) < MIN_REPEAT_GAP:
                continue
            last[k] = start
            cues.append({"kind": "sfx", "label": label, "prompt": prompt, "block": b["block"],
                         "start": round(start, 2), "dur": float(dur)})
    t, first = (span[0] if span else 0.0), tl["blocks"][0]["block"] if tl["blocks"] else 1
    total = span[1] if span else tl["total"]
    while t < total - 3:
        part = min(MUSIC_PART, total - t)
        blk = next((b["block"] for b in tl["blocks"] if b["end"] > t), first)
        cues.append({"kind": "music", "label": "Música de fundo", "prompt": MUSIC_PROMPT, "block": blk,
                     "start": round(t, 2), "dur": round(max(10.0, part), 2)})
        t += part
    return cues


def assign_lanes(cues: list[dict]) -> None:
    busy: dict[int, list[tuple[float, float]]] = {k: [] for k in range(N_SFX_LANES + 1)}
    for c in sorted(cues, key=lambda x: x["start"]):
        if c["kind"] == "music":
            c["lane"] = MUSIC_LANE
        else:
            c["lane"] = next((k for k in range(N_SFX_LANES)
                              if all(c["start"] >= e or c["start"] + c["dur"] <= s0 for s0, e in busy[k])),
                             N_SFX_LANES - 1)
        busy[c["lane"]].append((c["start"], c["start"] + c["dur"]))


# ---------- geração com a ElevenLabs ----------
def generate(cue: dict, eleven_key: str, dst: Path) -> None:
    headers = {"xi-api-key": eleven_key}
    if cue["kind"] == "music":
        r = httpx.post(MUSIC_URL, headers=headers, timeout=300,
                       json={"prompt": cue["prompt"], "music_length_ms": int(max(3, min(cue["dur"], 300)) * 1000)})
    else:
        r = httpx.post(SFX_URL, headers=headers, timeout=120,
                       json={"text": cue["prompt"], "duration_seconds": round(max(0.5, min(cue["dur"], 22)), 1),
                             "prompt_influence": 0.4})
    if r.status_code != 200:
        raise RuntimeError(f"A ElevenLabs devolveu {r.status_code}: {r.text[:300]}")
    dst.write_bytes(r.content)
