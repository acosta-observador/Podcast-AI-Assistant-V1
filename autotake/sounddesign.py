"""Estrutura Plus: planeia e gera sound design (efeitos e música) a partir do guião.

As regras locais procuram no guião, já alinhado com os takes, as palavras que pedem som
(porta, tiro, comboio, mar...) e propõem uma cama de música. A ElevenLabs gera cada áudio.
Tudo fica numa linha do tempo medida na montagem final, não na gravação original.
"""
import json
import os
import re
import subprocess
from pathlib import Path

import httpx
import numpy as np

from .pipeline import build_cuts

SFX_URL = "https://api.elevenlabs.io/v1/sound-generation"
MUSIC_URL = "https://api.elevenlabs.io/v1/music"
LANES = ("Efeitos 1", "Efeitos 2", "Música")

# ---------- linha do tempo da montagem ----------
def timeline(result: dict, selections: dict) -> dict:
    """Posição de cada bloco e de cada corte na montagem final (segundos)."""
    cuts = build_cuts(result, selections)
    texts = {b["index"]: b["text"] for b in result["blocks"]}
    pos, spans, segs = 0.0, {}, []
    for c in cuts:
        length = c["end"] - c["start"]
        sp = spans.setdefault(c["block"], [pos, pos])
        segs.append({"start": round(pos, 3), "end": round(pos + length, 3),
                     "src": round(c["start"], 3), "block": c["block"]})
        pos += length
        sp[1] = pos
        pos += c["gap_after"]
    blocks = [{"block": i, "start": round(s, 3), "end": round(e, 3), "text": texts[i]}
              for i, (s, e) in sorted(spans.items())]
    return {"blocks": blocks, "segments": segs, "total": round(pos, 3)}


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


def plan_local(tl: dict) -> list[dict]:
    """Escolhe os sons sem LLM: procura palavras-chave no texto de cada bloco e põe uma cama de música."""
    cues, last = [], {}
    for b in tl["blocks"]:
        text, span = _norm(b["text"]), max(0.1, b["end"] - b["start"])
        hits = []
        for k, (pat, label, prompt, dur) in enumerate(RULES):
            m = re.search(pat, text)
            if m:
                hits.append((m.start(), k, label, prompt, dur))
        for pos, k, label, prompt, dur in sorted(hits)[:MAX_PER_BLOCK]:
            start = b["start"] + span * pos / max(1, len(text))
            start = max(b["start"], min(b["end"] - 0.5, start - 0.3))
            if start - last.get(k, -1e9) < MIN_REPEAT_GAP:
                continue
            last[k] = start
            cues.append({"kind": "sfx", "label": label, "prompt": prompt, "block": b["block"],
                         "start": round(start, 2), "dur": float(dur)})
    t, first = 0.0, tl["blocks"][0]["block"] if tl["blocks"] else 1
    total = tl["total"]
    while t < total - 3:
        part = min(MUSIC_PART, total - t)
        blk = next((b["block"] for b in tl["blocks"] if b["end"] > t), first)
        cues.append({"kind": "music", "label": "Música de fundo", "prompt": MUSIC_PROMPT, "block": blk,
                     "start": round(t, 2), "dur": round(max(10.0, part), 2)})
        t += part
    return cues


def assign_lanes(cues: list[dict]) -> None:
    busy: dict[int, list[tuple[float, float]]] = {0: [], 1: [], 2: []}
    for c in sorted(cues, key=lambda x: x["start"]):
        if c["kind"] == "music":
            c["lane"] = 2
        else:
            c["lane"] = 0 if all(c["start"] >= e or c["start"] + c["dur"] <= s for s, e in busy[0]) else 1
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
