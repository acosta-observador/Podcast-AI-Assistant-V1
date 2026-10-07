"""Estrutura Plus: planeia e gera sound design (efeitos e música) a partir do guião.

O Claude lê o guião, já alinhado com os takes, e propõe os momentos de sound design.
A ElevenLabs gera cada áudio (efeitos e música). Tudo fica numa linha do tempo
medida na montagem final, não na gravação original.
"""
import json
import os
import re
import subprocess
from pathlib import Path

import httpx
import numpy as np

from .pipeline import build_cuts

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
SFX_URL = "https://api.elevenlabs.io/v1/sound-generation"
MUSIC_URL = "https://api.elevenlabs.io/v1/music"
MAX_CUES = 40
LANES = ("Efeitos 1", "Efeitos 2", "Música")

SYSTEM = """És sonoplasta de um podcast narrativo em português (Podcast Plus do Observador).
Recebes os blocos do guião com o tempo de início e fim de cada um, em segundos, na montagem final.
Propõe o sound design: efeitos sonoros curtos que ilustram o que a narração descreve e, se fizer sentido,
camas de música. Responde APENAS com um array JSON, sem texto à volta. Cada elemento:
{"block": nº do bloco onde começa, "kind": "sfx" ou "music", "label": nome curto em português (2 a 4 palavras),
 "prompt": descrição do som EM INGLÊS para um gerador de áudio (concreta, sonora, sem nomes próprios),
 "at": segundos depois do início do bloco em que entra (só sfx), "duration": segundos (sfx 1 a 10),
 "until_block": nº do bloco onde a música acaba (só music)}.
Regras: no máximo 3 efeitos por bloco e só quando acrescentam algo audível (portas, passos, tiros, comboios,
multidões, mar, fogo, sinos). Não repitas o mesmo som sem necessidade. A música não pode tapar a voz:
descreve-a como cama discreta. Não ponhas efeitos sobre indicações entre parênteses retos."""


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


# ---------- planeamento com o Claude ----------
def plan_cues(tl: dict, anthropic_key: str) -> list[dict]:
    lines = [f"Bloco {b['block']} ({b['start']:.1f}s a {b['end']:.1f}s): {b['text']}" for b in tl["blocks"]]
    body = {"model": ANTHROPIC_MODEL, "max_tokens": 4096, "system": SYSTEM,
            "messages": [{"role": "user", "content": "\n".join(lines)}]}
    r = httpx.post(ANTHROPIC_URL, json=body, timeout=120,
                   headers={"x-api-key": anthropic_key, "anthropic-version": "2023-06-01"})
    if r.status_code != 200:
        raise RuntimeError(f"A Anthropic devolveu {r.status_code}: {r.text[:300]}")
    text = "".join(p.get("text", "") for p in r.json().get("content", []))
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise RuntimeError("O Claude não devolveu uma lista de sons.")
    raw = json.loads(m.group(0))
    by_block = {b["block"]: b for b in tl["blocks"]}
    cues = []
    for c in raw[:MAX_CUES]:
        b = by_block.get(int(c.get("block", 0)))
        if not b or c.get("kind") not in ("sfx", "music") or not c.get("prompt"):
            continue
        if c["kind"] == "music":
            end_b = by_block.get(int(c.get("until_block") or c["block"]), b)
            start, dur = b["start"], max(10.0, end_b["end"] - b["start"])
            dur = min(dur, 120.0)
        else:
            start = min(b["end"] - 0.5, b["start"] + max(0.0, float(c.get("at", 0))))
            dur = max(1.0, min(10.0, float(c.get("duration", 3))))
        cues.append({"kind": c["kind"], "label": str(c.get("label", "Som"))[:40], "prompt": str(c["prompt"])[:400],
                     "block": b["block"], "start": round(start, 2), "dur": round(dur, 2)})
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
