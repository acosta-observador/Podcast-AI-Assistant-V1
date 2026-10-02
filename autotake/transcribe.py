"""Transcrição com timestamps por palavra via ElevenLabs Scribe."""
import json
import os
from pathlib import Path

import httpx

from .text import tokenize

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
SCRIBE_MODEL = os.getenv("SCRIBE_MODEL", "scribe_v2")


def transcribe_scribe(audio_path: Path, api_key: str, language: str = "por",
                      keyterms: list[str] | None = None) -> dict:
    data = {
        "model_id": SCRIBE_MODEL,
        "language_code": language,
        "timestamps_granularity": "word",
        "tag_audio_events": "true",
        "diarize": "false",
    }
    files = {"file": (audio_path.name, open(audio_path, "rb"), "application/octet-stream")}
    form = list(data.items())
    for term in (keyterms or [])[:100]:
        form.append(("keyterms", term))
    try:
        with httpx.Client(timeout=httpx.Timeout(3600, connect=30)) as client:
            r = client.post(SCRIBE_URL, headers={"xi-api-key": api_key}, data=form, files=files)
    finally:
        files["file"][1].close()
    if r.status_code != 200:
        raise RuntimeError(f"Scribe devolveu {r.status_code}: {r.text[:400]}")
    return r.json()


def words_from_scribe(raw: dict) -> list[dict]:
    """Lista plana de tokens normalizados com início/fim, pronta para alinhar.

    Cada token guarda o texto original (para mostrar) e o normalizado (para comparar).
    Palavras que dão vários tokens (ex.: "2026") dividem o intervalo em partes iguais.
    """
    out = []
    for w in raw.get("words", []):
        kind = w.get("type", "word")
        if kind == "spacing":
            continue
        start, end = float(w.get("start", 0)), float(w.get("end", 0))
        if kind == "audio_event":
            out.append({"t": "", "raw": w.get("text", ""), "s": start, "e": end, "event": True})
            continue
        toks = tokenize(w.get("text", ""))
        if not toks:
            continue
        step = (end - start) / len(toks) if end > start else 0
        for i, tok in enumerate(toks):
            out.append({
                "t": tok,
                "raw": w.get("text", "") if i == 0 else "",
                "s": round(start + i * step, 3),
                "e": round(start + (i + 1) * step, 3) if step else end,
            })
    return out


def load_or_transcribe(audio_path: Path, cache_path: Path, api_key: str | None,
                       keyterms: list[str] | None = None) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    if not api_key:
        raise RuntimeError("Falta a chave da API do Scribe (ELEVENLABS_API_KEY).")
    raw = transcribe_scribe(audio_path, api_key, keyterms=keyterms)
    cache_path.write_text(json.dumps(raw, ensure_ascii=False))
    return raw
