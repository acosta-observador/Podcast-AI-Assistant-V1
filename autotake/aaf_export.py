"""Exportação AAF para o Pro Tools: uma faixa de narração (um clip por take) e as faixas de sons, com o áudio embebido.

Todo o áudio vai em WAV PCM 24-bit a 48 kHz (o volume de cada som já vem aplicado). Importa-se no Pro Tools com
File > Import > Session Data... e escolhendo o ficheiro .aaf (Copy from source media).
"""
from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import aaf2

from . import audio

SR = 48000


def pcm24(src: Path, dst: Path, start: float = 0.0, dur: float | None = None, gain: float = 1.0, fade: float = 0.012) -> int:
    """Corta e converte para WAV PCM 24-bit 48 kHz (cabeçalho simples, sem 'extensible'). Devolve o nº de amostras."""
    try:
        ch = max(1, min(2, int(audio.probe(src).get("channels") or 1)))
    except Exception:
        ch = 1
    af = []
    if gain and abs(gain - 1.0) > 1e-3:
        af.append(f"volume={gain:.3f}")
    if dur:
        af.append(f"afade=t=in:d={fade},afade=t=out:st={max(0.0, dur - fade * 2):.3f}:d={fade * 2}")
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", f"{start:.3f}"] if start else []) + ["-i", str(src)] + (["-t", f"{dur:.3f}"] if dur else [])
    cmd += (["-af", ",".join(af)] if af else []) + ["-ac", str(ch), "-ar", str(SR), "-f", "s24le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(f"ffmpeg falhou: {p.stderr[-300:].decode(errors='ignore')}")
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(p.stdout)
    return len(p.stdout) // (3 * ch)


def split_overlaps(clips: list[dict]) -> list[list[dict]]:
    """Uma faixa AAF é uma sequência: sons que se sobrepõem vão para faixas extra."""
    rows: list[tuple[float, list[dict]]] = []
    for c in sorted(clips, key=lambda x: x["pos"]):
        for i, (end, items) in enumerate(rows):
            if c["pos"] >= end - 1e-6:
                items.append(c)
                rows[i] = (c["pos"] + c["dur"], items)
                break
        else:
            rows.append((c["pos"] + c["dur"], [c]))
    return [items for _, items in rows]


def build_aaf(dst: Path, name: str, tracks: list[dict], progress=lambda done, total: None) -> None:
    """tracks: [{"name", "clips": [{"name", "src", "start", "dur", "gain", "pos"}]}]; 'pos' e 'dur' em segundos,
    'start' é o ponto de corte no ficheiro de origem."""
    laid = []
    for t in tracks:
        for k, items in enumerate(split_overlaps(t["clips"])):
            laid.append({"name": t["name"] if k == 0 else f"{t['name']} ({k + 1})", "clips": items})
    total = sum(len(t["clips"]) for t in laid)
    done = 0
    with tempfile.TemporaryDirectory() as tmp, aaf2.open(str(dst), "w") as f:
        comp = f.create.CompositionMob(name)
        f.content.mobs.append(comp)
        for t in laid:
            slot = comp.create_sound_slot(edit_rate=SR)
            slot.name = t["name"]
            seq, pos = slot.segment, 0
            for c in t["clips"]:
                wav = Path(tmp) / "clip.wav"
                n = pcm24(Path(c["src"]), wav, c.get("start", 0.0), c.get("dur"), c.get("gain", 1.0))
                start = int(round(c["pos"] * SR))
                if start > pos:
                    seq.components.append(f.create.Filler("sound", start - pos))
                    pos = start
                mob = f.create.MasterMob(c["name"])
                f.content.mobs.append(mob)
                src_slot = mob.import_audio_essence(str(wav), SR)
                seq.components.append(mob.create_source_clip(src_slot.slot_id, start=0, length=n))
                pos += n
                wav.unlink(missing_ok=True)
                done += 1
                progress(done, total)
