"""Exportação AAF para o Pro Tools: uma faixa de narração (um clip por take) e as faixas de sons, com o áudio embebido.

O áudio vai embebido em PCM a 48 kHz (o volume de cada som já vem aplicado), em 16 ou 24 bits e em little-endian (WAV) ou big-endian
(estilo AIFF), porque o Pro Tools é esquisito com o áudio embebido num AAF. Importa-se com File > Import > Session Data...
"""
from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import aaf2

from . import audio

SR = 48000


def pcm_wav(src: Path, dst: Path, start: float = 0.0, dur: float | None = None, gain: float = 1.0, fade: float = 0.012, bits: int = 16) -> int:
    """Corta e converte para WAV PCM a 48 kHz, 16 ou 24 bits (cabeçalho simples, sem 'extensible'). Devolve o nº de amostras."""
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
    cmd += (["-af", ",".join(af)] if af else []) + ["-ac", str(ch), "-ar", str(SR), "-f", "s24le" if bits == 24 else "s16le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(f"ffmpeg falhou: {p.stderr[-300:].decode(errors='ignore')}")
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(3 if bits == 24 else 2)
        w.setframerate(SR)
        w.writeframes(p.stdout)
    return len(p.stdout) // ((3 if bits == 24 else 2) * ch)


def _swap(data: bytes, width: int) -> bytes:
    b = bytearray(data)
    if width == 2:
        b[0::2], b[1::2] = data[1::2], data[0::2]
    else:
        b[0::3], b[2::3] = data[2::3], data[0::3]
    return bytes(b)


def import_pcm(f, mob, wav: Path, big_endian: bool):
    """Como MasterMob.import_audio_essence, mas com a opção de gravar o áudio em big-endian."""
    a = aaf2.audio.WaveReader(str(wav))
    rate, ch, width, frames = a.getframerate(), a.getnchannels(), a.getsampwidth(), a.getnframes()
    src_mob = f.create.SourceMob(f"{mob.name}.PHYS")
    f.content.mobs.append(src_mob)
    essence, src_slot = src_mob.create_essence(rate, "sound")
    d = f.create.PCMDescriptor()
    src_mob.descriptor = d
    d["Channels"].value = ch
    d["BlockAlign"].value = a.getblockalign()
    d["SampleRate"].value = rate
    d["AverageBPS"].value = rate * ch * width
    d["QuantizationBits"].value = width * 8
    d["AudioSamplingRate"].value = rate
    d.length = frames
    src_slot.segment.length = frames
    stream = essence.open("w")
    while True:
        data = a.readframes(rate)
        if not data:
            break
        stream.write(_swap(data, width) if big_endian else data)
    slot = mob.create_timeline_slot(edit_rate=rate)
    slot.segment = src_mob.create_source_clip(src_slot.slot_id, media_kind="sound")
    slot.segment.length = frames
    return slot


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


def build_aaf(dst: Path, name: str, tracks: list[dict], progress=lambda done, total: None, bits: int = 16, big_endian: bool = False) -> None:
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
                n = pcm_wav(Path(c["src"]), wav, c.get("start", 0.0), c.get("dur"), c.get("gain", 1.0), bits=bits)
                start = int(round(c["pos"] * SR))
                if start > pos:
                    seq.components.append(f.create.Filler("sound", start - pos))
                    pos = start
                mob = f.create.MasterMob(c["name"])
                f.content.mobs.append(mob)
                src_slot = import_pcm(f, mob, wav, big_endian)
                seq.components.append(mob.create_source_clip(src_slot.slot_id, start=0, length=n))
                pos += n
                wav.unlink(missing_ok=True)
                done += 1
                progress(done, total)
