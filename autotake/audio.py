"""Utilitários de áudio com ffmpeg: conversões, envelope de energia, cortes e montagem."""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR_ANALYSIS = 16000
FRAME = 0.01  # 10 ms


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg falhou: {p.stderr[-600:]}")


def probe(path: Path) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration:stream=sample_rate,channels,codec_name",
                        "-select_streams", "a:0", "-of", "json", str(path)],
                       capture_output=True, text=True)
    info = json.loads(p.stdout or "{}")
    st = (info.get("streams") or [{}])[0]
    return {"duration": float(info.get("format", {}).get("duration", 0)),
            "sample_rate": int(st.get("sample_rate", 48000)),
            "channels": int(st.get("channels", 1)),
            "codec": st.get("codec_name", "")}


def make_transcription_copy(src: Path, dst: Path) -> None:
    """FLAC mono 16 kHz: sem perdas nem atraso de codec, e muito mais leve para enviar."""
    run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR_ANALYSIS), "-c:a", "flac", str(dst)])


def make_preview(src: Path, dst: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "96k", str(dst)])


def energy_envelope(src: Path) -> np.ndarray:
    """RMS em dB por janela de 10 ms, calculado em blocos para gastar pouca memória."""
    hop = int(SR_ANALYSIS * FRAME)
    block_frames = 6000                      # 60 s por bloco
    block_bytes = block_frames * hop * 2
    out: list[np.ndarray] = []
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(SR_ANALYSIS),
                             "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    leftover = b""
    try:
        while True:
            chunk = proc.stdout.read(block_bytes)
            if not chunk:
                break
            chunk = leftover + chunk
            usable = (len(chunk) // (hop * 2)) * hop * 2
            leftover = chunk[usable:]
            if usable == 0:
                continue
            x = np.frombuffer(chunk[:usable], dtype=np.int16).astype(np.float32) / 32768.0
            frames = x.reshape(-1, hop)
            rms = np.sqrt((frames * frames).mean(axis=1) + 1e-12)
            out.append((20 * np.log10(rms)).astype(np.float32))
    finally:
        proc.stdout.close()
        proc.wait()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def quietest_point(env: np.ndarray, t_from: float, t_to: float, prefer: float) -> float:
    """Ponto de menor energia entre t_from e t_to, com ligeira preferência por 'prefer'."""
    a, b = int(max(0, t_from) / FRAME), int(max(0, t_to) / FRAME)
    if b <= a or a >= len(env):
        return prefer
    seg = env[a:min(b, len(env))]
    dist = np.abs(np.arange(a, a + len(seg)) * FRAME - prefer)
    cost = seg + 4.0 * dist  # 4 dB por segundo de afastamento
    return float((a + int(np.argmin(cost))) * FRAME)


def refine_cut(env, words, i0, i1, pre=0.35, post=0.45) -> tuple[float, float]:
    s, e = words[i0]["s"], words[i1]["e"]
    prev_end = words[i0 - 1]["e"] if i0 > 0 else 0.0
    next_start = words[i1 + 1]["s"] if i1 + 1 < len(words) else e + post
    start = quietest_point(env, max(prev_end, s - pre), s - 0.02, s - 0.08)
    end = quietest_point(env, e + 0.02, min(next_start, e + post), e + 0.12)
    return round(start, 3), round(end, 3)


def render(src: Path, cuts: list[dict], dst: Path, fade: float = 0.012) -> None:
    """cuts: [{start, end, gap_after}] no tempo do ficheiro original. Saída WAV 24-bit."""
    info = probe(src)
    sr, ch = info["sample_rate"], info["channels"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = []
        for k, c in enumerate(cuts):
            dur = max(0.05, c["end"] - c["start"])
            out = tmp / f"p{k:04d}.wav"
            af = f"afade=t=in:d={fade},afade=t=out:st={max(0, dur - fade * 2):.3f}:d={fade * 2}"
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{c['start']:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
                 "-af", af, "-ar", str(sr), "-ac", str(ch), "-c:a", "pcm_s24le", str(out)])
            parts.append(out)
            if c.get("gap_after", 0) > 0:
                sil = tmp / f"s{k:04d}.wav"
                run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                     f"anullsrc=r={sr}:cl={'mono' if ch == 1 else 'stereo'}", "-t", f"{c['gap_after']:.3f}",
                     "-ac", str(ch), "-c:a", "pcm_s24le", str(sil)])
                parts.append(sil)
        lst = tmp / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(dst)])


def mix_track(items: list[dict], total: float, dst: Path) -> None:
    """Uma faixa de sound design como WAV 24-bit: cada som no seu instante (items: {file, start, gain}), com silêncio entre eles.
    O ficheiro tem 'total' segundos, para todas as faixas ficarem alinhadas quando se importam numa DAW."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
    for it in items:
        cmd += ["-i", str(it["file"])]
    parts, labels = [], ["[0:a]"]
    for k, it in enumerate(items, 1):
        ms = max(0, int(it["start"] * 1000))
        parts.append(f"[{k}:a]aformat=sample_rates=44100:channel_layouts=stereo,adelay={ms}|{ms},volume={it['gain']:.3f}[a{k}]")
        labels.append(f"[a{k}]")
    parts.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first:normalize=0:dropout_transition=0[out]")
    run(cmd + ["-filter_complex", ";".join(parts), "-map", "[out]", "-c:a", "pcm_s24le", str(dst)])
