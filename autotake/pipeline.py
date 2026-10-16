"""Orquestração do AutoTake para um episódio guardado numa pasta."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import audio
from .align import align
from .script_parser import parse_script
from .takes import build_takes
from .transcribe import load_or_transcribe, words_from_scribe

GAP_BETWEEN_BLOCKS = 2.0      # pausa entre parágrafos
GAP_BETWEEN_SCENES = 10.0     # pausa entre cenas
GAP_INSIDE_COMPOSITE = 0.25


def _files(ep: Path) -> dict:
    meta = json.loads((ep / "meta.json").read_text())
    return {
        "raw": ep / meta["audio_file"],
        "script": ep / meta["script_file"],
        "flac": ep / "work" / "transcribe.flac",
        "preview": ep / "work" / "preview.mp3",
        "transcript": ep / "work" / "transcript_scribe.json",
        "envelope": ep / "work" / "envelope.npy",
        "result": ep / "result.json",
    }


def process(ep: Path, api_key: str | None, status=lambda stage, msg="": None) -> dict:
    f = _files(ep)
    (ep / "work").mkdir(exist_ok=True)
    status("a preparar", "A converter o áudio")
    info = audio.probe(f["raw"])
    if not f["flac"].exists():
        audio.make_transcription_copy(f["raw"], f["flac"])
    if not f["preview"].exists():
        audio.make_preview(f["raw"], f["preview"])
    if not f["envelope"].exists():
        np.save(f["envelope"], audio.energy_envelope(f["flac"]))

    status("a transcrever", "A transcrever com o Scribe")
    raw = load_or_transcribe(f["flac"], f["transcript"], api_key)
    words = words_from_scribe(raw)

    status("a alinhar", "A alinhar com o guião")
    blocks = parse_script(f["script"])
    occs = align(blocks, words)
    build_takes(blocks, words, occs)

    env = np.load(f["envelope"])
    for b in blocks:
        for t in b["takes"]:
            t["cut"] = list(audio.refine_cut(env, words, t["i0"], t["i1"]))
        if b.get("composite"):
            b["composite"]["cuts"] = [list(audio.refine_cut(env, words, i0, i1))
                                      for i0, i1 in b["composite"]["segments"]]
    used = {i for o in occs for i in range(o.i0, o.i1 + 1)}
    off_script_words = sum(1 for i, w in enumerate(words) if w["t"] and i not in used)
    result = {
        "duration": info["duration"],
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "n_words": len(words),
        "off_script_words": off_script_words,
        "blocks": blocks,
    }
    f["result"].write_text(json.dumps(result, ensure_ascii=False))
    return result


def effective_choice(block: dict, selections: dict) -> dict | None:
    return selections.get(block["id"]) or block.get("auto")


def build_cuts(result: dict, selections: dict, scenes: list[dict] | None = None) -> list[dict]:
    """Converte as escolhas (auto ou do sonoplasta) em cortes no tempo do original.

    Entre parágrafos há GAP_BETWEEN_BLOCKS; antes do primeiro parágrafo de uma cena, GAP_BETWEEN_SCENES.
    'scenes' é a lista de cenas ({"blocks": [índices]}); sem ela só há pausas entre parágrafos.
    """
    per_block: list[tuple[int, list[dict]]] = []
    for b in result["blocks"]:
        choice = effective_choice(b, selections)
        if not choice:
            continue
        block_cuts = []
        if choice["mode"] == "composite" and b.get("composite"):
            for (s, e), n in zip(b["composite"]["cuts"], b["composite"]["takes"]):
                block_cuts.append({"start": s, "end": e, "gap_after": GAP_INSIDE_COMPOSITE,
                                   "block": b["index"], "take": n})
        else:
            t = next((t for t in b["takes"] if t["n"] == choice.get("take")), None)
            if not t:
                continue
            block_cuts.append({"start": t["cut"][0], "end": t["cut"][1], "gap_after": 0,
                               "block": b["index"], "take": t["n"]})
        per_block.append((b["index"], block_cuts))
    have = {i for i, _ in per_block}
    firsts = {next((x for x in sc["blocks"] if x in have), None) for sc in (scenes or [])}
    cuts = []
    for k, (_, bc) in enumerate(per_block):
        nxt = per_block[k + 1][0] if k + 1 < len(per_block) else None
        bc[-1]["gap_after"] = 0 if nxt is None else (GAP_BETWEEN_SCENES if nxt in firsts else GAP_BETWEEN_BLOCKS)
        cuts.extend(bc)
    return cuts


def render_episode(ep: Path, result: dict, selections: dict, dst: Path, scenes: list[dict] | None = None) -> None:
    audio.render(_files(ep)["raw"], build_cuts(result, selections, scenes), dst)
