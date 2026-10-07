"""Exportação para DAW: projeto Reaper (.rpp) com montagem, alternativas e regiões."""
from pathlib import Path

from .pipeline import build_cuts, effective_choice

_SOURCE_TYPES = {".wav": "WAVE", ".mp3": "MP3", ".flac": "FLAC", ".ogg": "VORBIS",
                 ".aif": "WAVE", ".aiff": "WAVE", ".m4a": "VIDEO", ".mp4": "VIDEO"}


def _q(s: str) -> str:
    return '"' + s.replace('"', "'") + '"'


def _item(pos, length, soffs, name, src_name, src_type, mute=False) -> str:
    return (f"    <ITEM\n      POSITION {pos:.4f}\n      LENGTH {length:.4f}\n"
            f"      FADEIN 1 0.01 0\n      FADEOUT 1 0.02 0\n      MUTE {1 if mute else 0} 0\n"
            f"      SOFFS {soffs:.4f}\n      NAME {_q(name)}\n"
            f"      <SOURCE {src_type}\n        FILE {_q(src_name)}\n      >\n    >\n")


def to_reaper(result: dict, selections: dict, audio_name: str, title: str, sound: list | None = None) -> str:
    src_type = _SOURCE_TYPES.get(Path(audio_name).suffix.lower(), "WAVE")
    cuts = build_cuts(result, selections)
    blocks = {b["index"]: b for b in result["blocks"]}
    main, markers = [], []
    alt_tracks: dict[int, list[str]] = {}
    pos, block_start, current = 0.0, {}, None
    for c in cuts:
        if c["block"] != current:
            current = c["block"]
            block_start[current] = pos
        length = c["end"] - c["start"]
        main.append(_item(pos, length, c["start"], f"Bloco {c['block']} take {c['take']}", audio_name, src_type))
        pos += length + c["gap_after"]
        b = blocks[c["block"]]
        if c is cuts[-1] or cuts[cuts.index(c) + 1]["block"] != current:
            choice = effective_choice(b, selections) or {}
            review = b["confidence"] != "alta"
            label = f"{'REVER ' if review else ''}Bloco {current}"
            idx = len(markers) // 2 + 1
            markers.append(f"  MARKER {idx} {block_start[current]:.4f} {_q(label)} 1\n")
            markers.append(f"  MARKER {idx} {pos - c['gap_after']:.4f} \"\" 1\n")
            chosen = choice.get("take") if choice.get("mode") == "take" else None
            alts = [t for t in b["takes"] if t["n"] != chosen]
            for k, t in enumerate(alts):
                alt_tracks.setdefault(k, []).append(
                    _item(block_start[current], t["cut"][1] - t["cut"][0], t["cut"][0],
                          f"Bloco {current} take {t['n']}{'' if t['complete'] else ' (incompleto)'}",
                          audio_name, src_type))
    out = [f'<REAPER_PROJECT 0.1 "7.0" 0\n  TITLE {_q(title)}\n'] + markers
    out.append('  <TRACK\n    NAME "AutoTake"\n' + "".join(main) + "  >\n")
    for k in sorted(alt_tracks):
        out.append(f'  <TRACK\n    NAME "Alternativa {k + 1}"\n    MUTESOLO 1 0 0\n'
                   + "".join(alt_tracks[k]) + "  >\n")
    for k, lane in enumerate(("Efeitos 1", "Efeitos 2", "Música")):
        items = [_item(c["start"], c["dur"], 0, c["label"], f"sd/{c['id']}.mp3", "MP3")
                 for c in (sound or []) if c.get("lane") == k]
        if items:
            out.append(f'  <TRACK\n    NAME {_q(lane)}\n' + "".join(items) + "  >\n")
    out.append(f'  <TRACK\n    NAME "Original"\n    MUTESOLO 1 0 0\n'
               + _item(0, result["duration"], 0, "Gravação original", audio_name, src_type) + "  >\n")
    out.append(">\n")
    return "".join(out)
