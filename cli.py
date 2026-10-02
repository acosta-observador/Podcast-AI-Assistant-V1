"""Correr o AutoTake sem o site, útil para afinar o algoritmo.

  python cli.py pasta_do_episodio            # processa (usa a transcrição em cache se existir)
  python cli.py pasta_do_episodio --render   # e monta o WAV

A pasta precisa de um meta.json com audio_file e script_file.
"""
import argparse
import json
import os
from pathlib import Path

from autotake.pipeline import process, render_episode

ap = argparse.ArgumentParser()
ap.add_argument("episode")
ap.add_argument("--render", action="store_true")
a = ap.parse_args()
ep = Path(a.episode)
r = process(ep, os.getenv("ELEVENLABS_API_KEY"), lambda s, m="": print(f"> {s}"))
for b in r["blocks"]:
    print(f"\nBloco {b['index']}: {b['auto']}  confiança {b['confidence']}")
    for f in b["flags"]:
        print(f"   ! {f}")
    for t in b["takes"]:
        print(f"   take {t['n']}: {'completo' if t['complete'] else 'incompleto'}  cobertura {t['coverage']:.0%}  "
              f"fora do guião {t['off_script']}  {t['cut'][0]:.1f}s a {t['cut'][1]:.1f}s")
if a.render:
    sel = json.loads((ep / "selections.json").read_text()) if (ep / "selections.json").exists() else {}
    render_episode(ep, r, sel, ep / "montagem.wav")
    print("\nMontagem em", ep / "montagem.wav")
