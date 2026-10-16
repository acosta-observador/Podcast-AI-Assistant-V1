"""Maestro: o Claude lê cada cena e decide as músicas e os stems; a ElevenLabs gera-os.

Cada cena leva 3 a 5 peças (mais ou menos, conforme o tamanho): camas completas, stems de um só elemento (por exemplo só
uma percussão de tique-taque), stingers e transições. Cada prompt é diferente dos outros.
"""
from __future__ import annotations

import re
from pathlib import Path

import httpx

from . import analysis, sounddesign as sd

STEMS_URL = "https://api.elevenlabs.io/v1/music/stem-separation"
ROLES = ("bed", "stem", "stinger", "transition")
ROLE_NAMES = {"bed": "CAMA", "stem": "STEM", "stinger": "STINGER", "transition": "TRANSIÇÃO"}
DUR_LIMITS = {"bed": (20, 120), "stem": (20, 90), "stinger": (3, 15), "transition": (5, 20)}
MAX_PIECES = 8

MAESTRO_SYSTEM = """És o compositor e supervisor musical de um podcast true crime em português (Podcast Plus, Observador). Recebes o texto de UMA
cena e decides as músicas e os stems que a ElevenLabs vai gerar para ela. Cada peça é gerada por uma IA de música a partir de um prompt em
inglês, por isso o prompt é tudo.

ESTILO: true crime, sombrio e tenso, mas contido: NÃO épico, NÃO orquestral de trailer. Paleta: pads eletrónicos e sintetizadores analógicos,
pianos (piano de feltro, piano preparado, piano esparso), percussão suave (tique-taque, shakers abafados, caixas com escovas, clicks, glitches),
sub-graves discretos, guitarras elétricas escuras e distantes, cordas em pizzicato ou com surdina, vibrafone com arco, caixa de música, texturas de
vinil, vozes sem palavras muito ao fundo. A música serve a narração: fica por baixo da voz.

QUANTAS: de 3 a 5 peças numa cena normal; 2 a 3 numa cena curta (menos de 40 s de narração); até 7 numa cena longa (mais de 3 minutos).
CADA PEÇA É DIFERENTE DAS OUTRAS: instrumentação, textura, andamento e carácter distintos. É PROIBIDO repetir a mesma ideia com palavras
diferentes e é PROIBIDO "drone sombrio" ou "pulso grave" como ideia principal: tem de haver instrumentos e movimento concretos. Acompanha o
arco emocional da cena: o que a música faz no princípio, na viragem e no fecho.

PAPÉIS:
- "bed": peça completa e contínua (harmonia, textura e ritmo), 30 a 90 s. É o corpo musical de um troço da cena.
- "stem": UM só elemento isolado, para sobrepor a uma cama, 30 a 60 s. O prompt tem de dizer que é só esse elemento ("... only, no other
  instruments, no melody"), por exemplo "soft ticking mysterious percussion only, dry and quiet, no other instruments".
- "stinger": acento curto, 4 a 10 s, numa viragem ou revelação (um acorde de piano com cauda longa, um impacto surdo com textura metálica).
- "transition": 8 a 15 s, para entrar na cena seguinte ou sair desta (riser discreto, cauda invertida de piano, glitch que desvanece).

PROMPTS (campo "prompt", em inglês, 20 a 45 palavras): em camadas, como a descrição de uma banda sonora: ritmo ou percussão, harmonia e clima,
instrumento em destaque com a sua posição na mistura, o género ou a referência de ecrã ("for a detective TV show", "nordic noir", "true crime
documentary underscore") e o carácter (mysterious, tense, restrained). Sem nomes de artistas. Sempre instrumental. Exemplo:
"soft ticking percussion, ominous harmony for a detective TV show, mysterious, a gentle dark electric guitar in the distance".

"item": o número do parágrafo (o número entre parênteses retos) onde a peça entra. Ordena as peças pela posição. As peças podem sobrepor-se (um stem
por cima de uma cama). Os excertos de ENTREVISTA não levam música nova: não faças nenhuma peça entrar numa entrevista.

Responde APENAS com JSON:
{"direction": "o arco musical da cena em 1 ou 2 frases",
 "pieces": [{"role": "bed|stem|stinger|transition", "label": "nome curto em português (2 a 4 palavras)",
   "description": "o que a peça faz na cena, em português, 1 frase", "prompt": "...", "item": n, "duration": segundos, "gain": 0.2 a 0.6}]}"""


def plan_scene(an: dict, n: int, key: str, notes: str = "") -> dict:
    """O Claude lê a cena e propõe as peças. Devolve {"direction", "pieces": [...]}."""
    sc = next(x for x in an["scenes"] if x["n"] == n)
    mine = [it for it in an["items"] if sc["from"] <= it["id"] <= sc["to"] and it["type"] in analysis.VISIBLE]
    tag = {"narracao": "NARRAÇÃO", "entrevista": "ENTREVISTA"}
    words = sum(len(it["text"].split()) for it in mine if it["type"] == "narracao")
    prev = an["scenes"][n - 2]["title"] if n > 1 else None
    nxt = an["scenes"][n]["title"] if n < len(an["scenes"]) else None
    msg = (f"Cena {sc['n']}: {sc['title']}\n(cerca de {max(10, round(words / 2.6))} segundos de narração, {words} palavras)\n"
           + (f"Cena anterior: {prev}\n" if prev else "") + (f"Cena seguinte: {nxt}\n" if nxt else "") + "\n"
           + "\n\n".join(f"[{it['id']}] {tag[it['type']]}{' · ' + it['speaker'] if it.get('speaker') else ''}: {it['text'][:1500]}" for it in mine))
    if notes.strip():
        msg += f"\n\nNOTAS DO UTILIZADOR PARA ESTA CENA (têm prioridade):\n{notes.strip()}"
    data = analysis._call(MAESTRO_SYSTEM, msg, key, 6000, timeout=300)
    valid = {it["id"] for it in mine if it["type"] == "narracao"}
    first = next(iter(sorted(valid)), None)
    pieces = []
    for p in data.get("pieces", [])[:MAX_PIECES]:
        role = p.get("role") if p.get("role") in ROLES else "bed"
        prompt = re.sub(r"\s+", " ", str(p.get("prompt", ""))).strip()
        if len(prompt.split()) < 4:
            continue
        try:
            item = int(p.get("item"))
        except (TypeError, ValueError):
            item = first
        if item not in valid:
            item = first
        lo, hi = DUR_LIMITS[role]
        try:
            dur = float(p.get("duration", (lo + hi) / 2))
        except (TypeError, ValueError):
            dur = (lo + hi) / 2
        try:
            gain = max(0.1, min(0.8, float(p.get("gain", 0.4))))
        except (TypeError, ValueError):
            gain = 0.4
        pieces.append({"role": role, "label": str(p.get("label", "Música"))[:40], "description": str(p.get("description", ""))[:300],
                       "prompt": prompt[:500], "item": item, "duration": round(max(lo, min(hi, dur)), 1), "gain": round(gain, 2)})
    if not pieces:
        raise RuntimeError("O Claude leu a cena mas não propôs músicas. Escreve uma nota a pedir o que queres, ou acrescenta as peças à mão.")
    pieces.sort(key=lambda x: (x["item"] or 0))
    return {"direction": str(data.get("direction", ""))[:400], "pieces": pieces}


def generate_piece(piece: dict, key: str, dst: Path) -> None:
    """Gera a música na ElevenLabs (instrumental, modelo de música mais recente com recuo para o por defeito)."""
    sd.generate({"kind": "music", "prompt": piece["prompt"], "dur": float(piece.get("duration", 30))}, key, dst)


def separate_stems(src: Path, key: str, dst_zip: Path, variation: str = "six_stems_v1") -> None:
    """Separa uma música gerada em stems (a ElevenLabs devolve um ZIP com um ficheiro por stem)."""
    with open(src, "rb") as fh:
        r = httpx.post(STEMS_URL, headers={"xi-api-key": key}, timeout=900,
                       data={"stem_variation_id": variation}, files={"file": (src.name, fh, "audio/mpeg")})
    if r.status_code != 200:
        raise RuntimeError(f"A ElevenLabs devolveu {r.status_code}: {r.text[:300]}")
    dst_zip.write_bytes(r.content)
