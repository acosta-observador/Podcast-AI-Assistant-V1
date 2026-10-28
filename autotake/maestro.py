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

# Cada peça recebe um "kit" diferente (instrumento principal, ritmo, textura, andamento, harmonia), escolhido por código, para que
# as peças de uma cena e as de cenas vizinhas não soem todas ao mesmo piano ou à mesma cama sombria.
KITS = [
    {"lead": "bowed vibraphone", "rhythm": "dry rim clicks", "texture": "tape-warped synth pad", "bpm": 68, "harmony": "suspended chords"},
    {"lead": "detuned music box", "rhythm": "soft ticking clockwork", "texture": "vinyl crackle and room tone", "bpm": 72, "harmony": "minor with a raised sixth"},
    {"lead": "muted upright bass played pizzicato", "rhythm": "brushed snare", "texture": "distant tremolo electric guitar", "bpm": 84, "harmony": "minor seventh chords, noir jazz"},
    {"lead": "analog modular synth arpeggio", "rhythm": "filtered hi-hat loop", "texture": "warm sub pad", "bpm": 96, "harmony": "minor with a tritone"},
    {"lead": "lap steel guitar with long reverb", "rhythm": "finger snaps and a soft shaker", "texture": "airy choir pad", "bpm": 66, "harmony": "open fifths"},
    {"lead": "glass harmonica", "rhythm": "soft felt kick on beats one and three", "texture": "granular noise bed", "bpm": 62, "harmony": "whole-tone colour"},
    {"lead": "cello playing harmonics and slow bowing", "rhythm": "typewriter-like clicks", "texture": "cold digital pad", "bpm": 60, "harmony": "unresolved minor"},
    {"lead": "Rhodes electric piano with tremolo", "rhythm": "muted bongos", "texture": "wide muted strings", "bpm": 78, "harmony": "minor ninth chords"},
    {"lead": "baritone saxophone, breathy and distant", "rhythm": "brushed hi-hat", "texture": "low string swells", "bpm": 58, "harmony": "chromatic descending bass"},
    {"lead": "kalimba and marimba", "rhythm": "dry wood blocks", "texture": "warm synth pad", "bpm": 88, "harmony": "minor pentatonic"},
    {"lead": "tremolo electric guitar with spring reverb", "rhythm": "soft rim shots", "texture": "hollow organ swells", "bpm": 70, "harmony": "minor with a flat fifth"},
    {"lead": "toy piano and celesta", "rhythm": "tape-loop clicks", "texture": "dark ambient wash", "bpm": 76, "harmony": "diminished colour"},
    {"lead": "harmonium", "rhythm": "slow tambourine taps", "texture": "sparse high strings", "bpm": 64, "harmony": "modal, drifting"},
    {"lead": "quiet distorted bass synth line", "rhythm": "glitchy micro percussion", "texture": "cold analog pad", "bpm": 100, "harmony": "minor with a chromatic walk"},
    {"lead": "low clarinet", "rhythm": "soft timpani rolls", "texture": "pizzicato strings", "bpm": 74, "harmony": "dorian"},
    {"lead": "Mellotron flutes", "rhythm": "brushed snare and upright bass", "texture": "analog tape hiss", "bpm": 82, "harmony": "minor with suspended fourths"},
    {"lead": "waterphone and bowed metal textures", "rhythm": "soft ticking", "texture": "sub bass", "bpm": 62, "harmony": "dissonant clusters, restrained"},
    {"lead": "Wurlitzer electric piano with chorus", "rhythm": "drum machine rimshot", "texture": "warm analog strings", "bpm": 90, "harmony": "minor seventh with an added ninth"},
    {"lead": "fingerpicked nylon-string guitar", "rhythm": "soft shaker", "texture": "low cello", "bpm": 72, "harmony": "phrygian colour"},
    {"lead": "wordless choir pads", "rhythm": "muted clock ticks", "texture": "sparse low piano notes", "bpm": 56, "harmony": "slow minor progression"},
    {"lead": "electric guitar with e-bow sustain", "rhythm": "brushed floor tom", "texture": "reverse cymbal shimmer", "bpm": 68, "harmony": "suspended minor"},
    {"lead": "slow distant accordion", "rhythm": "soft metronome click", "texture": "low synth bass", "bpm": 80, "harmony": "minor with a raised seventh"},
    {"lead": "muted french horn", "rhythm": "tapped wooden percussion", "texture": "pizzicato strings", "bpm": 66, "harmony": "open fourths"},
    {"lead": "muted clavinet stabs", "rhythm": "dry kick and ride", "texture": "deep analog bass", "bpm": 104, "harmony": "minor funk-noir vamp"},
    {"lead": "harp arpeggios", "rhythm": "fingers tapping on a table", "texture": "dark synth swell", "bpm": 70, "harmony": "minor with a lowered second"},
    {"lead": "thin lonely oboe", "rhythm": "slow muffled floor tom", "texture": "airy synth pad", "bpm": 60, "harmony": "unresolved suspended"},
    {"lead": "bass clarinet with muted trumpet", "rhythm": "brush swirl", "texture": "ambient guitar loop", "bpm": 76, "harmony": "minor, smoky"},
    {"lead": "hang drum", "rhythm": "soft rain-stick shakes", "texture": "airy analog pad", "bpm": 86, "harmony": "minor, floating"},
    {"lead": "string quartet playing sul ponticello", "rhythm": "light pizzicato ticking", "texture": "sub bass", "bpm": 92, "harmony": "chromatic, tense"},
]


def kits_for_scene(n: int, title: str, count: int = 7) -> list[dict]:
    """Kits diferentes para cada peça da cena, e diferentes entre cenas (passo coprimo com o nº de kits)."""
    import zlib
    base = (zlib.crc32(title.encode()) + n * 7) % len(KITS)
    return [KITS[(base + k * 11) % len(KITS)] for k in range(count)]


def _words(p: str) -> set:
    stop = {"a", "an", "the", "of", "with", "and", "for", "in", "on", "to", "no", "only", "soft", "slow", "dark", "tense", "mysterious", "restrained", "other", "instruments"}
    return {w for w in re.findall(r"[a-z]+", p.lower()) if w not in stop and len(w) > 2}


def too_similar(pieces: list[dict]) -> list[tuple[int, int]]:
    bad = []
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            a, b = _words(pieces[i]["prompt"]), _words(pieces[j]["prompt"])
            if a and b and len(a & b) / len(a | b) > 0.45:
                bad.append((i, j))
    return bad


MAESTRO_SYSTEM = """És o compositor e supervisor musical de um podcast true crime em português (Podcast Plus, Observador). Recebes o texto de UMA
cena e decides as músicas e os stems que a ElevenLabs vai gerar para ela. Cada peça é gerada por uma IA de música a partir de um prompt em
inglês, por isso o prompt é tudo.

ESTILO: true crime, sombrio e tenso, mas contido: NÃO épico, NÃO orquestral de trailer. A paleta é LARGA e as peças têm de soar a coisas
diferentes: sintetizadores analógicos e modulares, guitarras (tremolo, lap steel, e-bow, nylon), contrabaixo em pizzicato, saxofone barítono,
clarinete, trompa com surdina, oboé, harmónio, acordeão, harpa, vibrafone com arco, marimba, kalimba, caixa de música, celesta, Rhodes e Wurlitzer,
cordas em harmónicos ou pizzicato, vozes sem palavras, percussão de todos os tipos (escovas, rim clicks, tique-taque, bongos, tímpanos abafados,
shakers, glitches) e texturas (vinil, fita, ruído granular). O PIANO É RARO: no máximo UMA peça por cena e só se o kit atribuído o trouxer;
nunca "felt piano" por defeito. A música serve a narração: fica por baixo da voz.

KITS: a mensagem traz, para cada peça, um KIT atribuído (instrumento principal, ritmo, textura, andamento e harmonia). Usa os kits POR ORDEM,
um por peça, e não os repitas. Cada peça tem de ter o instrumento principal do seu kit bem presente no prompt (num stem, só um elemento do kit:
o ritmo, a textura ou o instrumento principal, sozinho). Adapta as palavras ao clima da cena, mas respeita o instrumento, o andamento aproximado
(BPM) e a cor harmónica. Duas peças da mesma cena nunca partilham o instrumento principal nem andamentos a menos de 8 BPM de diferença.

QUANTAS: de 3 a 5 peças numa cena normal; 2 a 3 numa cena curta (menos de 40 s de narração); até 7 numa cena longa (mais de 3 minutos).
CADA PEÇA É DIFERENTE DAS OUTRAS: instrumentação, textura, andamento e carácter distintos (os kits garantem isso). É PROIBIDO repetir a mesma
ideia com palavras diferentes e é PROIBIDO "drone sombrio" ou "pulso grave" como ideia principal: tem de haver instrumentos e movimento concretos. Acompanha o
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


def _parse_pieces(data: dict, valid: set, first) -> list[dict]:
    """Peças devolvidas pelo Claude → peças limpas (papel, duração, parágrafo de entrada válidos)."""
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
    return pieces


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
    kits = kits_for_scene(n, sc["title"])
    msg += "\n\nKITS ATRIBUÍDOS (por ordem, um por peça; não os repitas):\n" + "\n".join(
        f"{i + 1}. instrumento principal: {k['lead']}; ritmo: {k['rhythm']}; textura: {k['texture']}; ~{k['bpm']} BPM; harmonia: {k['harmony']}"
        for i, k in enumerate(kits))
    data = analysis._call(MAESTRO_SYSTEM, msg, key, 6000, timeout=300)
    valid = {it["id"] for it in mine if it["type"] == "narracao"}
    first = next(iter(sorted(valid)), None)
    pieces = _parse_pieces(data, valid, first)
    if not pieces:
        raise RuntimeError("O Claude leu a cena mas não propôs músicas. Escreve uma nota a pedir o que queres, ou acrescenta as peças à mão.")
    for _ in range(1):                                  # uma segunda tentativa se as peças saírem parecidas ou com piano a mais
        bad, pianos = too_similar(pieces), [i for i, p in enumerate(pieces) if "piano" in p["prompt"].lower()]
        if not bad and len(pianos) <= 1:
            break
        why = ("as peças " + ", ".join(f"{i + 1} e {j + 1}" for i, j in bad) + " estão demasiado parecidas. " if bad else "") + \
              ("há piano em mais de uma peça. " if len(pianos) > 1 else "")
        data2 = analysis._call(MAESTRO_SYSTEM, msg + f"\n\nCORREÇÃO: na tua resposta anterior {why}Refaz o JSON completo: cada peça com o instrumento principal do "
                               "seu kit, andamentos diferentes, sem piano exceto se o kit o trouxer e no máximo numa peça.", key, 6000, timeout=300)
        retry = _parse_pieces(data2, valid, first)
        if retry:
            pieces, data = retry, data2
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
