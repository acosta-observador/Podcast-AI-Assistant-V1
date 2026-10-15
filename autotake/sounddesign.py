"""Estrutura Plus: planeia e gera sound design (efeitos e música) a partir do guião.

As regras locais procuram no guião, já alinhado com os takes, as palavras que pedem som
(porta, tiro, comboio, mar...) e propõem uma cama de música. A ElevenLabs gera cada áudio.
Tudo fica numa linha do tempo medida na montagem final, não na gravação original.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import httpx
import numpy as np

from .pipeline import build_cuts
from .script_parser import _is_direction
from .text import tokenize

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
SFX_URL = "https://api.elevenlabs.io/v1/sound-generation"
MUSIC_URL = "https://api.elevenlabs.io/v1/music"
N_SFX_LANES = 5
AMB_LANE = N_SFX_LANES
MUSIC_LANE = N_SFX_LANES + 1
LANE_NAMES = tuple(f"Efeitos {i + 1}" for i in range(N_SFX_LANES)) + ("Ambiente", "Música")
GUIDE_FILE = Path(__file__).with_name("guia_sound_design.md")
MAX_CUES = 120
# pausas na linha do tempo da Estrutura Plus (a montagem do AutoTake continua com as suas)
BLOCK_GAP = 1.5
SCENE_GAP = 4.0
SCENE_RE = re.compile(r"^[\[(#\s]*(cena|scene)\s*\d+", re.I)
FALLBACK_SCENE_BLOCKS = 4

# ---------- cenas ----------
def _raw_items(path: Path) -> list[str]:
    if path.suffix.lower() == ".docx":
        import docx
        return [p.text for p in docx.Document(str(path)).paragraphs]
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if re.search(r"\n\s*\n", text):
        return [" ".join(c.split("\n")) for c in re.split(r"\n\s*\n", text)]
    return text.split("\n")


def scenes_for(script_path: Path, n_blocks: int) -> list[dict]:
    """Cenas do episódio: linhas 'CENA 1', 'CENA 2'... no guião; sem elas, grupos de 4 blocos."""
    starts, count = [], 0
    try:
        for item in _raw_items(script_path):
            s = item.strip()
            if SCENE_RE.match(s) and len(s) < 60:
                starts.append(count + 1)
            elif not _is_direction(s) and tokenize(s):
                count += 1
    except Exception:
        starts = []
    starts = sorted({x for x in starts if 1 <= x <= n_blocks})
    if not starts:
        starts = list(range(1, n_blocks + 1, FALLBACK_SCENE_BLOCKS))
    elif starts[0] != 1:
        starts.insert(0, 1)
    out = []
    for i, st in enumerate(starts):
        end = (starts[i + 1] - 1) if i + 1 < len(starts) else n_blocks
        out.append({"n": i + 1, "title": f"Cena {i + 1}", "blocks": list(range(st, end + 1))})
    return out


# ---------- linha do tempo da montagem ----------
def timeline(result: dict, selections: dict, scenes: list[dict] | None = None) -> dict:
    """Posição de cada bloco e de cada corte na linha do tempo (segundos).

    Entre blocos há uma pausa maior do que na montagem do AutoTake, e maior ainda entre cenas.
    """
    cuts = build_cuts(result, selections)
    texts = {b["index"]: b["text"] for b in result["blocks"]}
    scene_first = {sc["blocks"][0] for sc in (scenes or [])}
    pos, spans, segs, gaps = 0.0, {}, [], {}
    for k, c in enumerate(cuts):
        length = c["end"] - c["start"]
        sp = spans.setdefault(c["block"], [pos, pos])
        segs.append({"start": round(pos, 3), "end": round(pos + length, 3),
                     "src": round(c["start"], 3), "block": c["block"]})
        pos += length
        sp[1] = pos
        last = k == len(cuts) - 1 or cuts[k + 1]["block"] != c["block"]
        if last:
            nxt = cuts[k + 1]["block"] if k + 1 < len(cuts) else None
            gap = 0.0 if nxt is None else (SCENE_GAP if nxt in scene_first else BLOCK_GAP)
            gaps[c["block"]] = gap
            pos += gap
        else:
            pos += c["gap_after"]
    blocks = [{"block": i, "start": round(s, 3), "end": round(e, 3), "text": texts[i]}
              for i, (s, e) in sorted(spans.items())]
    by_block = {b["block"]: b for b in blocks}
    sc_out = []
    for sc in scenes or []:
        bs = [by_block[i] for i in sc["blocks"] if i in by_block]
        if bs:
            sc_out.append({"n": sc["n"], "title": sc["title"], "blocks": [b["block"] for b in bs],
                           "start": bs[0]["start"], "end": bs[-1]["end"]})
    return {"blocks": blocks, "segments": segs, "total": round(pos, 3), "gaps": gaps, "scenes": sc_out}


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


# ---------- planeamento local (sem LLM): palavras-chave do guião ----------
# (padrão sobre o texto sem acentos e em minúsculas, nome, descrição para a ElevenLabs, duração em s)
RULES = [
    (r"\bporta(s)?\b", "Porta a abrir", "A heavy wooden door creaking open and then closing", 3),
    (r"\b(tiros?|disparos?|disparou|balas?)\b", "Tiro", "A single gunshot echoing in the distance", 2),
    (r"\b(comboios?|carruagens?|locomotiva)\b", "Comboio a passar", "A steam train passing by on the tracks with rhythmic clatter", 6),
    (r"\b(passos|caminhou|andou)\b", "Passos", "Slow footsteps on gravel", 3),
    (r"\b(sinos?|igrejas?)\b", "Sino de igreja", "A church bell tolling slowly", 5),
    (r"\b(mar|ondas?|maremoto|oceano|tsunami)\b", "Ondas do mar", "Large ocean waves crashing and receding", 6),
    (r"\b(fogo|incendios?|chamas|ardeu|arder|ardia)\b", "Fogo a crepitar", "A large fire crackling and roaring", 6),
    (r"\b(terramoto|sismo|abalo|tremer|tremor)\b", "Terra a tremer", "Deep low rumbling of an earthquake with debris falling", 6),
    (r"\b(multidao|gritos?|gritaram|povo|populacao)\b", "Multidão", "A frightened crowd shouting and murmuring in the distance", 5),
    (r"\b(chuva|trovoada|trovao|tempestade)\b", "Chuva e trovoada", "Heavy rain with distant thunder", 6),
    (r"\b(telefone|telefonou|chamada)\b", "Telefone", "An old telephone ringing", 3),
    (r"\b(carros?|automovel|viatura|viaturas)\b", "Carro a passar", "A car passing by on a road", 4),
    (r"\b(sirenes?|policia|ambulancia|ambulancias)\b", "Sirene", "A distant police siren", 5),
    (r"\bvento\b", "Vento", "Wind howling", 5),
    (r"\b(relogio|relogios)\b", "Relógio", "A ticking clock in a quiet room", 4),
    (r"\b(aviao|avioes|helicoptero)\b", "Avião", "An airplane passing overhead", 5),
    (r"\b(cao|caes|ladrar|ladrou)\b", "Cão a ladrar", "A dog barking in the distance", 3),
    (r"\b(explosao|explodiu|bomba|bombas)\b", "Explosão", "A loud explosion followed by a rumbling echo", 4),
    (r"\b(janela|vidros?)\b", "Vidro a partir", "Glass breaking", 2),
    (r"\b(sussurro|sussurrou|segredo)\b", "Sussurros", "Quiet indistinct whispering in a room", 4),
]
CLAUDE_SYSTEM = """És o sonoplasta de um podcast true crime em português (Podcast Plus, Observador). Fazes a folha de sonoplastia
de cada cena, como um sonoplasta profissional: ambientes, efeitos sonoros e música, ligados a pontos exatos do texto.
Tom: sombrio, tenso, sóbrio e realista. O som serve a narração e nunca a compete.
Trabalhas UMA cena de cada vez: os blocos marcados são a cena; os outros só dão contexto. Segue o GUIA DE ABORDAGEM no fim
destas instruções: é o método do utilizador, com exemplos reais, e tem prioridade. Calibra a densidade e o estilo por ele.

Faz o trabalho em passos, e devolve-os.

PASSO 1, "direction": a direção sonora da cena em 1 ou 2 frases (o arco e o que o som deve fazer).
PASSO 2, "reading": lê cada bloco da cena literalmente, frase a frase, e escreve para cada bloco
  {"block": nº, "events": [coisas que ACONTECEM fisicamente e podem fazer som], "not_happening": [coisas negadas,
   hipotéticas, perguntadas, metafóricas ou só contadas de passagem, para as quais NÃO podes pôr som]}.
  Atenção às negações: "não bateu à porta" vai para not_happening, nunca para events.
PASSO 3, "cues": só podes criar sons a partir de "events". Nunca a partir de "not_happening". Blocos sem take do narrador
  são excertos de entrevista ou arquivo: não ponhas sons por cima deles.

Responde APENAS com JSON:
{"direction": "...", "reading": [ ... ], "cues": [ ... ]}

Cada elemento de "cues":
{"block": nº do bloco onde começa, "kind": "sfx" | "ambience" | "music",
 "label": nome curto em português (2 a 4 palavras),
 "description": descrição em português como na folha de sonoplastia (detalhe físico, época, lugar, andamento e instrumentos
   na música; diz também o que o som faz na cena),
 "prompt": descrição EM INGLÊS para o gerador de áudio (concreta, física, com distância, espaço e época; para music: estilo,
   instrumentos, andamento, atmosfera; sem nomes de artistas),
 "anchor": 3 a 6 palavras EXATAS do texto do bloco onde o som entra ("Entra em"; em ambience e music pode faltar: entra no
   início do bloco),
 "duration": segundos (sfx 0.5 a 8, ambience 5 a 40),
 "until_block": nº do bloco onde acaba (só ambience e music),
 "gain": volume relativo de 0.1 a 1}.
A música é opcional e há cenas sem ela. Se uma frase for ambígua, não ponhas o som."""
MUSIC_SUFFIX = ", instrumental, no vocals, very subtle dark background underscore with low presence, leaves room for a spoken voice"
AMBIENCE_SUFFIX = ", seamless ambient bed, no music, no speech"
MUSIC_PROMPT = ("Subtle atmospheric instrumental underscore for a narrative documentary podcast, "
                "low intensity, slow, understated, no vocals, leaves room for a spoken voice")
MUSIC_PART = 120.0
MAX_PER_BLOCK = 3
MIN_REPEAT_GAP = 12.0


def _norm(t: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()


def plan_local(tl: dict, span: tuple | None = None) -> list[dict]:
    """Escolhe os sons sem LLM: procura palavras-chave no texto de cada bloco e põe uma cama de música."""
    cues, last = [], {}
    for b in tl["blocks"]:
        text, blen = _norm(b["text"]), max(0.1, b["end"] - b["start"])
        hits = []
        for k, (pat, label, prompt, dur) in enumerate(RULES):
            m = re.search(pat, text)
            if m:
                hits.append((m.start(), k, label, prompt, dur))
        for pos, k, label, prompt, dur in sorted(hits)[:MAX_PER_BLOCK]:
            start = b["start"] + blen * pos / max(1, len(text))
            start = max(b["start"], min(b["end"] - 0.5, start - 0.3))
            if start - last.get(k, -1e9) < MIN_REPEAT_GAP:
                continue
            last[k] = start
            cues.append({"kind": "sfx", "label": label, "prompt": prompt, "block": b["block"],
                         "start": round(start, 2), "dur": float(dur)})
    t, first = (span[0] if span else 0.0), tl["blocks"][0]["block"] if tl["blocks"] else 1
    total = span[1] if span else tl["total"]
    while t < total - 3:
        part = min(MUSIC_PART, total - t)
        blk = next((b["block"] for b in tl["blocks"] if b["end"] > t), first)
        cues.append({"kind": "music", "label": "Música de fundo", "prompt": MUSIC_PROMPT, "block": blk,
                     "start": round(t, 2), "dur": round(max(10.0, part), 2)})
        t += part
    return cues


def load_guide(custom_path: Path | None = None) -> str:
    """Guia de abordagem: o do utilizador (editado na app) ou o que vem com o código."""
    if custom_path and custom_path.exists():
        return custom_path.read_text(encoding="utf-8")
    return GUIDE_FILE.read_text(encoding="utf-8")


def plan_claude(tl: dict, scene: dict, key: str, guide: str = "", notes: str = "",
                all_blocks: list | None = None) -> dict:
    """O Claude lê a cena (com o resto do episódio como contexto) e decide sons, ambiente e música."""
    ids = set(scene["blocks"])
    timed = {b["block"]: b for b in tl["blocks"]}
    order = sorted(set(timed) | {x["index"] for x in (all_blocks or [])})
    texts = {x["index"]: x["text"] for x in (all_blocks or [])}
    first, last = scene["blocks"][0], scene["blocks"][-1]
    parts = []
    for i in order:
        b = timed.get(i)
        if b and i in ids:
            parts.append(f"[BLOCO {i} | {b['start']:.1f}s a {b['end']:.1f}s | {b['end'] - b['start']:.1f}s]\n{b['text']}")
        elif b:
            parts.append(f"(bloco {i}, fora desta cena: {b['text'][:160]}...)")
        elif first <= i <= last:
            parts.append(f"[BLOCO {i} | sem take do narrador: provável excerto de entrevista ou arquivo, sem sons por cima]\n{texts.get(i, '')[:400]}")
        else:
            parts.append(f"(bloco {i}, fora desta cena, sem take do narrador: {texts.get(i, '')[:100]}...)")
    msg = (f"Cena a trabalhar: {scene['title']}, blocos {scene['blocks'][0]} a {scene['blocks'][-1]}, "
           f"de {scene['start']:.1f}s a {scene['end']:.1f}s.\n\n" + "\n\n".join(parts))
    if notes.strip():
        msg += f"\n\nNOTAS DO UTILIZADOR PARA ESTA CENA (têm prioridade):\n{notes.strip()}"
    system = CLAUDE_SYSTEM + ("\n\n# GUIA DE ABORDAGEM\n\n" + guide if guide.strip() else "")
    body = {"model": ANTHROPIC_MODEL, "max_tokens": 8000, "system": system,
            "messages": [{"role": "user", "content": msg}]}
    r = httpx.post(ANTHROPIC_URL, json=body, timeout=240,
                   headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
    if r.status_code != 200:
        raise RuntimeError(f"A Anthropic devolveu {r.status_code}: {r.text[:300]}")
    text = "".join(p.get("text", "") for p in r.json().get("content", []))
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise RuntimeError("O Claude não devolveu JSON.")
    data = json.loads(m.group(0))
    reading = [{"block": int(x.get("block", 0)), "events": [str(e)[:200] for e in x.get("events", [])][:12],
                "not_happening": [str(e)[:200] for e in x.get("not_happening", [])][:12]}
               for x in data.get("reading", []) if isinstance(x, dict)]
    return {"mood": str(data.get("direction", data.get("mood", "")))[:400], "reading": reading,
            "cues": cues_from_claude(data.get("cues", []), tl, scene)}


def cues_from_claude(raw: list, tl: dict, scene: dict) -> list[dict]:
    by = {b["block"]: b for b in tl["blocks"]}
    ids = set(scene["blocks"])
    cues = []

    def add(kind, label, prompt, block, start, dur, gain, desc="", anchor=""):
        cues.append({"kind": kind, "label": label, "prompt": prompt, "block": block,
                     "start": round(start, 2), "dur": round(dur, 2), "gain": round(gain, 2),
                     "desc": desc, "anchor": anchor})

    for c in raw[:MAX_CUES]:
        try:
            blk = int(c.get("block", 0))
        except (TypeError, ValueError):
            continue
        b = by.get(blk)
        kind = c.get("kind")
        if blk not in ids or not b or kind not in ("sfx", "ambience", "music") or not c.get("prompt"):
            continue
        label, prompt = str(c.get("label", "Som"))[:40], str(c["prompt"])[:400]
        desc, anchor_raw = str(c.get("description", ""))[:500], str(c.get("anchor", ""))[:120]
        try:
            gain = max(0.05, min(1.0, float(c.get("gain", 0.8 if kind == "sfx" else 0.35))))
        except (TypeError, ValueError):
            gain = 0.8 if kind == "sfx" else 0.35
        start, anchor = b["start"], _norm(str(c.get("anchor", "")).strip())
        if anchor:
            text = _norm(b["text"])
            i = text.find(anchor)
            if i >= 0:
                start = b["start"] + (b["end"] - b["start"]) * i / max(1, len(text))
                start = max(b["start"], start - 0.3)
        start = min(start, b["end"] - 0.5)
        if kind == "sfx":
            try:
                dur = max(0.5, min(10.0, float(c.get("duration", 3))))
            except (TypeError, ValueError):
                dur = 3.0
            add("sfx", label, prompt, blk, start, dur, gain, desc, anchor_raw)
            continue
        end_b = by.get(int(c.get("until_block") or blk), b) if str(c.get("until_block") or blk).isdigit() else b
        end = min(scene["end"], max(end_b["end"], start + 5))
        chunk = 120.0 if kind == "music" else 20.0
        suffix = MUSIC_SUFFIX if kind == "music" else AMBIENCE_SUFFIX
        prompt = prompt.rstrip(". ") + suffix
        t, k = start, 1
        while t < end - 2:
            d = min(chunk, end - t)
            add(kind, label if k == 1 else f"{label} ({k})", prompt, blk, t, max(5.0, d), gain, desc, anchor_raw)
            t += d
            k += 1
    return cues


def assign_lanes(cues: list[dict]) -> None:
    busy: dict[int, list[tuple[float, float]]] = {k: [] for k in range(N_SFX_LANES + 2)}
    for c in sorted(cues, key=lambda x: x["start"]):
        if c["kind"] == "music":
            c["lane"] = MUSIC_LANE
        elif c["kind"] == "ambience":
            c["lane"] = AMB_LANE
        else:
            c["lane"] = next((k for k in range(N_SFX_LANES)
                              if all(c["start"] >= e or c["start"] + c["dur"] <= s0 for s0, e in busy[k])),
                             N_SFX_LANES - 1)
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
