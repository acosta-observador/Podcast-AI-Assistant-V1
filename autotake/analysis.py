"""Análise de Guião: o Claude lê o guião (narração e entrevistas) e prepara a folha de sonoplastia.

Fases:
1. read_items: o guião em itens classificados (narração, entrevista, título, nota).
2. segment: o Claude divide o episódio em cenas sonoras (lugar, época, direção sonora).
3. analyze_scene: o Claude analisa cada cena e propõe ambientes, efeitos e música ligados a pontos do texto.
A Estrutura Plus usa o resultado para pôr os sons na linha do tempo (cues_for_scene).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import httpx

from . import sounddesign as sd
from .script_parser import _is_direction
from .text import tokenize

ELLIPSIS = re.compile(r"^\s*(…|\.{3})\s*")
INTERVIEW_MARK = re.compile(r"\bSOM\s*\.?\s*\d+|\bEP\.?\s*\d|ARQUIVO|\(\s*\d{1,2}:\d{2}", re.I)
QUOTE_START = ("\"", "“", "«")
TYPES = {"narracao": "NARRAÇÃO", "entrevista": "ENTREVISTA", "titulo": "TÍTULO", "nota": "NOTA"}
MAX_ITEM_CHARS = 1500


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()).strip()


# ---------- 1. itens do guião ----------
def read_items(script_path: Path) -> list[dict]:
    """Guião → itens ordenados {id, type, text, speaker?}. Os títulos e notas orientam a gravação do narrador;
    a narração e as entrevistas são o que se ouve."""
    items: list[dict] = []
    expecting, speaker = False, ""

    def add(kind: str, text: str, **kw):
        items.append({"id": len(items) + 1, "type": kind, "text": text, **kw})

    for raw in sd._raw_items(script_path):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("http") or (s[0] in "[(" and s[-1] in "])"):
            if s[0] in "[(" and not s.lstrip("[(").startswith("http"):
                add("nota", s)
            continue
        if sd.SCENE_RE.match(s) and len(s) < 140:
            add("titulo", s)
            expecting = False
            continue
        if ELLIPSIS.match(s):
            body = ELLIPSIS.sub("", s).strip()
            if INTERVIEW_MARK.search(body):
                expecting, speaker = True, body
                add("nota", s, marker="entrevista")
            else:
                add("nota", s)
            continue
        letters = [c for c in s if c.isalpha()]
        words = [w for w in re.findall(r"[^\W\d_]+", s) if len(w) > 2][:2]
        if len(s) < 90 and ((letters and all(c.isupper() for c in letters)) or (len(words) == 2 and all(w.isupper() for w in words))):
            add("titulo", s)
            continue
        if not tokenize(s):
            continue
        if expecting:
            add("entrevista", s, speaker=speaker)
            expecting = False
        elif s.startswith(QUOTE_START) and items and items[-1]["type"] == "entrevista":
            add("entrevista", s, speaker=items[-1].get("speaker", ""))
        else:
            add("narracao", s)
    return items


def link_blocks(items: list[dict], blocks: list[dict]) -> None:
    """Liga cada parágrafo do guião ao bloco do AutoTake com o mesmo texto (a Estrutura Plus mede o tempo por blocos)."""
    pool: dict[str, list[int]] = {}
    for b in blocks:
        pool.setdefault(_norm(b["text"]), []).append(b["index"])
    for it in items:
        it.pop("block", None)
        if it["type"] in ("narracao", "entrevista") or (it["type"] == "nota" and not it.get("marker")):
            lst = pool.get(_norm(it["text"]))
            if lst:
                it["block"] = lst.pop(0)


def items_text(items: list[dict], limit: int | None = None) -> str:
    out = []
    for it in items:
        tag = TYPES[it["type"]] + (f" · {it['speaker']}" if it.get("speaker") else "")
        text = it["text"] if limit is None else it["text"][:limit]
        out.append(f"[{it['id']}] {tag}: {text}")
    return "\n".join(out)


# ---------- Claude ----------
def _call(system: str, user: str, key: str, max_tokens: int, timeout: int = 240) -> dict:
    r = httpx.post(sd.ANTHROPIC_URL, timeout=timeout, headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                   json={"model": sd.ANTHROPIC_MODEL, "max_tokens": max_tokens, "system": system,
                         "messages": [{"role": "user", "content": user}]})
    if r.status_code != 200:
        raise RuntimeError(f"A Anthropic devolveu {r.status_code}: {r.text[:300]}")
    text = "".join(p.get("text", "") for p in r.json().get("content", []))
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise RuntimeError("O Claude não devolveu JSON.")
    return json.loads(m.group(0))


SEGMENT_SYSTEM = """És sonoplasta de um podcast true crime (Podcast Plus, Observador). Vais dividir o guião de um episódio em
CENAS SONORAS para a folha de sonoplastia.

O guião tem: títulos e avisos de produção (DISCLAIMERS, TEASER INICIAL, GENÉRICO, CENA n...), narração lida pelo narrador,
excertos de entrevista ou arquivo (ENTREVISTA) e notas da equipa (NOTA). Os títulos e as notas servem sobretudo para
orientar a gravação do narrador: usa-os como pistas de estrutura, mas o que conta para o som é o texto lido (narração e
entrevistas).

Uma cena sonora é um troço contínuo com o mesmo lugar, época e clima. Exemplos de cenas de uma folha real:
"Disclaimers e abertura (estúdio, aviso inicial do podcast)", "Quem é Duarte Lima? (Lisboa e Rio de Janeiro, 2010)",
"Genérico (estúdio, genérico e apresentação da série)", "Da pobreza ao poder (Miranda do Douro e Lisboa, anos 70 e 80)",
"A casa cheia e a quinta de Sintra (Lisboa e Sintra, 1987-1994)", "A queda e o inquérito (Assembleia da República, 1994 a 1995)",
"No próximo episódio (Brasil, excertos do episódio 5)", "Créditos (estúdio)".
Podes subdividir as "CENA n" do guião quando o lugar, a época ou o clima mudam, e juntar o que for do mesmo ambiente.
Em regra saem entre 8 e 18 cenas num episódio de 40 a 60 minutos.

Responde APENAS com JSON:
{"scenes": [{"title": "título curto", "setting": "lugar e época", "direction": "direção sonora em 1 ou 2 frases: o arco e o que o som deve fazer",
 "from_item": n, "to_item": n}]}
As cenas cobrem todos os itens, por ordem, sem lacunas nem sobreposições."""

ANALYSIS_SYSTEM = """És o sonoplasta de um podcast true crime em português (Podcast Plus, Observador). Fazes a folha de sonoplastia de uma
cena, como um sonoplasta profissional: ambientes, efeitos sonoros e música, ligados a pontos exatos do texto lido.
Tom: sombrio, tenso, sóbrio e realista. O som serve a narração e nunca a compete.
Segue o GUIA DE ABORDAGEM no fim destas instruções: é o método do utilizador, com exemplos reais, e tem prioridade.

Os itens do guião têm etiquetas: NARRAÇÃO (lida pelo narrador), ENTREVISTA (excerto de entrevista ou arquivo: é o centro
dramático, não ponhas sons por cima, só os leias como contexto para o que vem antes e depois), TÍTULO e NOTA (indicações da equipa
para orientar a gravação; úteis como pistas, mas não são texto lido). Os teus sons entram nos itens de NARRAÇÃO ou, quando
se ligam a um título como "TEASER INICIAL", nesse título.

Faz o trabalho em passos e devolve-os.
PASSO 1, "reading": para cada item de NARRAÇÃO, lê-o literalmente e escreve {"item": n, "events": [coisas que ACONTECEM fisicamente
  e podem fazer som], "not_happening": [coisas negadas, hipotéticas, perguntadas, metafóricas ou só contadas de passagem, para
  as quais NÃO podes pôr som]}. "não bateu à porta" vai para not_happening.
PASSO 2, "cues": só podes criar sons a partir de "events", nunca de "not_happening".

Responde APENAS com JSON: {"reading": [ ... ], "cues": [ ... ]}
Cada elemento de "cues":
{"item": nº do item onde entra, "kind": "sfx" | "ambience" | "music",
 "label": nome curto em português (2 a 4 palavras),
 "description": descrição em português como na folha de sonoplastia (detalhe físico, época, lugar; andamento e instrumentos na
   música; o que o som faz na cena),
 "prompt": prompt CURTO em inglês para o gerador de áudio: 3 a 8 palavras, UM só som, simples e direto ("electric typewriter
   typing", "desk phone ringing twice"). NUNCA juntes dois sons num prompt: cria dois sons separados, cada um com o seu "anchor".
   Para music: estilo, instrumentos e andamento em poucas palavras; sem nomes de artistas,
 "anchor": 3 a 6 palavras EXATAS do texto do item onde o som entra (em ambience e music pode faltar),
 "duration": segundos (sfx 0.5 a 8, ambience 5 a 40),
 "until_item": nº do item onde acaba (só ambience e music),
 "gain": volume relativo de 0.1 a 1 (ambiente 0.5 a 0.8, efeitos 0.6 a 1, música em cama 0.3 a 0.5)}.
Cria mais sons, mas cada um simples: cada objeto ou ação que o texto nomeia merece o seu próprio som. A música é opcional e há
cenas sem ela. Se uma frase for ambígua, não ponhas o som."""


# ---------- 2. cenas ----------
def segment(items: list[dict], key: str) -> list[dict]:
    data = _call(SEGMENT_SYSTEM, items_text(items, limit=220), key, 6000)
    ids = [it["id"] for it in items]
    scenes, nxt = [], ids[0] if ids else 1
    for sc in data.get("scenes", []):
        try:
            a, b = int(sc["from_item"]), int(sc["to_item"])
        except (KeyError, TypeError, ValueError):
            continue
        a = max(a, nxt)
        if b < a or a > ids[-1]:
            continue
        scenes.append({"title": str(sc.get("title", ""))[:80], "setting": str(sc.get("setting", ""))[:120],
                       "direction": str(sc.get("direction", ""))[:400], "from": a, "to": min(b, ids[-1])})
        nxt = scenes[-1]["to"] + 1
    if not scenes:
        raise RuntimeError("O Claude não conseguiu dividir o guião em cenas.")
    scenes[0]["from"] = ids[0]
    scenes[-1]["to"] = ids[-1]
    for i in range(1, len(scenes)):          # sem lacunas
        scenes[i]["from"] = scenes[i - 1]["to"] + 1
    for i, sc in enumerate(scenes):
        sc["n"] = i + 1
    return scenes


# ---------- 3. análise de uma cena ----------
def analyze_scene(items: list[dict], scenes: list[dict], n: int, key: str, guide: str = "", notes: str = "") -> dict:
    sc = next(x for x in scenes if x["n"] == n)
    mine = [it for it in items if sc["from"] <= it["id"] <= sc["to"]]
    prev = scenes[n - 2] if n > 1 else None
    nxt = scenes[n] if n < len(scenes) else None
    msg = (f"Cena a trabalhar: {sc['title']} ({sc['setting']}).\nDireção sonora prevista: {sc['direction']}\n"
           f"Itens {sc['from']} a {sc['to']}.\n")
    if prev:
        msg += f"Cena anterior: {prev['title']} ({prev['setting']}).\n"
    if nxt:
        msg += f"Cena seguinte: {nxt['title']} ({nxt['setting']}).\n"
    msg += "\n" + items_text([{**it, "text": it["text"][:MAX_ITEM_CHARS]} for it in mine])
    if notes.strip():
        msg += f"\n\nNOTAS DO UTILIZADOR PARA ESTA CENA (têm prioridade):\n{notes.strip()}"
    system = ANALYSIS_SYSTEM + ("\n\n# GUIA DE ABORDAGEM\n\n" + guide if guide.strip() else "")
    data = _call(system, msg, key, 8000)
    by_id = {it["id"]: it for it in mine}
    reading = [{"item": int(x.get("item", 0)), "events": [str(e)[:200] for e in x.get("events", [])][:12],
                "not_happening": [str(e)[:200] for e in x.get("not_happening", [])][:12]}
               for x in data.get("reading", []) if isinstance(x, dict) and str(x.get("item", "")).isdigit()]
    return {"reading": reading, "cues": clean_cues(data.get("cues", []), by_id, sc)}


def clean_cues(raw: list, by_id: dict, sc: dict) -> list[dict]:
    out = []
    for c in raw[:sd.MAX_CUES]:
        try:
            item = int(c.get("item", 0))
        except (TypeError, ValueError):
            continue
        it, kind = by_id.get(item), c.get("kind")
        if not it or it["type"] == "entrevista" or kind not in ("sfx", "ambience", "music") or not c.get("prompt"):
            continue
        try:
            gain = max(0.05, min(1.0, float(c.get("gain", sd.DEFAULT_GAIN[kind]))))
        except (TypeError, ValueError):
            gain = sd.DEFAULT_GAIN[kind]
        if kind == "ambience":
            gain = max(gain, 0.55)
        try:
            dur = float(c.get("duration", 3 if kind == "sfx" else 20))
        except (TypeError, ValueError):
            dur = 3.0
        until = c.get("until_item")
        out.append({"id": f"a{len(out) + 1}", "item": item, "kind": kind, "label": str(c.get("label", "Som"))[:40],
                    "description": str(c.get("description", ""))[:500], "prompt": str(c["prompt"])[:300],
                    "anchor": str(c.get("anchor", ""))[:120], "duration": dur,
                    "until_item": int(until) if str(until).isdigit() and sc["from"] <= int(until) <= sc["to"] else None,
                    "gain": round(gain, 2)})
    return out


# ---------- ponte para a Estrutura Plus ----------
def scenes_for_structure(analysis: dict, blocks: list[dict]) -> list[dict]:
    """Cenas da análise em blocos do AutoTake (a Estrutura Plus trabalha por blocos)."""
    items = analysis.get("items") or []
    link_blocks(items, blocks)
    out = []
    for sc in analysis.get("scenes", []):
        bl = [it["block"] for it in items if sc["from"] <= it["id"] <= sc["to"] and it.get("block")]
        if bl:
            out.append({"n": sc["n"], "title": f"Cena {sc['n']}", "subtitle": sc["title"], "setting": sc.get("setting", ""),
                        "blocks": sorted(set(bl))})
    return out


def cues_for_scene(analysis: dict, n: int, tl: dict, structure_scene: dict) -> tuple[list[dict], int]:
    """Sons da análise de uma cena → sons da linha do tempo. Devolve (sons, nº de sons sem tempo, por cairem em entrevistas)."""
    items = analysis.get("items") or []
    sc = next((x for x in analysis.get("scenes", []) if x["n"] == n), None)
    if not sc or not sc.get("cues"):
        return [], 0
    mine = [it for it in items if sc["from"] <= it["id"] <= sc["to"]]

    timed = {b["block"] for b in tl["blocks"]}

    def block_of(item_id: int, forward: bool = True):
        seq = [it for it in mine if it["id"] >= item_id] if forward else [it for it in reversed(mine) if it["id"] <= item_id]
        return next((it["block"] for it in seq if it.get("block") in timed), None)

    raw, skipped = [], 0
    for c in sc["cues"]:
        blk = block_of(c["item"])
        if not blk:
            skipped += 1
            continue
        until = block_of(c["until_item"], forward=False) if c.get("until_item") else None
        raw.append({"block": blk, "kind": c["kind"], "label": c["label"], "description": c.get("description", ""),
                    "prompt": c["prompt"], "anchor": c.get("anchor", ""), "duration": c.get("duration", 3),
                    "until_block": until or blk, "gain": c.get("gain")})
    return sd.cues_from_claude(raw, tl, structure_scene), skipped
