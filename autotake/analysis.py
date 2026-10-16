"""Análise de Guião: as cenas do guião, os parágrafos de cada cena e, ao lado, os sons a ir buscar à ElevenLabs.

Os sons são texto livre, uma linha por som (escrito à mão ou sugerido pelo Claude). A Estrutura Plus lê estas linhas,
pede os áudios à ElevenLabs e põe-nos na linha do tempo.
"""
from __future__ import annotations

import hashlib
import json
import re

import httpx

from . import sounddesign as sd
from .script_parser import SCENE_RE, classify_items

read_items = classify_items
VISIBLE = ("narracao", "entrevista")      # o resto (títulos, avisos, notas da equipa) não interessa para o som
MUSIC_PREFIX = re.compile(r"^(m[úu]sica|music)\s*:\s*", re.I)


# ---------- itens e cenas ----------
def link_blocks(items: list[dict], blocks: list[dict]) -> None:
    """Liga cada parágrafo de narração ao bloco do AutoTake com o mesmo texto (a Estrutura Plus mede o tempo por blocos)."""
    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", t).strip().lower()
    pool: dict[str, list[int]] = {}
    for b in blocks:
        pool.setdefault(norm(b["text"]), []).append(b["index"])
    for it in items:
        it.pop("block", None)
        if it["type"] == "narracao":
            lst = pool.get(norm(it["text"]))
            if lst:
                it["block"] = lst.pop(0)


def local_scenes(items: list[dict]) -> list[dict]:
    """As cenas são as 'CENA n - título' do guião. O que vem antes da primeira (avisos, teaser) junta-se à primeira cena
    e o que vem depois da última (próximo episódio, créditos) à última."""
    if not items:
        return []
    heads = [it for it in items if it["type"] == "titulo" and SCENE_RE.match(it["text"])]
    if not heads:
        return [{"n": 1, "title": "Episódio", "from": items[0]["id"], "to": items[-1]["id"]}]
    out = []
    for i, h in enumerate(heads):
        title = re.sub(r"^[\[(#\s]*(cena|scene)\s*\d+\s*[-–—:.]?\s*", "", h["text"], flags=re.I).strip()
        out.append({"n": i + 1, "title": title or f"Cena {i + 1}", "from": h["id"],
                    "to": (heads[i + 1]["id"] - 1) if i + 1 < len(heads) else items[-1]["id"]})
    out[0]["from"] = items[0]["id"]
    return out


def scenes_for_structure(an: dict, blocks: list[dict]) -> list[dict]:
    """Cenas da análise em blocos do AutoTake (a Estrutura Plus trabalha por blocos)."""
    items = an.get("items") or []
    link_blocks(items, blocks)
    out = []
    for sc in an.get("scenes", []):
        bl = [it["block"] for it in items if sc["from"] <= it["id"] <= sc["to"] and it.get("block")]
        if bl:
            out.append({"n": sc["n"], "title": f"Cena {sc['n']}", "subtitle": sc["title"], "setting": "", "blocks": sorted(set(bl))})
    return out


# ---------- linhas de som ----------
def parse_lines(text: str) -> list[dict]:
    """Uma linha por som. 'porta pesada a fechar @ A sede da Polícia' fixa o ponto do texto onde entra.
    'música: piano lento' pede música em vez de um efeito."""
    out = []
    for raw in (text or "").splitlines():
        s = raw.strip().lstrip("-•*·").strip()
        if not s:
            continue
        kind = "sfx"
        if MUSIC_PREFIX.match(s):
            kind, s = "music", MUSIC_PREFIX.sub("", s).strip()
        body, _, anchor = s.partition("@")
        if body.strip():
            out.append({"text": body.strip()[:300], "anchor": anchor.strip()[:120], "kind": kind})
    return out


def scene_lines(an: dict, sc: dict) -> dict[int, list[dict]]:
    items = {it["id"]: it for it in an.get("items", [])}
    out = {}
    for iid in range(sc["from"], sc["to"] + 1):
        it = items.get(iid)
        if it and it["type"] == "narracao":
            ls = parse_lines((an.get("lines") or {}).get(str(iid), ""))
            if ls:
                out[iid] = ls
    return out


def count_lines(an: dict, sc: dict) -> int:
    return sum(len(v) for v in scene_lines(an, sc).values())


def lines_sig(an: dict, sc: dict) -> str:
    keep = [(iid, [(l["text"], l["anchor"], l["kind"]) for l in ls]) for iid, ls in scene_lines(an, sc).items()]
    return hashlib.md5(json.dumps(keep, ensure_ascii=False).encode()).hexdigest()[:12]


# ---------- Claude ----------
def _call(system: str, user: str, key: str, max_tokens: int, timeout: int = 300, model: str | None = None) -> dict:
    r = httpx.post(sd.ANTHROPIC_URL, timeout=timeout, headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                   json={"model": model or sd.ANTHROPIC_MODEL, "max_tokens": max_tokens, "system": system,
                         "messages": [{"role": "user", "content": user}]})
    if r.status_code != 200:
        raise RuntimeError(f"A Anthropic devolveu {r.status_code}: {r.text[:300]}")
    body = r.json()
    stop = body.get("stop_reason")
    text = "".join(p.get("text", "") for p in body.get("content", []))
    if stop == "max_tokens":
        raise RuntimeError("A resposta do Claude foi cortada por ser demasiado longa. Tenta outra vez ou divide a cena.")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise RuntimeError(f"O Claude não devolveu JSON (fim: {stop}). Início da resposta: {text[:160]!r}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"O Claude devolveu JSON inválido ({e}). Início da resposta: {text[:160]!r}")


SUGGEST_SYSTEM = """És o sonoplasta de um podcast true crime em português (Podcast Plus, Observador). Recebes o texto de UMA cena, parágrafo
a parágrafo, e sugeres os sons a ir buscar a um gerador de efeitos sonoros (ElevenLabs) para os pôr na linha do tempo.
Tom: sombrio, tenso, sóbrio e realista. O som serve a narração e nunca a compete.

Cada parágrafo tem uma etiqueta: NARRAÇÃO (lida pelo narrador) ou ENTREVISTA (excerto de entrevista ou arquivo: não leva sons).
Só sugeres sons para parágrafos de NARRAÇÃO.

Escreve cada som numa linha curta, em português, com 3 a 8 palavras e UM só som. Exemplos: "porta pesada a fechar",
"telefone fixo a tocar duas vezes", "carros a buzinar ao longe", "avião a passar", "máquina de escrever", "folhear de jornal",
"passos em corredor", "carimbo sobre papel". Nunca juntes dois sons na mesma linha: faz duas linhas.
Depois de " @ " podes pôr 3 a 6 palavras EXATAS do parágrafo onde o som entra (ex.: "porta pesada a fechar @ A sede da Polícia").
Sem " @ ", o som entra distribuído ao longo do parágrafo.

NÃO sugiras sons de ambiente (chuva, trânsito de fundo, sala vazia, murmúrio): o gerador ainda não os faz bem. Só efeitos pontuais
ligados a objetos e ações que o texto nomeia ou sugere fisicamente. Música: no máximo UMA linha na cena toda, só se a cena a pedir,
no parágrafo onde entra, escrita assim: "música: piano solo lento, melancólico".

Quando há ação, objetos, deslocações ou confronto, sugere vários sons (cada objeto ou ação com o seu), para a cena ficar bem
preenchida. Parágrafos de opinião, análise, datas e cronologia seca não levam sons.

Lê cada parágrafo literalmente. Só podes sugerir sons de coisas que ACONTECEM fisicamente. Negações ("não bateu à porta"), hipóteses,
perguntas, metáforas e factos só contados de passagem NÃO dão som.
Faz dois passos e devolve-os.
PASSO 1, "reading": só para os parágrafos que têm uma negação, hipótese, pergunta, metáfora ou facto contado de passagem que
  poderia enganar, escreve {"item": n, "not_happening": ["o que NÃO acontece"]}. Os restantes não entram.
PASSO 2, "paragraphs": {"item": n, "sounds": ["linha 1", "linha 2"]} para cada parágrafo de NARRAÇÃO com sons.

Responde APENAS com JSON: {"reading": [...], "paragraphs": [...]}"""


def suggest_scene(an: dict, n: int, key: str, guide: str = "", notes: str = "") -> dict[int, str]:
    """O Claude lê a cena e sugere linhas de som por parágrafo. Devolve {item: 'linha\\nlinha'}."""
    sc = next(x for x in an["scenes"] if x["n"] == n)
    mine = [it for it in an["items"] if sc["from"] <= it["id"] <= sc["to"] and it["type"] in VISIBLE]
    tag = {"narracao": "NARRAÇÃO", "entrevista": "ENTREVISTA"}
    msg = f"Cena {sc['n']}: {sc['title']}\n\n" + "\n\n".join(
        f"[{it['id']}] {tag[it['type']]}{' · ' + it['speaker'] if it.get('speaker') else ''}: {it['text'][:1500]}" for it in mine)
    if notes.strip():
        msg += f"\n\nNOTAS DO UTILIZADOR PARA ESTA CENA (têm prioridade):\n{notes.strip()}"
    system = SUGGEST_SYSTEM
    if guide.strip():
        system += ("\n\n# GUIA DE ABORDAGEM DO UTILIZADOR (ignora tudo o que fala de ambientes: não os sugiras)\n\n" + guide)
    data = _call(system, msg, key, 8000, timeout=420)
    valid = {it["id"] for it in mine if it["type"] == "narracao"}
    out: dict[int, str] = {}
    paras = data.get("paragraphs", [])
    if isinstance(paras, dict):                         # {"8": ["som", ...]}
        paras = [{"item": k, "sounds": v} for k, v in paras.items()]
    for p in paras:
        if not isinstance(p, dict):
            continue
        try:
            iid = int(p.get("item"))
        except (TypeError, ValueError):
            continue
        sounds = [str(s).strip() for s in p.get("sounds", []) if str(s).strip()]
        sounds = [s for s in sounds if not re.match(r"^(ambiente|ambience)\b", s, re.I)]
        if isinstance(p.get("sounds"), str):
            sounds = [s.strip() for s in p["sounds"].splitlines() if s.strip()]
        if iid in valid and sounds:
            out[iid] = "\n".join(sounds[:8])
    if not out:
        raise RuntimeError(f"O Claude leu a cena mas não sugeriu sons (devolveu {len(paras)} parágrafos, {len(valid)} de narração). "
                           "Escreve uma nota para o Claude a pedir sons concretos, ou escreve os sons à mão.")
    return out


NORMALIZE_SYSTEM = """Convertes pedidos de som (em português ou inglês) para um gerador de áudio. Para cada pedido, na mesma ordem,
devolve: "prompt" curto em inglês (3 a 8 palavras, UM só som, simples e direto), "label" (nome curto em português, 2 a 3 palavras)
e "duration" em segundos (efeito: 1 a 8, conforme o som; música: 20 a 60). Pedidos que começam por "música" ou "music" são música: o
prompt descreve estilo, instrumentos e andamento em poucas palavras, sem nomes de artistas.
Responde APENAS com JSON: {"items": [{"prompt": "...", "label": "...", "duration": n}]}"""


def normalize_lines(lines: list[dict], key: str | None) -> list[dict]:
    """Linhas de som → prompts curtos em inglês (um pedido rápido ao Claude para a cena toda)."""
    fallback = [{"prompt": l["text"], "label": " ".join(l["text"].split()[:3]),
                 "duration": 20.0 if l["kind"] == "music" else 3.0} for l in lines]
    if not key or not lines:
        return fallback
    try:
        req = "\n".join(f"{i + 1}. {'música: ' if l['kind'] == 'music' else ''}{l['text']}" for i, l in enumerate(lines))
        data = _call(NORMALIZE_SYSTEM, req, key, 4000, timeout=90, model=sd.ANTHROPIC_FAST_MODEL)
        items = data.get("items", [])
        if len(items) != len(lines):
            return fallback
        out = []
        for x, fb in zip(items, fallback):
            try:
                dur = float(x.get("duration", fb["duration"]))
            except (TypeError, ValueError):
                dur = fb["duration"]
            out.append({"prompt": str(x.get("prompt") or fb["prompt"])[:300], "label": str(x.get("label") or fb["label"])[:40], "duration": dur})
        return out
    except Exception:
        return fallback


# ---------- ponte para a Estrutura Plus ----------
def cues_for_scene(an: dict, n: int, tl: dict, key: str | None) -> tuple[list[dict], int]:
    """Linhas de som de uma cena → sons na linha do tempo. Devolve (sons, nº de linhas sem tempo)."""
    sc = next((x for x in an.get("scenes", []) if x["n"] == n), None)
    tsc = next((x for x in tl.get("scenes", []) if x["n"] == n), None)
    if not sc or not tsc:
        return [], 0
    items = {it["id"]: it for it in an["items"]}
    timed = {b["block"]: b for b in tl["blocks"]}
    flat = [(iid, k, len(ls), l) for iid, ls in scene_lines(an, sc).items() for k, l in enumerate(ls)]
    if not flat:
        return [], 0
    norm = normalize_lines([f[3] for f in flat], key)
    cues, skipped = [], 0
    for (iid, k, m, l), nm in zip(flat, norm):
        blk = items[iid].get("block")
        b = timed.get(blk)
        if not b:
            skipped += 1
            continue
        blen = max(0.5, b["end"] - b["start"])
        start = b["start"] + blen * (k + 1) / (m + 1) - 0.3
        if l["anchor"]:
            text, a = sd._norm(items[iid]["text"]), sd._norm(l["anchor"])
            i = text.find(a)
            if i >= 0:
                start = b["start"] + blen * i / max(1, len(text)) - 0.3
        start = max(b["start"], min(b["end"] - 0.4, start))
        if l["kind"] == "music":
            dur = max(10.0, min(120.0, tsc["end"] - start, max(30.0, nm["duration"])))
            cues.append({"kind": "music", "label": nm["label"] or "Música", "prompt": nm["prompt"] + sd.MUSIC_SUFFIX, "block": blk,
                         "start": round(start, 2), "dur": round(dur, 2), "gain": sd.DEFAULT_GAIN["music"],
                         "desc": l["text"], "anchor": l["anchor"]})
            continue
        parts = sd.split_prompt(nm["prompt"])
        dur = max(0.8, min(10.0, nm["duration"]))
        for j, part in enumerate(parts):
            cues.append({"kind": "sfx", "label": nm["label"] if j == 0 else f"{nm['label']} ({j + 1})", "prompt": part, "block": blk,
                         "start": round(min(start + j * dur / len(parts), b["end"] - 0.3), 2), "dur": round(max(0.8, dur / len(parts)), 2),
                         "gain": sd.DEFAULT_GAIN["sfx"], "desc": l["text"], "anchor": l["anchor"]})
    return cues, skipped
