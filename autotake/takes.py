"""Agrupa ocorrências em takes por bloco, pontua e escolhe."""
from .text import HESITATIONS

MAX_GAP_IN_TAKE = 25.0     # segundos sem guião dentro do mesmo take
LONG_PAUSE = 1.2           # pausa "estranha" entre palavras seguidas
COMPLETE_COVERAGE = 0.85
FULL_SENTENCE = 0.8


def _take_metrics(take: dict, block: dict, words: list[dict], owned: set[int]) -> None:
    sents = take["sentences"]
    total = block["n_tokens"] or 1
    covered = sum(block["sentences"][s]["_n"] * o.coverage for s, o in sents.items())
    take["coverage"] = round(min(1.0, covered / total), 3)
    n_sent = len(block["sentences"])
    first_ok = 0 in sents and sents[0].coverage >= 0.5
    last_ok = (n_sent - 1) in sents and sents[n_sent - 1].coverage >= 0.7
    take["complete"] = take["coverage"] >= COMPLETE_COVERAGE and first_ok and last_ok
    i0 = min(o.i0 for o in sents.values())
    i1 = max(o.i1 for o in sents.values())
    take["i0"], take["i1"] = i0, i1
    take["start"], take["end"] = words[i0]["s"], words[i1]["e"]
    off, hes, pauses = 0, 0, 0
    for i in range(i0, i1 + 1):
        if i not in owned and words[i]["t"]:
            off += 1
            if words[i]["t"] in HESITATIONS:
                hes += 1
        if i > i0 and words[i]["s"] - words[i - 1]["e"] > LONG_PAUSE:
            pauses += 1
    take["off_script"], take["hesitations"], take["long_pauses"] = off, hes, pauses
    take["score"] = round(take["coverage"] * 100 - 2.5 * off - 3 * hes - 3 * pauses, 1)
    take["text"] = " ".join(words[i]["raw"] for i in range(i0, i1 + 1) if words[i]["raw"])
    take["sentences_covered"] = sorted(sents.keys())


def build_takes(blocks: list[dict], words: list[dict], occurrences) -> None:
    for block in blocks:
        for s in block["sentences"]:
            s["_n"] = len(s["tokens"])
    by_block: dict[int, list] = {}
    for o in occurrences:
        by_block.setdefault(o.block, []).append(o)

    for b, block in enumerate(blocks):
        occs = sorted(by_block.get(b, []), key=lambda o: o.i0)
        owned = {i for o in occs for i in range(o.i0, o.i1 + 1)}
        takes, cur, last_sent, last_end = [], None, -1, None
        for o in occs:
            gap = words[o.i0]["s"] - last_end if last_end is not None else 0
            if cur is None or o.sent <= last_sent or gap > MAX_GAP_IN_TAKE:
                cur = {"sentences": {}}
                takes.append(cur)
            cur["sentences"][o.sent] = o
            last_sent, last_end = o.sent, words[o.i1]["e"]
        for n, t in enumerate(takes, start=1):
            _take_metrics(t, block, words, owned)
            t["id"] = f"{block['id']}t{n}"
            t["n"] = n
        block["takes"] = takes
        _choose(block)
        for t in takes:
            t["segments"] = [[t["i0"], t["i1"]]]
            t.pop("sentences")
        for s in block["sentences"]:
            s.pop("_n", None)


def _choose(block: dict) -> None:
    takes = block["takes"]
    flags = []
    if not takes:
        block["auto"] = None
        block["confidence"] = "baixa"
        block["flags"] = ["Não encontrei este bloco na gravação."]
        block["composite"] = None
        return
    complete = [t for t in takes if t["complete"]]
    composite = _composite(block)
    block["composite"] = composite
    if complete:
        chosen = complete[-1]
        later = [t for t in takes if t["n"] > chosen["n"]]
        uses_pickup = composite and chosen["n"] in composite["takes"] and \
            any(n > chosen["n"] for n in composite["takes"])
        if uses_pickup:
            # o narrador regravou só uma parte depois do take completo: usa a retoma
            nums = ", ".join(str(n) for n in sorted(set(composite["takes"])) if n > chosen["n"])
            block["auto"] = {"mode": "composite"}
            block["confidence"] = "média"
            block["flags"] = [f"Usei o take {chosen['n']} com a retoma do take {nums}. Confirma o ponto de corte."]
            return
        if later:
            flags.append(f"Há {len(later)} retoma(s) parcial(is) depois do take {chosen['n']}.")
        best_other = max((t for t in complete if t is not chosen), key=lambda t: t["score"], default=None)
        if best_other and best_other["score"] > chosen["score"] + 12:
            flags.append(f"O take {best_other['n']} parece mais limpo.")
        if chosen["off_script"] + chosen["hesitations"] > 2:
            flags.append("O take escolhido tem palavras fora do guião.")
        block["auto"] = {"mode": "take", "take": chosen["n"]}
        conf = "alta" if chosen["coverage"] >= 0.93 and not flags else "média"
    elif composite:
        flags.append("Nenhum take está completo: proposta composta por frases de takes diferentes.")
        block["auto"] = {"mode": "composite"}
        conf = "baixa"
    else:
        flags.append("Nenhum take está completo.")
        block["auto"] = {"mode": "take", "take": max(takes, key=lambda t: t["coverage"])["n"]}
        conf = "baixa"
    block["confidence"] = conf
    block["flags"] = flags


def _composite(block: dict):
    """Para cada frase, a ocorrência completa mais recente em qualquer take."""
    segs = []
    for s in range(len(block["sentences"])):
        best = None
        for t in block["takes"]:
            o = t["sentences"].get(s)
            if o and o.coverage >= FULL_SENTENCE:
                best = (t["n"], o)
        if best is None:
            return None
        segs.append({"sent": s, "take": best[0], "i0": best[1].i0, "i1": best[1].i1})
    # frases seguidas do mesmo take juntam-se num só corte
    merged = []
    for g in segs:
        if merged and merged[-1]["take"] == g["take"] and g["i0"] > merged[-1]["i1"]:
            merged[-1]["i1"] = g["i1"]
        else:
            merged.append(dict(g))
    if len(merged) == 1:
        return None  # é igual a um take normal
    return {"segments": [[m["i0"], m["i1"]] for m in merged], "takes": [m["take"] for m in merged]}
