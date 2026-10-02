"""Encontra todas as ocorrências de cada frase do guião na transcrição.

Método determinístico:
  1. Sementes: palavras distintivas (>=4 letras) das primeiras posições da frase
     dão posições candidatas de início na transcrição.
  2. Para cada candidato, alinhamento LCS palavra a palavra numa janela
     (rapidfuzz), escolhendo o fim que maximiza acertos menos palavras a mais.
  3. Cada ocorrência fica com cobertura (quanto da frase foi dita) e precisão
     (quanto do que foi dito pertence à frase).
  4. Resolução global: ocorrências sobrepostas de frases diferentes disputam
     a mesma zona do áudio; ganha a de melhor score.
"""
from collections import defaultdict
from dataclasses import dataclass, asdict

from rapidfuzz.distance import LCSseq

SEED_POSITIONS = 6
MIN_SEED_LEN = 4
EXTRA_PENALTY = 0.6
MIN_PRECISION = 0.6
MIN_COVERAGE = 0.3
MAX_PAUSE_IN_SENTENCE = 2.5


@dataclass
class Occurrence:
    block: int        # índice do bloco (0-based)
    sent: int         # índice da frase dentro do bloco
    i0: int           # primeira palavra na transcrição
    i1: int           # última palavra (inclusive)
    matched: int
    coverage: float
    precision: float
    score: float

    def to_dict(self):
        return asdict(self)


def _best_span(S: list[str], W: list[str], times: list[tuple] | None = None):
    """Alinha S dentro de W (a partir do início de W). Devolve (j0, j1, matched).

    O alinhamento pára quando deteta um recomeço: uma sequência de palavras a mais
    que repete o que já foi dito ("três ondas gigantes... três ondas gigantes varreram").
    Assim uma tentativa falhada e a seguinte ficam como ocorrências separadas.
    """
    pairs = []  # (i, j) de palavras iguais
    for tag, a0, a1, b0, b1 in LCSseq.opcodes(S, W):
        if tag == "insert" and pairs and (b1 - b0) >= 2:
            said = set(S[:a0])
            extra = W[b0:b1]
            if sum(1 for w in extra if w in said) / len(extra) >= 0.6:
                break
        if tag == "equal":
            new_pairs = list(zip(range(a0, a1), range(b0, b1)))
            if times and pairs:
                gap = times[new_pairs[0][1]][0] - times[pairs[-1][1]][1]
                if gap > MAX_PAUSE_IN_SENTENCE:
                    break
            pairs.extend(new_pairs)
    if not pairs:
        return None
    j0 = pairs[0][1]
    best, best_f = None, float("-inf")
    for k, (_, j) in enumerate(pairs):
        m = k + 1
        length = j - j0 + 1
        f = m - EXTRA_PENALTY * (length - m)
        if f > best_f:
            best_f, best = f, (j0, j, m)
    return best


def find_sentence(S: list[str], T: list[str], index: dict, block: int, sent: int, times=None) -> list[Occurrence]:
    n = len(S)
    if n == 0:
        return []
    seed_ks = [k for k in range(min(n, SEED_POSITIONS)) if len(S[k]) >= MIN_SEED_LEN] or list(range(min(n, 3)))
    starts = set()
    for k in seed_ks:
        for p in index.get(S[k], ()):
            starts.add(max(0, p - k))
    win = int(n * 1.35) + 3
    found: dict[tuple, Occurrence] = {}
    for s in sorted(starts):
        span = _best_span(S, T[s:s + win], times[s:s + win] if times else None)
        if not span:
            continue
        j0, j1, m = span
        i0, i1 = s + j0, s + j1
        length = i1 - i0 + 1
        cov, prec = m / n, m / length
        if m < min(3, n) or prec < MIN_PRECISION or cov < MIN_COVERAGE:
            continue
        score = m - EXTRA_PENALTY * (length - m)
        key = (i0, i1)
        if key not in found or found[key].score < score:
            found[key] = Occurrence(block, sent, i0, i1, m, round(cov, 3), round(prec, 3), round(score, 2))
    # supressão de não-máximos entre ocorrências da mesma frase
    kept: list[Occurrence] = []
    for o in sorted(found.values(), key=lambda o: -o.score):
        if all(o.i1 < k.i0 or o.i0 > k.i1 for k in kept):
            kept.append(o)
    return kept


def align(blocks: list[dict], words: list[dict]) -> list[Occurrence]:
    T = [w["t"] for w in words]
    times = [(w["s"], w["e"]) for w in words]
    index = defaultdict(list)
    for i, tok in enumerate(T):
        if tok:
            index[tok].append(i)
    candidates: list[Occurrence] = []
    for b, block in enumerate(blocks):
        for s, sent in enumerate(block["sentences"]):
            candidates.extend(find_sentence(sent["tokens"], T, index, b, s, times))
    # resolução global: cada palavra da transcrição pertence no máximo a uma frase
    owner = [None] * len(T)
    accepted: list[Occurrence] = []
    for o in sorted(candidates, key=lambda o: -o.score):
        span = o.i1 - o.i0 + 1
        taken = sum(1 for i in range(o.i0, o.i1 + 1) if owner[i] is not None)
        if taken / span > 0.3:
            continue
        for i in range(o.i0, o.i1 + 1):
            if owner[i] is None:
                owner[i] = o
        accepted.append(o)
    return sorted(accepted, key=lambda o: o.i0)
