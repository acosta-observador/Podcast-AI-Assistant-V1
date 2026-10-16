"""Podcast Plus AI Assistant: servidor web (API + interface)."""
import base64
import json
import os
import re
import secrets
import shutil
import threading
import time
import traceback
import unicodedata
import tempfile
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from starlette.background import BackgroundTask
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

import numpy as np

from autotake import analysis as analysis_tools, audio as audio_tools, sounddesign
from autotake.export import to_reaper
from autotake.pipeline import GAP_BETWEEN_BLOCKS, build_cuts, process, render_episode

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("DATA_DIR", ROOT / "data")) / "episodes"
DATA.mkdir(parents=True, exist_ok=True)
API_KEY = os.getenv("ELEVENLABS_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GEN_WORKERS = int(os.getenv("GEN_WORKERS", "4"))
APP_PASSWORD = os.getenv("APP_PASSWORD")

GUIDE_PATH = DATA.parent / "guia_sound_design.md"

app = FastAPI(title="Podcast Plus AI Assistant")
jobs = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "1")))          # AutoTake (pesado)
structure_jobs = ThreadPoolExecutor(max_workers=2)                              # Estrutura Plus: não espera pelo AutoTake
_lock = threading.Lock()


# ---------- acesso: palavra-passe única da equipa (HTTP Basic) ----------
@app.middleware("http")
async def password_gate(request: Request, call_next):
    if APP_PASSWORD and request.url.path != "/health":
        header = request.headers.get("authorization", "")
        ok = False
        if header.startswith("Basic "):
            try:
                _, pw = base64.b64decode(header[6:]).decode().split(":", 1)
                ok = secrets.compare_digest(pw, APP_PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Podcast Plus"'})
    return await call_next(request)


# ---------- helpers de armazenamento (uma pasta por episódio) ----------
def ep_dir(eid: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", eid):
        raise HTTPException(404)
    d = DATA / eid
    if not d.exists():
        raise HTTPException(404, "Episódio não encontrado")
    return d


def read_json(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(p)


def set_state(d: Path, **kw) -> None:
    with _lock:
        st = read_json(d / "state.json", {})
        st.update(kw, updated=time.time())
        write_json(d / "state.json", st)


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:40] or "episodio"


def safe_name(name: str) -> str:
    base = Path(name or "ficheiro").name
    return re.sub(r"[^\w.\-]+", "_", base)


# ---------- jobs em background ----------
def run_autotake(d: Path) -> None:
    try:
        tries = read_json(d / "state.json", {}).get("attempts", 0) + 1
        set_state(d, attempts=tries)
        process(d, API_KEY, lambda stage, msg="": set_state(d, status=stage, message=msg))
        set_state(d, status="pronto", message="", attempts=0)
    except Exception as e:
        traceback.print_exc()
        set_state(d, status="erro", message=str(e))


def run_render(d: Path) -> None:
    try:
        set_state(d, render="a montar")
        result = read_json(d / "result.json")
        render_episode(d, result, read_json(d / "selections.json", {}), d / "work" / "montagem.wav", scenes_of(d, result))
        set_state(d, render="pronto")
    except Exception as e:
        traceback.print_exc()
        set_state(d, render="erro", render_message=str(e))


def struct_path(d: Path) -> Path:
    return d / "structure.json"


def read_struct(d: Path) -> dict:
    st = read_json(struct_path(d), {"status": None, "message": "", "done": 0, "total": 0, "scene": None, "clips": []})
    for c in st["clips"]:
        if c.get("kind") == "music":
            c["lane"] = sounddesign.MUSIC_LANE    # versões antigas tinham menos faixas
        elif c.get("kind") == "ambience":
            c["lane"] = sounddesign.AMB_LANE
        else:
            c["lane"] = min(c.get("lane", 0), sounddesign.N_SFX_LANES - 1)
        c.setdefault("scene", 1)
    return st


def update_struct(d: Path, fn) -> None:
    with _lock:
        st = read_struct(d)
        fn(st)
        st["ts"] = time.time()
        write_json(struct_path(d), st)


def struct_busy(d: Path) -> bool:
    """Há sons a ser criados? Um trabalho parado há mais de 15 minutos (servidor reiniciado, falha) já não conta."""
    st = read_struct(d)
    running = st.get("status") in {"a planear", "a gerar"} or st.get("batch")
    return bool(running and time.time() - st.get("ts", 0) < 900)


def scenes_of(d: Path, result: dict) -> list[dict]:
    """Cenas da Estrutura Plus: as da Análise de Guião (as 'CENA n' do guião)."""
    an = load_analysis(d)
    sc = analysis_tools.scenes_for_structure(an, result["blocks"])
    if sc:
        return sc
    meta = read_json(d / "meta.json")
    return sounddesign.scenes_for(d / meta["script_file"], len(result["blocks"]))


def repositioned(clips: list[dict], tl: dict) -> list[dict]:
    """Os sons seguem o bloco a que pertencem: se os takes ou as pausas mudarem, mudam de sítio com ele."""
    by_block = {b["block"]: b for b in tl["blocks"]}
    out = []
    for c in clips:
        b = by_block.get(c.get("block"))
        out.append({**c, "start": round(b["start"] + c["rel"], 2)} if b and c.get("rel") is not None else c)
    return out


def run_structure(d: Path, scene: int) -> None:
    """Estrutura Plus, uma cena de cada vez: as regras locais escolhem os sons a partir do guião
    e a ElevenLabs gera cada um. Só os sons desta cena são substituídos."""
    try:
        update_struct(d, lambda st: st.update(status="a planear", message="", done=0, total=0, scene=scene))
        result, sel = read_json(d / "result.json"), read_json(d / "selections.json", {})
        scenes = scenes_of(d, result)
        tl = sounddesign.timeline(result, sel, scenes)
        sc = next((x for x in tl["scenes"] if x["n"] == scene), None)
        if not sc:
            raise RuntimeError("Cena não encontrada.")
        an = load_analysis(d)
        analysis_tools.link_blocks(an["items"], result["blocks"])
        asc = next((x for x in an["scenes"] if x["n"] == scene), None)
        cues, skipped = analysis_tools.cues_for_scene(an, scene, tl, ANTHROPIC_KEY)
        if not cues:
            raise RuntimeError("Esta cena não tem sons escritos na Análise de Guião. Escreve (ou pede ao Claude para sugerir) "
                               "os sons de cada parágrafo e volta a carregar em Create.")
        planner, mood, reading = "analysis", asc["title"] if asc else "", []
        sig = analysis_tools.lines_sig(an, asc)
        note = f"{skipped} linhas de som não entraram: o parágrafo não tem take do narrador." if skipped else ""
        by_block = {b["block"]: b for b in tl["blocks"]}
        for c in cues:
            b = by_block.get(c["block"])
            c["rel"] = round(c["start"] - b["start"], 2) if b else None
        sounddesign.assign_lanes(cues)
        (d / "sd").mkdir(exist_ok=True)

        locals_sig = [sig]

        def start(st):
            for old in [c for c in st["clips"] if c.get("scene") == scene]:
                (d / "sd" / f"{old['id']}.mp3").unlink(missing_ok=True)
            st["clips"] = [c for c in st["clips"] if c.get("scene") != scene]
            nxt = st.get("next", 1 + max([int(c["id"][1:]) for c in st["clips"]] or [0]))
            for c in cues:
                st["clips"].append({**c, "id": f"c{nxt}", "scene": scene, "state": "a gerar", "peaks": [], "message": ""})
                nxt += 1
            st.update(status="a gerar", next=nxt, total=len(cues), done=0, planner=planner, mood=mood, note=note)
            st.setdefault("sigs", {})[str(scene)] = locals_sig[0]
            st.setdefault("readings", {})[str(scene)] = reading
        update_struct(d, start)
        todo = [c["id"] for c in read_struct(d)["clips"] if c.get("scene") == scene and c["state"] == "a gerar"]

        def work(cid):
            gen_clip(d, cid)
            update_struct(d, lambda st: st.update(done=st["done"] + 1))
        with ThreadPoolExecutor(max_workers=GEN_WORKERS) as pool:
            list(pool.map(work, todo))
        update_struct(d, lambda st: st.update(status="pronto"))
    except Exception as e:
        traceback.print_exc()
        update_struct(d, lambda st: st.update(status="erro", message=str(e)))


def run_structure_all(d: Path, scenes: list[int]) -> None:
    """Todas as cenas, uma a seguir à outra. 'batch' diz quantas faltam, para a página continuar a atualizar."""
    for i, n in enumerate(scenes):
        update_struct(d, lambda st: st.update(batch=len(scenes) - i))
        run_structure(d, n)
        if read_struct(d).get("status") == "erro":
            break
    update_struct(d, lambda st: st.update(batch=0))


def gen_clip(d: Path, cid: str) -> None:
    """Gera (ou volta a gerar) o áudio de um clip; os erros ficam no próprio clip."""
    clip = next(c for c in read_struct(d)["clips"] if c["id"] == cid)
    (d / "sd").mkdir(exist_ok=True)
    dst = d / "sd" / f"{cid}.mp3"
    try:
        sounddesign.generate(clip, API_KEY, dst)
        peaks = sounddesign.audio_peaks(dst, max(8, min(160, round(clip["dur"] * 3))))
        patch = {"state": "gerado", "peaks": peaks, "message": ""}
    except Exception as e:
        traceback.print_exc()
        patch = {"state": "erro", "message": str(e)}

    def apply(st):
        for c in st["clips"]:
            if c["id"] == cid:
                c.update(patch)
    update_struct(d, apply)


def run_clip(d: Path, cid: str) -> None:
    gen_clip(d, cid)


@app.on_event("startup")
def resume_unfinished():
    # jobs interrompidos por um reinício do servidor voltam para a fila
    for d in DATA.iterdir():
        aa = read_json(d / "analysis.json", {})
        if aa.get("status") == "a sugerir sons":
            aa.update(status="erro", message="O servidor reiniciou a meio. Carrega em Sugerir sons para tentar outra vez.")
            write_json(d / "analysis.json", aa)
        ss = read_json(struct_path(d), {})
        if ss.get("status") in {"a planear", "a gerar"}:
            ss.update(status="erro", message="O servidor reiniciou a meio. Carrega em Create para tentar outra vez.", batch=0)
            write_json(struct_path(d), ss)
        st = read_json(d / "state.json", {})
        if st.get("status") in {"na fila", "a preparar", "a transcrever", "a alinhar"}:
            if st.get("attempts", 0) >= 2:
                # o servidor já reiniciou duas vezes a meio deste episódio (provavelmente falta de memória)
                set_state(d, status="erro", message="O servidor ficou sem memória a processar este episódio. "
                                                    "Tenta um ficheiro mais curto ou um plano com mais memória.")
                continue
            set_state(d, status="na fila")
            jobs.submit(run_autotake, d)


# ---------- API ----------
@app.get("/health")
def health():
    return {"ok": True, "scribe": bool(API_KEY), "anthropic": bool(ANTHROPIC_KEY)}


@app.get("/api/episodes")
def list_episodes():
    items = []
    for d in sorted(DATA.iterdir(), reverse=True):
        meta = read_json(d / "meta.json")
        if meta:
            items.append({**meta, "state": read_json(d / "state.json", {})})
    return items


@app.post("/api/episodes")
def create_episode(title: str = Form(...), audio: UploadFile = File(...), script: UploadFile = File(...)):
    if not API_KEY:
        raise HTTPException(400, "O servidor não tem a chave do Scribe configurada (ELEVENLABS_API_KEY).")
    eid = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(title)}"
    d = DATA / eid
    d.mkdir(parents=True)
    audio_name, script_name = safe_name(audio.filename), "guiao-" + safe_name(script.filename)
    for up, name in ((audio, audio_name), (script, script_name)):
        with open(d / name, "wb") as out:
            shutil.copyfileobj(up.file, out, length=8 * 1024 * 1024)
    meta = {"id": eid, "title": title.strip(), "created": time.time(),
            "audio_file": audio_name, "script_file": script_name}
    write_json(d / "meta.json", meta)
    set_state(d, status="na fila", message="")
    jobs.submit(run_autotake, d)
    return meta


def _montage_total(d: Path):
    try:
        result = read_json(d / "result.json")
        return sounddesign.timeline(result, read_json(d / "selections.json", {}), scenes_of(d, result))["total"] if result else None
    except Exception:
        return None


@app.get("/api/episodes/{eid}")
def get_episode(eid: str):
    d = ep_dir(eid)
    return {
        "meta": read_json(d / "meta.json"),
        "state": read_json(d / "state.json", {}),
        "result": read_json(d / "result.json"),
        "selections": read_json(d / "selections.json", {}),
        "reviewed": read_json(d / "reviewed.json", []),
        "montage": _montage_total(d),
    }


@app.post("/api/episodes/{eid}/blocks/{bid}")
async def update_block(eid: str, bid: str, request: Request):
    """Corpo: {"choice": {"mode": "take", "take": 2} | {"mode": "composite"} | null, "reviewed": bool}"""
    d = ep_dir(eid)
    body = await request.json()
    with _lock:
        if "choice" in body:
            sel = read_json(d / "selections.json", {})
            if body["choice"] is None:
                sel.pop(bid, None)
            else:
                sel[bid] = body["choice"]
            write_json(d / "selections.json", sel)
        if "reviewed" in body:
            rev = set(read_json(d / "reviewed.json", []))
            (rev.add if body["reviewed"] else rev.discard)(bid)
            write_json(d / "reviewed.json", sorted(rev))
    # qualquer alteração invalida a montagem já feita
    set_state(d, render=None)
    return {"ok": True}


def run_reprocess(d: Path) -> None:
    """Volta a alinhar e mantém as escolhas dos blocos cujo texto não mudou."""
    old = read_json(d / "result.json")
    old_texts = {b["id"]: b["text"] for b in old["blocks"]} if old else {}
    run_autotake(d)
    new = read_json(d / "result.json")
    if not old or not new:
        return
    key = lambda t: re.sub(r"\s+", " ", t).strip()
    by_text: dict[str, str] = {}
    for b in new["blocks"]:
        by_text.setdefault(key(b["text"]), b["id"])
    mapping = {oid: by_text.get(key(t)) for oid, t in old_texts.items()}
    with _lock:
        sel = read_json(d / "selections.json", {})
        write_json(d / "selections.json", {mapping[k]: v for k, v in sel.items() if mapping.get(k)})
        rev = read_json(d / "reviewed.json", [])
        write_json(d / "reviewed.json", sorted({mapping[k] for k in rev if mapping.get(k)}))
    if [key(b["text"]) for b in old["blocks"]] != [key(b["text"]) for b in new["blocks"]]:
        # os blocos mudaram: as posições da Estrutura Plus deixam de bater certo
        shutil.rmtree(d / "sd", ignore_errors=True)
        update_struct(d, lambda st: st.update(clips=[], status=None, done=0, total=0, batch=0,
                                              note="Os blocos do episódio mudaram ao voltar a alinhar o guião. Refaz o Create das cenas."))


@app.post("/api/episodes/{eid}/reprocess")
def reprocess(eid: str):
    """Volta a alinhar (usa a transcrição em cache, não gasta Scribe)."""
    d = ep_dir(eid)
    set_state(d, status="na fila", message="", attempts=0)
    jobs.submit(run_reprocess, d)
    return {"ok": True}


@app.get("/api/episodes/{eid}/audio")
def preview_audio(eid: str):
    p = ep_dir(eid) / "work" / "preview.mp3"
    if not p.exists():
        raise HTTPException(404, "Pré-visualização ainda não disponível")
    return FileResponse(p, media_type="audio/mpeg")


@app.post("/api/episodes/{eid}/render")
def start_render(eid: str):
    d = ep_dir(eid)
    if not (d / "result.json").exists():
        raise HTTPException(400, "O episódio ainda não foi processado")
    set_state(d, render="na fila")
    jobs.submit(run_render, d)
    return {"ok": True}


@app.get("/api/episodes/{eid}/export/wav")
def export_wav(eid: str):
    d = ep_dir(eid)
    p = d / "work" / "montagem.wav"
    if not p.exists():
        raise HTTPException(404, "Ainda não há montagem")
    return FileResponse(p, media_type="audio/wav", filename=f"{slug(read_json(d / 'meta.json')['title'])}-autotake.wav")


@app.get("/api/episodes/{eid}/blocks/{bid}/export")
def export_block(eid: str, bid: str):
    """WAV só com o take escolhido (ou a montagem composta) de um bloco."""
    d = ep_dir(eid)
    meta, result = read_json(d / "meta.json"), read_json(d / "result.json")
    if not result:
        raise HTTPException(400, "O episódio ainda não foi processado")
    block = next((b for b in result["blocks"] if b["id"] == bid), None)
    if not block:
        raise HTTPException(404, "Bloco não encontrado")
    cuts = [dict(c) for c in build_cuts(result, read_json(d / "selections.json", {})) if c["block"] == block["index"]]
    if not cuts:
        raise HTTPException(400, "Este bloco não tem take escolhido")
    cuts[-1]["gap_after"] = 0
    which = "composta" if len({c["take"] for c in cuts}) > 1 or len(cuts) > 1 else f"take-{cuts[0]['take']}"
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "bloco.wav"
        audio_tools.render(d / meta["audio_file"], cuts, dst)
        data = dst.read_bytes()
    name = f"{slug(meta['title'])}-bloco-{block['index']}-{which}.wav"
    return Response(data, media_type="audio/wav", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/episodes/{eid}/export/reaper")
def export_reaper(eid: str):
    d = ep_dir(eid)
    meta, result = read_json(d / "meta.json"), read_json(d / "result.json")
    if not result:
        raise HTTPException(400, "O episódio ainda não foi processado")
    rpp = to_reaper(result, read_json(d / "selections.json", {}), meta["audio_file"], meta["title"], scenes=scenes_of(d, result))
    name = f"{slug(meta['title'])}-autotake.rpp"
    return PlainTextResponse(rpp, headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ---------- Análise de Guião ----------
def analysis_path(d: Path) -> Path:
    return d / "analysis.json"


def load_analysis(d: Path) -> dict:
    """Cenas e parágrafos do guião, sempre prontos (sem Claude). Migra análises antigas."""
    with _lock:
        an = read_json(analysis_path(d))
        if not an or an.get("v") != 2:
            meta = read_json(d / "meta.json")
            items = analysis_tools.read_items(d / meta["script_file"])
            lines: dict[str, str] = {}
            for sc in (an or {}).get("scenes", []):
                for c in sc.get("cues", []):                  # sons da versão anterior → linhas
                    if c.get("kind") in ("sfx", "music"):
                        txt = (c.get("label") or c.get("description") or "").strip()
                        line = ("música: " if c["kind"] == "music" else "") + txt + (f" @ {c['anchor']}" if c.get("anchor") else "")
                        lines[str(c["item"])] = (lines.get(str(c["item"]), "") + "\n" + line).strip()
            an = {"v": 2, "status": None, "message": "", "done": 0, "total": 0, "items": items,
                  "scenes": analysis_tools.local_scenes(items), "lines": lines, "notes": (an or {}).get("notes", {})}
            write_json(analysis_path(d), an)
        return an


def update_analysis(d: Path, fn) -> None:
    load_analysis(d)
    with _lock:
        an = read_json(analysis_path(d))
        fn(an)
        write_json(analysis_path(d), an)


analysis_jobs = ThreadPoolExecutor(max_workers=1)


def run_suggest(d: Path, todo: list[int], notes: dict) -> None:
    """O Claude sugere as linhas de som de cada cena (3 de cada vez)."""
    try:
        guide = sounddesign.load_guide(GUIDE_PATH)
        update_analysis(d, lambda an: an.update(status="a sugerir sons", message="", done=0, total=len(todo), errors={}))

        def work(n):
            an = read_json(analysis_path(d))
            try:
                got = analysis_tools.suggest_scene(an, n, ANTHROPIC_KEY, guide, notes.get(str(n), ""))
                err = ""
            except Exception as e:
                traceback.print_exc()
                got, err = None, str(e)

            def apply(a):
                sc = next(x for x in a["scenes"] if x["n"] == n)
                if got is not None:
                    norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
                    for iid, text in got.items():          # acrescenta, sem apagar nem repetir o que já lá está
                        old = [l for l in a["lines"].get(str(iid), "").splitlines() if l.strip()]
                        seen = {norm(l) for l in old}
                        a["lines"][str(iid)] = "\n".join(old + [l for l in text.splitlines() if l.strip() and norm(l) not in seen])
                else:
                    a.setdefault("errors", {})[str(n)] = err
                a["done"] += 1
            update_analysis(d, apply)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(work, todo))
        update_analysis(d, lambda an: an.update(status="pronto"))
    except Exception as e:
        traceback.print_exc()
        update_analysis(d, lambda an: an.update(status="erro", message=str(e)))


@app.get("/api/episodes/{eid}/analysis")
def get_analysis(eid: str):
    d = ep_dir(eid)
    return {"meta": read_json(d / "meta.json"), "analysis": load_analysis(d), "claude": bool(ANTHROPIC_KEY)}


@app.put("/api/episodes/{eid}/analysis/lines")
def put_lines(eid: str, body: dict = Body(...)):
    """Os sons de um parágrafo: texto livre, uma linha por som."""
    d = ep_dir(eid)
    text = str(body.get("text", ""))[:3000]
    iid = str(int(body.get("item")))

    def apply(an):
        if text.strip():
            an["lines"][iid] = text
        else:
            an["lines"].pop(iid, None)
    update_analysis(d, apply)
    return {"ok": True}


@app.post("/api/episodes/{eid}/analysis/clear")
def clear_analysis_scene(eid: str, body: dict = Body(...)):
    """Apaga os sons escritos de uma cena."""
    d = ep_dir(eid)
    n = int(body.get("scene") or 0)

    def apply(an):
        sc = next((s for s in an["scenes"] if s["n"] == n), None)
        if sc:
            for iid in range(sc["from"], sc["to"] + 1):
                an["lines"].pop(str(iid), None)
    update_analysis(d, apply)
    return {"ok": True}


@app.post("/api/episodes/{eid}/analysis/run")
def suggest_endpoint(eid: str, body: dict = Body(default={})):
    """Corpo: {"scene": n | "all", "replace": bool, "notes": "..."}. Sugere sons (substitui o texto da cena)."""
    d = ep_dir(eid)
    if not ANTHROPIC_KEY:
        raise HTTPException(400, "O servidor não tem a chave da Anthropic configurada (ANTHROPIC_API_KEY).")
    an = load_analysis(d)
    if an.get("status") == "a sugerir sons":
        raise HTTPException(409, "O Claude já está a sugerir sons")
    if body.get("scene") == "all":
        todo = [s["n"] for s in an["scenes"] if body.get("replace") or analysis_tools.count_lines(an, s) == 0]
    else:
        todo = [int(body.get("scene") or 0)]
        if not any(s["n"] == todo[0] for s in an["scenes"]):
            raise HTTPException(400, "Cena não encontrada")
    if not todo:
        raise HTTPException(400, "Todas as cenas já têm sons escritos")
    notes = dict(an.get("notes") or {})
    if body.get("scene") != "all" and "notes" in body:
        notes[str(todo[0])] = str(body["notes"])[:2000]
        update_analysis(d, lambda a: a.setdefault("notes", {}).__setitem__(str(todo[0]), notes[str(todo[0])]))
    update_analysis(d, lambda a: a.update(status="a sugerir sons", message="", done=0, total=len(todo)))
    analysis_jobs.submit(run_suggest, d, todo, notes)
    return {"ok": True, "scenes": todo}


# ---------- guia de sound design (editável) ----------
@app.get("/api/sound-guide")
def get_guide():
    return {"text": sounddesign.load_guide(GUIDE_PATH), "custom": GUIDE_PATH.exists()}


@app.put("/api/sound-guide")
async def put_guide(request: Request):
    body = await request.json()
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(400, "O guia não pode ficar vazio")
    GUIDE_PATH.write_text(text[:30000], encoding="utf-8")
    return {"ok": True}


@app.delete("/api/sound-guide")
def reset_guide():
    GUIDE_PATH.unlink(missing_ok=True)
    return {"ok": True}


# ---------- Estrutura Plus ----------
@app.get("/api/episodes/{eid}/structure")
def get_structure(eid: str):
    d = ep_dir(eid)
    result = read_json(d / "result.json")
    if not result:
        raise HTTPException(400, "O episódio ainda não foi processado")
    tl = sounddesign.timeline(result, read_json(d / "selections.json", {}), scenes_of(d, result))
    env_p = d / "work" / "envelope.npy"
    peaks = sounddesign.narration_peaks(np.load(env_p), tl["blocks"], tl["segments"]) if env_p.exists() else {}
    for b in tl["blocks"]:
        b["peaks"] = peaks.get(b["block"], [])
    an = load_analysis(d)
    planned = {str(x["n"]): analysis_tools.count_lines(an, x) for x in an.get("scenes", [])}
    sigs = {str(x["n"]): analysis_tools.lines_sig(an, x) for x in an.get("scenes", [])}
    structure = read_struct(d)
    structure["clips"] = repositioned(structure["clips"], tl)
    return {"meta": read_json(d / "meta.json"), "structure": structure, "timeline": tl, "planned": planned, "sigs": sigs, "has_analysis": True,
            "ready": bool(API_KEY), "claude": bool(ANTHROPIC_KEY), "missing": [] if API_KEY else ["ELEVENLABS_API_KEY"]}


@app.post("/api/episodes/{eid}/structure/create")
async def create_structure(eid: str, request: Request):
    """Corpo: {"scene": 1}. Cria o sound design só dessa cena."""
    d = ep_dir(eid)
    result = read_json(d / "result.json")
    if not result:
        raise HTTPException(400, "O episódio ainda não foi processado")
    if not API_KEY:
        raise HTTPException(400, "O servidor não tem a chave da ElevenLabs configurada (ELEVENLABS_API_KEY).")
    body = await request.json() if (await request.body()) else {}
    notes = str(body.get("notes") or "")[:2000]
    all_scenes = scenes_of(d, result)
    if body.get("scene") == "all":
        have = {c.get("scene") for c in read_struct(d)["clips"]}
        todo = [sc["n"] for sc in all_scenes if not (body.get("missing_only") and sc["n"] in have)]
        if not todo:
            raise HTTPException(400, "Todas as cenas já têm sons")
        if struct_busy(d):
            raise HTTPException(409, "Já há sound design a ser criado")
        update_struct(d, lambda st: st.update(status="a planear", message="", done=0, total=0, scene=todo[0], batch=len(todo)))
        structure_jobs.submit(run_structure_all, d, todo)
        return {"ok": True, "scenes": todo}
    scene = int(body.get("scene") or 0)
    if not any(sc["n"] == scene for sc in all_scenes):
        raise HTTPException(400, "Escolhe uma cena")
    if struct_busy(d):
        raise HTTPException(409, "Já há sound design a ser criado")
    update_struct(d, lambda st: (st.update(status="a planear", message="", done=0, total=0, scene=scene),
                                 st.setdefault("notes", {}).__setitem__(str(scene), notes)))
    structure_jobs.submit(run_structure, d, scene)
    return {"ok": True}


@app.post("/api/episodes/{eid}/structure/clips")
def add_clip(eid: str, body: dict = Body(...)):
    """Som criado à mão: {"prompt": "...", "kind": "sfx|ambience|music", "start": s, "dur": s, "lane": n}."""
    d = ep_dir(eid)
    result = read_json(d / "result.json")
    if not result:
        raise HTTPException(400, "O episódio ainda não foi processado")
    if not API_KEY:
        raise HTTPException(400, "O servidor não tem a chave da ElevenLabs configurada (ELEVENLABS_API_KEY).")
    text = str(body.get("prompt", "")).strip()[:300]
    if not text:
        raise HTTPException(400, "Escreve o que queres ouvir")
    kind = body.get("kind") if body.get("kind") in ("sfx", "ambience", "music") else "sfx"
    try:
        start = max(0.0, float(body.get("start", 0)))
        dur = float(body.get("dur", 3))
    except (TypeError, ValueError):
        raise HTTPException(400, "Tempo inválido")
    dur = max(0.5, min(dur, {"sfx": 22.0, "ambience": 60.0, "music": 120.0}[kind]))
    tl = sounddesign.timeline(result, read_json(d / "selections.json", {}), scenes_of(d, result))
    blk = next((b["block"] for b in tl["blocks"] if b["start"] <= start <= b["end"] + GAP_BETWEEN_BLOCKS),
               max([b["block"] for b in tl["blocks"] if b["start"] <= start] or [1]))
    scene = next((x["n"] for x in tl["scenes"] if x["start"] - 1 <= start <= x["end"] + 3), 1)
    lane = {"ambience": sounddesign.AMB_LANE, "music": sounddesign.MUSIC_LANE}.get(
        kind, min(max(int(body.get("lane", 0)), 0), sounddesign.N_SFX_LANES - 1))
    sp = sounddesign.simple_prompt(text, ANTHROPIC_KEY, kind)
    made = {}

    def apply(st):
        nxt = st.get("next", 1 + max([int(c["id"][1:]) for c in st["clips"]] or [0]))
        clip = {"id": f"c{nxt}", "kind": kind, "label": sp["label"], "prompt": sp["prompt"], "block": blk, "scene": scene,
                "lane": lane, "start": round(start, 2), "dur": round(dur, 2), "gain": sounddesign.DEFAULT_GAIN[kind],
                "rel": round(start - next((b["start"] for b in tl["blocks"] if b["block"] == blk), start), 2),
                "desc": text, "anchor": "", "state": "a gerar", "peaks": [], "message": ""}
        st["clips"].append(clip)
        st["next"] = nxt + 1
        made.update(clip)
    update_struct(d, apply)
    structure_jobs.submit(run_clip, d, made["id"])
    return made


@app.patch("/api/episodes/{eid}/structure/clips/{cid}")
def patch_clip(eid: str, cid: str, body: dict = Body(...)):
    d = ep_dir(eid)

    def apply(st):
        for c in st["clips"]:
            if c["id"] == cid and "gain" in body:
                c["gain"] = max(0.05, min(1.0, float(body["gain"])))
    update_struct(d, apply)
    return {"ok": True}


@app.post("/api/episodes/{eid}/structure/clear")
def clear_structure(eid: str, body: dict = Body(default={})):
    """Balde do lixo: apaga todos os sons de uma cena ({"scene": n}) ou do episódio todo ({"scene": "all"})."""
    d = ep_dir(eid)
    st = read_struct(d)
    if struct_busy(d):
        raise HTTPException(409, "Espera que acabe a criação dos sons")
    scene = None if body.get("scene") == "all" else int(body.get("scene") or 0)
    gone = []

    def apply(s):
        keep = []
        for c in s["clips"]:
            if scene is None or c.get("scene") == scene:
                gone.append(c["id"])
            else:
                keep.append(c)
        s["clips"] = keep
        sigs = s.setdefault("sigs", {})
        if scene is None:
            sigs.clear()
        else:
            sigs.pop(str(scene), None)
        if not keep:
            s.update(status=None, note="", done=0, total=0)
    update_struct(d, apply)
    for cid in gone:
        (d / "sd" / f"{cid}.mp3").unlink(missing_ok=True)
    return {"ok": True, "removed": len(gone)}


@app.post("/api/episodes/{eid}/structure/clips/{cid}/regenerate")
async def regenerate_clip(eid: str, cid: str, request: Request):
    d = ep_dir(eid)
    body = await request.json() if (await request.body()) else {}
    if not any(c["id"] == cid for c in read_struct(d)["clips"]):
        raise HTTPException(404)

    def apply(st):
        for c in st["clips"]:
            if c["id"] == cid:
                c["state"] = "a gerar"
                if body.get("prompt"):
                    c["prompt"] = str(body["prompt"])[:400]
    update_struct(d, apply)
    structure_jobs.submit(run_clip, d, cid)
    return {"ok": True}


@app.delete("/api/episodes/{eid}/structure/clips/{cid}")
def delete_clip(eid: str, cid: str):
    d = ep_dir(eid)
    if not re.fullmatch(r"c\d+", cid):
        raise HTTPException(404)
    update_struct(d, lambda st: st.update(clips=[c for c in st["clips"] if c["id"] != cid]))
    (d / "sd" / f"{cid}.mp3").unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/episodes/{eid}/structure/clips/{cid}/audio")
def clip_audio(eid: str, cid: str):
    if not re.fullmatch(r"c\d+", cid):
        raise HTTPException(404)
    p = ep_dir(eid) / "sd" / f"{cid}.mp3"
    if not p.exists():
        raise HTTPException(404, "Áudio ainda não disponível")
    return FileResponse(p, media_type="audio/mpeg")


def _track_items(d: Path, lane: int, tl: dict, scene: int | None):
    """Sons de uma faixa e a janela de tempo a exportar (episódio todo, ou só a cena escolhida)."""
    clips = repositioned([c for c in read_struct(d)["clips"] if c["state"] == "gerado" and c.get("lane") == lane], tl)
    t0, t1 = 0.0, max([tl["total"]] + [c["start"] + c["dur"] for c in clips]) + 1.0
    if scene:
        sc = next((x for x in tl["scenes"] if x["n"] == scene), None)
        if not sc:
            raise HTTPException(404, "Cena não encontrada")
        t0, t1 = max(0.0, sc["start"] - 1.0), sc["end"] + 1.0
        clips = [c for c in clips if c["start"] + c["dur"] > t0 and c["start"] < t1]
    items = [{"file": d / "sd" / f"{c['id']}.mp3", "start": c["start"] - t0, "gain": c.get("gain") or 0.8} for c in clips]
    return items, t1 - t0


def _structure_timeline(d: Path) -> dict:
    result = read_json(d / "result.json")
    if not result:
        raise HTTPException(400, "O episódio ainda não foi processado")
    return sounddesign.timeline(result, read_json(d / "selections.json", {}), scenes_of(d, result))


@app.get("/api/episodes/{eid}/export/track/{lane}")
def export_track(eid: str, lane: int, scene: int | None = None):
    """Uma faixa (Efeitos 1 a 5, Ambiente, Música) como WAV alinhado ao tempo. Com ?scene=n, só essa cena."""
    d = ep_dir(eid)
    if not 0 <= lane < len(sounddesign.LANE_NAMES):
        raise HTTPException(404)
    items, total = _track_items(d, lane, _structure_timeline(d), scene)
    if not items:
        raise HTTPException(400, "Esta faixa não tem sons")
    tmp = Path(tempfile.mkdtemp()) / "faixa.wav"
    audio_tools.mix_track(items, total, tmp)
    name = f"{slug(read_json(d / 'meta.json')['title'])}-{slug(sounddesign.LANE_NAMES[lane])}{f'-cena-{scene}' if scene else ''}.wav"
    return FileResponse(tmp, media_type="audio/wav", filename=name,
                        background=BackgroundTask(shutil.rmtree, tmp.parent, ignore_errors=True))


@app.get("/api/episodes/{eid}/export/tracks")
def export_tracks(eid: str, scene: int | None = None):
    """ZIP com uma WAV por faixa que tenha sons (episódio todo, ou só a cena escolhida)."""
    d = ep_dir(eid)
    tl = _structure_timeline(d)
    base = slug(read_json(d / "meta.json")["title"])
    work = Path(tempfile.mkdtemp())
    zpath = work / "faixas.zip"
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for lane, lname in enumerate(sounddesign.LANE_NAMES):
            items, total = _track_items(d, lane, tl, scene)
            if not items:
                continue
            wav = work / f"{lane}.wav"
            audio_tools.mix_track(items, total, wav)
            z.write(wav, f"{slug(lname)}{f'-cena-{scene}' if scene else ''}.wav")
            wav.unlink()
            n += 1
    if not n:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(400, "Ainda não há sons para exportar")
    return FileResponse(zpath, media_type="application/zip", filename=f"{base}-faixas{f'-cena-{scene}' if scene else ''}.zip",
                        background=BackgroundTask(shutil.rmtree, work, ignore_errors=True))


@app.get("/api/episodes/{eid}/export/structure")
def export_structure(eid: str):
    """ZIP com o projeto Reaper (takes + faixas de sound design) e os áudios gerados."""
    d = ep_dir(eid)
    meta, result = read_json(d / "meta.json"), read_json(d / "result.json")
    if not result:
        raise HTTPException(400, "O episódio ainda não foi processado")
    clips = [c for c in read_struct(d)["clips"] if c["state"] == "gerado"]
    tl = sounddesign.timeline(result, read_json(d / "selections.json", {}), scenes_of(d, result))
    rpp = to_reaper(result, read_json(d / "selections.json", {}), meta["audio_file"], meta["title"],
                    sound=repositioned(clips, tl), scenes=tl["scenes"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr(f"{slug(meta['title'])}-estrutura-plus.rpp", rpp)
        for c in clips:
            z.write(d / "sd" / f"{c['id']}.mp3", f"sd/{c['id']}.mp3")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{slug(meta["title"])}-estrutura-plus.zip"'})


@app.delete("/api/episodes/{eid}")
def delete_episode(eid: str):
    shutil.rmtree(ep_dir(eid))
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
