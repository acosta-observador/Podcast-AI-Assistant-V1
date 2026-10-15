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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

import numpy as np

from autotake import audio as audio_tools, sounddesign
from autotake.export import to_reaper
from autotake.pipeline import build_cuts, process, render_episode

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("DATA_DIR", ROOT / "data")) / "episodes"
DATA.mkdir(parents=True, exist_ok=True)
API_KEY = os.getenv("ELEVENLABS_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GEN_WORKERS = int(os.getenv("GEN_WORKERS", "4"))
APP_PASSWORD = os.getenv("APP_PASSWORD")

GUIDE_PATH = DATA.parent / "guia_sound_design.md"

app = FastAPI(title="Podcast Plus AI Assistant")
jobs = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "1")))
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
        render_episode(d, result, read_json(d / "selections.json", {}), d / "work" / "montagem.wav")
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


def scenes_of(d: Path, result: dict) -> list[dict]:
    meta = read_json(d / "meta.json")
    return sounddesign.scenes_for(d / meta["script_file"], len(result["blocks"]))


def update_struct(d: Path, fn) -> None:
    with _lock:
        st = read_struct(d)
        fn(st)
        write_json(struct_path(d), st)


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
        planner, mood, note, reading = "local", "", "", []
        cues = []
        if ANTHROPIC_KEY:
            try:
                guide = sounddesign.load_guide(GUIDE_PATH)
                notes = read_struct(d).get("notes", {}).get(str(scene), "")
                plan = sounddesign.plan_claude(tl, sc, ANTHROPIC_KEY, guide, notes, result["blocks"])
                cues, mood, planner, reading = plan["cues"], plan["mood"], "claude", plan["reading"]
            except Exception as e:
                traceback.print_exc()
                note = f"O Claude não conseguiu planear ({e}). Usei as regras simples."
        if not cues:
            part = {**tl, "blocks": [b for b in tl["blocks"] if b["block"] in sc["blocks"]]}
            cues = sounddesign.plan_local(part, span=(sc["start"], sc["end"]))
            planner = "local"
        if not cues:
            raise RuntimeError("Não encontrei momentos para sound design nesta cena.")
        sounddesign.assign_lanes(cues)
        (d / "sd").mkdir(exist_ok=True)

        def start(st):
            for old in [c for c in st["clips"] if c.get("scene") == scene]:
                (d / "sd" / f"{old['id']}.mp3").unlink(missing_ok=True)
            st["clips"] = [c for c in st["clips"] if c.get("scene") != scene]
            nxt = st.get("next", 1 + max([int(c["id"][1:]) for c in st["clips"]] or [0]))
            for c in cues:
                st["clips"].append({**c, "id": f"c{nxt}", "scene": scene, "state": "a gerar", "peaks": [], "message": ""})
                nxt += 1
            st.update(status="a gerar", next=nxt, total=len(cues), done=0, planner=planner, mood=mood, note=note)
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
        ss = read_json(struct_path(d), {})
        if ss.get("status") in {"a planear", "a gerar"}:
            ss.update(status="erro", message="O servidor reiniciou a meio. Carrega em Create para tentar outra vez.")
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


@app.get("/api/episodes/{eid}")
def get_episode(eid: str):
    d = ep_dir(eid)
    return {
        "meta": read_json(d / "meta.json"),
        "state": read_json(d / "state.json", {}),
        "result": read_json(d / "result.json"),
        "selections": read_json(d / "selections.json", {}),
        "reviewed": read_json(d / "reviewed.json", []),
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


@app.post("/api/episodes/{eid}/reprocess")
def reprocess(eid: str):
    """Volta a alinhar (usa a transcrição em cache, não gasta Scribe)."""
    d = ep_dir(eid)
    set_state(d, status="na fila", message="", attempts=0)
    jobs.submit(run_autotake, d)
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
    rpp = to_reaper(result, read_json(d / "selections.json", {}), meta["audio_file"], meta["title"])
    name = f"{slug(meta['title'])}-autotake.rpp"
    return PlainTextResponse(rpp, headers={"Content-Disposition": f'attachment; filename="{name}"'})


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
    return {"meta": read_json(d / "meta.json"), "structure": read_struct(d), "timeline": tl,
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
    scene = int(body.get("scene") or 0)
    notes = str(body.get("notes") or "")[:2000]
    if not any(sc["n"] == scene for sc in scenes_of(d, result)):
        raise HTTPException(400, "Escolhe uma cena")
    if read_struct(d).get("status") in {"a planear", "a gerar"}:
        raise HTTPException(409, "Já há sound design a ser criado")
    update_struct(d, lambda st: (st.update(status="a planear", message="", done=0, total=0, scene=scene),
                                 st.setdefault("notes", {}).__setitem__(str(scene), notes)))
    jobs.submit(run_structure, d, scene)
    return {"ok": True}


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
    jobs.submit(run_clip, d, cid)
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
                    sound=clips, gaps=tl["gaps"], scenes=tl["scenes"])
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
