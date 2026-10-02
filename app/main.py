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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

from autotake.export import to_reaper
from autotake.pipeline import process, render_episode

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("DATA_DIR", ROOT / "data")) / "episodes"
DATA.mkdir(parents=True, exist_ok=True)
API_KEY = os.getenv("ELEVENLABS_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD")

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
        process(d, API_KEY, lambda stage, msg="": set_state(d, status=stage, message=msg))
        set_state(d, status="pronto", message="")
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


@app.on_event("startup")
def resume_unfinished():
    # jobs interrompidos por um reinício do servidor voltam para a fila
    for d in DATA.iterdir():
        st = read_json(d / "state.json", {})
        if st.get("status") in {"na fila", "a preparar", "a transcrever", "a alinhar"}:
            set_state(d, status="na fila")
            jobs.submit(run_autotake, d)


# ---------- API ----------
@app.get("/health")
def health():
    return {"ok": True, "scribe": bool(API_KEY)}


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
    set_state(d, status="na fila", message="")
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


@app.get("/api/episodes/{eid}/export/reaper")
def export_reaper(eid: str):
    d = ep_dir(eid)
    meta, result = read_json(d / "meta.json"), read_json(d / "result.json")
    if not result:
        raise HTTPException(400, "O episódio ainda não foi processado")
    rpp = to_reaper(result, read_json(d / "selections.json", {}), meta["audio_file"], meta["title"])
    name = f"{slug(meta['title'])}-autotake.rpp"
    return PlainTextResponse(rpp, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.delete("/api/episodes/{eid}")
def delete_episode(eid: str):
    shutil.rmtree(ep_dir(eid))
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
