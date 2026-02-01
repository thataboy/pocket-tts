import io
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from os.path import getmtime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional, Tuple

import scipy.io.wavfile
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pocket_tts import TTSModel
from pocket_tts.data.audio import stream_audio_chunks

logger = logging.getLogger(__name__)

VOICES_PATH = "./voices"

# Global model instance
tts_model = TTSModel.load_model(temp=0.7, lsd_decode_steps=1)

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)


@web_app.get("/voices")
def list_voices():
    """
    Return list of available voice style names.
    """
    return {"voices": sorted(list(voices.keys()))}


@web_app.get("/voices/refresh")
def refresh_voices():
    process_voices()
    return {"voices": sorted(list(voices.keys()))}


class SynthesizeRequest(BaseModel):
    input: str = ""
    voice: str = ""


@web_app.post("/synthesize")
@web_app.post("/v1/audio/speech")
def synthesize(req: SynthesizeRequest):
    """
    Generate complete text in one go
    """
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if req.voice not in voices:
        req.voice = next(iter(voices.keys())) if len(voices) > 0 else ""
        if not req.voice:
            raise HTTPException(status_code=400, detail="No voice found")

    print(f"{req.voice}➡️{req.input}⬅️")
    t0 = time.perf_counter()

    model_state = tts_model._cached_get_state_for_audio_prompt(voices[req.voice])
    audio = tts_model.generate_audio(model_state, req.input, frames_after_eos=2)

    buffer = io.BytesIO()
    sample_rate = tts_model.sample_rate
    scipy.io.wavfile.write(buffer, sample_rate, audio.numpy())
    elapsed = time.perf_counter() - t0
    num_samples = audio.shape[-1]
    duration = num_samples / sample_rate
    spd = duration / elapsed
    print(f"[{elapsed:.3f}s] len={len(req.input)} dur={duration:.2f}s  {spd:.3f}x")

    return Response(content=buffer.getvalue(), media_type="audio/wav")


def process_voices():
    print("Loading voices...")
    voices.clear()
    for path in Path(VOICES_PATH).iterdir():
        if not path.is_file() or path.suffix not in [".safetensors", ".wav"]:
            continue
        voice = path.stem
        if voice in voices:
            continue
        sft = path.with_suffix(".safetensors")
        wav = path.with_suffix(".wav")
        if not sft.exists() or wav.exists() and getmtime(wav) > getmtime(sft):
            print(f"Extracting voice {voice}")
            tts_model.save_audio_prompt(wav, sft, truncate=True)
        voices[voice] = sft

    # print(f"{len(voices)} voices loaded; {size_of_dict(voices) // 1e6} MB.")
    print(f"{len(voices)} voices loaded")


def generate_data_stream(text_to_generate: str, model_state: dict):
    queue = Queue()
    t0 = time.perf_counter()

    thread = threading.Thread(target=write_to_queue, args=(queue, text_to_generate, model_state))
    thread.start()

    i = 0
    try:
        while True:
            data = queue.get()

            if data is None:
                break

            i += 1
            yield data

    finally:
        # This runs even if the client disconnects (GeneratorExit)
        elapsed = time.perf_counter() - t0
        print(f"Total time: {elapsed:.3f}s | Chunks: {i}")

        # Clean up the thread
        thread.join()


def write_to_queue(queue, text_to_generate, model_state):
    """Allows writing to the StreamingResponse as if it were a file."""

    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue):
            self.queue = queue

        def write(self, data):
            self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    audio_chunks = tts_model.generate_audio_stream(
        model_state=model_state, text_to_generate=text_to_generate
    )
    stream_audio_chunks(FileLikeToQueue(queue), audio_chunks, tts_model.config.mimi.sample_rate)


@web_app.post("/stream")
def stream(req: SynthesizeRequest):
    return _stream(req.input, req.voice)


@web_app.post("/tts")
def text_to_speech(
    text: str = Form(...),
    voice: str | None = Form(None),
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
):
    return _stream(text, voice or voice_url)


def _stream(text, voice):
    if not text.strip():
        raise HTTPException(status_code=401, detail="Text cannot be empty")

    if voice not in voices:
        voice = next(iter(voices.keys())) if len(voices) > 0 else ""
        if not voice:
            raise HTTPException(status_code=402, detail="No voice found")

    print(f"stream [{len(text)}]➡️{voice}➡️{text}⬅️")

    model_state = tts_model._cached_get_state_for_audio_prompt(voices[voice])

    return StreamingResponse(
        generate_data_stream(text, model_state),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


ITALK_DATA_PATH = Path(__file__).parent / "italk.yaml"
ITALK_LOCK = threading.Lock()


def _italk_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


def _italk_default_state() -> Dict[str, Any]:
    return {
        "meta": {"version": 1, "created_at": _italk_now_iso()},
        "settings": {"last_voice": ""},
        "tags": [
            {"id": "tag_yes", "label": "Yes", "text": "Yes", "created_at": _italk_now_iso()},
            {"id": "tag_no", "label": "No", "text": "No", "created_at": _italk_now_iso()},
            {
                "id": "tag_repeat",
                "label": "Repeat",
                "text": "Could you repeat that",
                "created_at": _italk_now_iso(),
            },
            {
                "id": "tag_xcus",
                "label": "Xcus",
                "text": "Excuse me",
                "created_at": _italk_now_iso(),
            },
            {"id": "tag_thx", "label": "Thx", "text": "Thank you", "created_at": _italk_now_iso()},
            {
                "id": "tag_tkvm",
                "label": "TKVM",
                "text": "Thank you very much",
                "created_at": _italk_now_iso(),
            },
            {
                "id": "tag_hand",
                "label": "HAND",
                "text": "Have a nice day",
                "created_at": _italk_now_iso(),
            },
            {"id": "tag_bye", "label": "Bye", "text": "Bye", "created_at": _italk_now_iso()},
        ],
        "favorites": {"categories": {"Intro": [], "Contact": [], "Insurance": []}},
        "last_session": {"name": "", "started_at": _italk_now_iso(), "lines": []},
        "history": [],
    }


def _italk_load() -> Dict[str, Any]:
    if not ITALK_DATA_PATH.exists():
        return _italk_default_state()
    try:
        data = yaml.safe_load(ITALK_DATA_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _italk_default_state()
        defaults = _italk_default_state()
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
        if "favorites" not in data or "categories" not in data.get("favorites", {}):
            data["favorites"] = defaults["favorites"]
        if "last_session" not in data or "lines" not in data.get("last_session", {}):
            data["last_session"] = defaults["last_session"]
        if "tags" not in data or not isinstance(data.get("tags"), list):
            data["tags"] = defaults["tags"]
        if "history" not in data or not isinstance(data.get("history"), list):
            data["history"] = defaults["history"]
        if "settings" not in data or not isinstance(data.get("settings"), dict):
            data["settings"] = defaults["settings"]
        return data
    except Exception:
        return _italk_default_state()


def _italk_save(data: Dict[str, Any]) -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    tmp = ITALK_DATA_PATH.with_suffix(ITALK_DATA_PATH.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(ITALK_DATA_PATH)


def _italk_find_fav(
    data: Dict[str, Any], fav_id: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    cats = data.get("favorites", {}).get("categories", {})
    for cat, items in cats.items():
        for it in items:
            if it.get("id") == fav_id:
                return cat, it
    return None, None


# ---------- Request models (match index.html JSON payloads) ----------


class SetVoiceReq(BaseModel):
    voice: str


class TagCreateReq(BaseModel):
    label: str
    text: str


class TagUpdateReq(BaseModel):
    label: str
    text: str


class FavoriteCreateReq(BaseModel):
    category: str
    text: str


class FavoriteUpdateReq(BaseModel):
    category: str
    text: str


class ReorderFavReq(BaseModel):
    category: str
    from_index: int = Field(alias="from")
    to_index: int = Field(alias="to")


class SessionNameReq(BaseModel):
    name: str = ""


class SessionSaveReq(BaseModel):
    name: str = ""


class AppendLineReq(BaseModel):
    text: str
    voice: str = ""  # present in client, not required for storage


class DeleteHistoryReq(BaseModel):
    id: str


class DeleteSessionLineReq(BaseModel):
    id: str


class DeleteHistoryLineReq(BaseModel):
    session_id: str
    line_id: str


@web_app.get("/italk/state")
def italk_state():
    with ITALK_LOCK:
        return _italk_load()


@web_app.post("/italk/settings/voice")
def italk_set_voice(req: SetVoiceReq):
    voice = (req.voice or "").strip()
    with ITALK_LOCK:
        data = _italk_load()
        data["settings"]["last_voice"] = voice
        _italk_save(data)
    return {"ok": True}


@web_app.post("/italk/tags")
def italk_add_tag(req: TagCreateReq):
    label = (req.label or "").strip()
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    tag_id = f"tag_{uuid.uuid4().hex[:8]}"
    with ITALK_LOCK:
        data = _italk_load()
        data["tags"].append(
            {
                "id": tag_id,
                "label": label or text[:12],
                "text": text,
                "created_at": _italk_now_iso(),
            }
        )
        _italk_save(data)
    return {"id": tag_id}


@web_app.put("/italk/tags/{tag_id}")
def italk_update_tag(tag_id: str, req: TagUpdateReq):
    label = (req.label or "").strip()
    text = (req.text or "").strip()
    with ITALK_LOCK:
        data = _italk_load()
        for t in data["tags"]:
            if t.get("id") == tag_id:
                if label:
                    t["label"] = label
                if text:
                    t["text"] = text
        _italk_save(data)
    return {"ok": True}


@web_app.delete("/italk/tags/{tag_id}")
def italk_delete_tag(tag_id: str):
    with ITALK_LOCK:
        data = _italk_load()
        data["tags"] = [t for t in data["tags"] if t.get("id") != tag_id]
        _italk_save(data)
    return {"ok": True}


@web_app.post("/italk/favorites")
def italk_add_fav(req: FavoriteCreateReq):
    category = (req.category or "Unsorted").strip() or "Unsorted"
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    fav_id = f"fav_{uuid.uuid4().hex[:8]}"
    with ITALK_LOCK:
        data = _italk_load()
        cats = data["favorites"]["categories"]
        cats.setdefault(category, []).append(
            {
                "id": fav_id,
                "text": text,
                "created_at": _italk_now_iso(),
                "updated_at": _italk_now_iso(),
            }
        )
        _italk_save(data)
    return {"id": fav_id}


@web_app.put("/italk/favorites/{fav_id}")
def italk_update_fav(fav_id: str, req: FavoriteUpdateReq):
    category = (req.category or "Unsorted").strip() or "Unsorted"
    text = (req.text or "").strip()
    with ITALK_LOCK:
        data = _italk_load()
        old_cat, fav = _italk_find_fav(data, fav_id)
        if fav is None or old_cat is None:
            raise HTTPException(status_code=404, detail="favorite not found")
        if old_cat != category:
            data["favorites"]["categories"][old_cat].remove(fav)
            data["favorites"]["categories"].setdefault(category, []).append(fav)
        if text:
            fav["text"] = text
            fav["updated_at"] = _italk_now_iso()
        _italk_save(data)
    return {"ok": True}


@web_app.delete("/italk/favorites/{fav_id}")
def italk_delete_fav(fav_id: str):
    with ITALK_LOCK:
        data = _italk_load()
        cat, fav = _italk_find_fav(data, fav_id)
        if fav and cat:
            data["favorites"]["categories"][cat].remove(fav)
            _italk_save(data)
    return {"ok": True}


@web_app.post("/italk/favorites/reorder")
def italk_reorder_fav(req: ReorderFavReq):
    with ITALK_LOCK:
        data = _italk_load()
        items = data["favorites"]["categories"].get(req.category, [])
        if not items:
            return {"ok": True}
        if req.from_index < 0 or req.from_index >= len(items):
            raise HTTPException(status_code=400, detail="from out of range")
        if req.to_index < 0 or req.to_index >= len(items):
            raise HTTPException(status_code=400, detail="to out of range")
        it = items.pop(req.from_index)
        items.insert(req.to_index, it)
        _italk_save(data)
    return {"ok": True}


@web_app.post("/italk/session/append_line")
def italk_append_line(req: AppendLineReq):
    text = (req.text or "").strip()
    if not text:
        return {"ok": True}


@web_app.post("/italk/session/delete_line")
def italk_delete_session_line(req: DeleteSessionLineReq):
    line_id = (req.id or "").strip()
    if not line_id:
        return {"ok": True}
    with ITALK_LOCK:
        data = _italk_load()
        before = len(data["last_session"]["lines"])
        data["last_session"]["lines"] = [
            l for l in data["last_session"]["lines"] if l.get("id") != line_id
        ]
        if len(data["last_session"]["lines"]) != before:
            _italk_save(data)
    return {"ok": True}


@web_app.post("/italk/history/delete_line")
def italk_delete_history_line(req: DeleteHistoryLineReq):
    session_id = (req.session_id or "").strip()
    line_id = (req.line_id or "").strip()
    if not session_id or not line_id:
        return {"ok": True}

    with ITALK_LOCK:
        data = _italk_load()
        changed = False
        for sess in data.get("history", []):
            if sess.get("id") == session_id:
                lines = sess.get("lines", [])
                new_lines = [l for l in lines if l.get("id") != line_id]
                if len(new_lines) != len(lines):
                    sess["lines"] = new_lines
                    changed = True
                break
        if changed:
            _italk_save(data)
    return {"ok": True}


@web_app.post("/italk/session/name")
def italk_set_session_name(req: SessionNameReq):
    name = (req.name or "").strip()
    with ITALK_LOCK:
        data = _italk_load()
        data["last_session"]["name"] = name
        _italk_save(data)
    return {"ok": True}


@web_app.post("/italk/session/new")
def italk_new_session():
    with ITALK_LOCK:
        data = _italk_load()
        data["last_session"] = {"name": "", "started_at": _italk_now_iso(), "lines": []}
        _italk_save(data)
    return {"ok": True}


@web_app.post("/italk/session/clear")
def italk_clear_session():
    return italk_new_session()


@web_app.post("/italk/session/save")
def italk_save_session(req: SessionSaveReq):
    """
    Save current session into history and clear last_session.
    Returns {"id": "<sess_id>"} so the client can auto-expand the saved session.
    """
    sess_id = ""
    with ITALK_LOCK:
        data = _italk_load()
        sess = data["last_session"]

        # Allow the client-provided name to override (but keep server as source of truth)
        name = (req.name or "").strip()
        if name:
            sess["name"] = name

        if sess.get("lines"):
            sess_id = f"sess_{uuid.uuid4().hex[:8]}"
            data["history"].insert(
                0,
                {
                    "id": sess_id,
                    "name": sess.get("name", ""),
                    "saved_at": _italk_now_iso(),
                    "lines": list(sess["lines"]),
                },
            )
            data["last_session"] = {"name": "", "started_at": _italk_now_iso(), "lines": []}
            _italk_save(data)

    return {"id": sess_id}


@web_app.post("/italk/history/delete")
def italk_delete_history(req: DeleteHistoryReq):
    with ITALK_LOCK:
        data = _italk_load()
        data["history"] = [h for h in data["history"] if h.get("id") != req.id]
        _italk_save(data)
    return {"ok": True}


@web_app.get("/")
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(static_path)


# hash of voice name -> safetensors Path
voices = {}
process_voices()
if voices:
    print("Warming up...")
    model_state = tts_model._cached_get_state_for_audio_prompt(next(iter(voices.values())))
    tts_model.generate_audio(model_state, "Hello, world.")
else:
    print("No voices found!")
    exit(1)

if __name__ == "__main__":
    web_app.mount("/", StaticFiles(directory="./static", html=True), name="static")
    uvicorn.run(
        "server:web_app", host="0.0.0.0", port=9800, reload=True, reload_includes="server.py"
    )
