import io
import logging
import os
import scipy.io.wavfile
import threading
import time
import uvicorn

from os.path import getmtime
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from html import escape
from pydantic import BaseModel
from urllib.parse import quote

from italk.italk import router as italk_router  # iTalk logic module
from pocket_tts import TTSModel, export_model_state
from pocket_tts.data.audio import stream_audio_chunks

logger = logging.getLogger(__name__)

VOICES_PATH = "./voices"



class StaticFilesEx(StaticFiles):
    async def get_response(self, path: str, scope):
        full_path, stat_result = self.lookup_path(path)

        if stat_result and os.path.isdir(full_path):
            # Serve index.html if exists
            index_path = os.path.join(full_path, "index.html")
            if os.path.exists(index_path):
                return await super().get_response(
                    os.path.join(path, "index.html"), scope
                )

            entries = os.listdir(full_path)

            # Sort: directories first, then files
            dirs = []
            files = []

            for name in entries:
                abs_entry = os.path.join(full_path, name)
                if os.path.isdir(abs_entry):
                    dirs.append(name)
                else:
                    files.append(name)

            dirs.sort()
            files.sort()

            items = []

            # Parent link stays relative
            if path not in ("", "/"):
                items.append('<li><a href="../">.. (parent directory)</a></li>')

            # Directories
            for name in dirs:
                href = quote(name) + "/"
                items.append(f'<li><a href="{href}">{escape(name)}/</a></li>')

            # Files
            for name in files:
                href = quote(name)
                items.append(f'<li><a href="{href}">{escape(name)}</a></li>')

            html = f"""
            <html>
                <head>
                    <title>Index of /{escape(path)}</title>
                    <style>li {{ margin: 1em 1em; }}</style>
                </head>
                <body>
                    <h2>Index of /{escape(path)}</h2>
                    <ul>
                        {''.join(items)}
                    </ul>
                </body>
            </html>
            """

            return HTMLResponse(content=html)

        return await super().get_response(path, scope)

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)


@web_app.get("/voices")
def list_voices():
    """Return list of available voice style names."""
    return {"voices": sorted(list(voices.keys()))}


@web_app.get("/voices/refresh")
def refresh_voices():
    process_voices()
    return {"voices": sorted(list(voices.keys()))}


class SynthesizeRequest(BaseModel):
    text: str = ""
    voice: str = ""


@web_app.post("/synthesize")
@web_app.post("/v1/audio/speech")
def synthesize(req: SynthesizeRequest):
    """Generate complete text in one go"""
    if not req.text.strip():
        raise HTTPException(status_code=406, detail="Text cannot be empty")

    if req.voice not in voices:
        req.voice = next(iter(voices.keys())) if len(voices) > 0 else ""
        if not req.voice:
            raise HTTPException(status_code=407, detail="No voice found")

    print(f"{req.voice}➡️{req.text}⬅️")
    t0 = time.perf_counter()

    model_state = tts_model._cached_get_state_for_audio_prompt(voices[req.voice])
    audio = tts_model.generate_audio(model_state, req.text, frames_after_eos=2)

    sample_rate = tts_model.sample_rate
    audio_i16 = (audio.numpy().clip(-1, 1) * 32767).astype("int16")
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, sample_rate, audio_i16)
    elapsed = time.perf_counter() - t0
    num_samples = audio.shape[-1]
    duration = num_samples / sample_rate
    spd = duration / elapsed
    print(f"[{elapsed:.3f}s] len={len(req.text)} dur={duration:.2f}s  {spd:.3f}x")

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
            model_state = tts_model.get_state_for_audio_prompt(
                audio_conditioning=wav, truncate=True
            )
            export_model_state(model_state, sft)
        voices[voice] = sft
    print(f"{len(voices)} voices loaded")


async def generate_data_stream(
    text_to_generate: str, model_state: dict, request: Request | None = None
):
    queue: Queue[bytes | None] = Queue()
    cancel = threading.Event()
    t0 = time.perf_counter()

    thread = threading.Thread(
        target=write_to_queue, args=(queue, text_to_generate, model_state, cancel), daemon=True
    )
    thread.start()

    i = 0
    try:
        while True:
            if request is not None and await request.is_disconnected():
                cancel.set()
                break
            try:
                data = queue.get(timeout=0.1)
            except Exception:
                continue
            if data is None:
                break
            i += 1
            yield data
    finally:
        cancel.set()
        elapsed = time.perf_counter() - t0
        print(f"Total time: {elapsed:.3f}s | Chunks: {i}")
        thread.join(timeout=0.2)


def write_to_queue(queue: Queue, text_to_generate: str, model_state: dict, cancel: threading.Event):
    class FileLikeToQueue(io.IOBase):
        def __init__(self, queue: Queue, cancel: threading.Event):
            self.queue = queue
            self.cancel = cancel

        def write(self, data):
            if not self.cancel.is_set():
                self.queue.put(data)

        def flush(self):
            pass

        def close(self):
            self.queue.put(None)

    def cancellable_chunks(chunks_iter):
        for ch in chunks_iter:
            if cancel.is_set():
                break
            yield ch

    try:
        audio_chunks = tts_model.generate_audio_stream(
            model_state=model_state, text_to_generate=text_to_generate
        )
        stream_audio_chunks(
            FileLikeToQueue(queue, cancel),
            cancellable_chunks(audio_chunks),
            tts_model.config.mimi.sample_rate,
        )
    except Exception as e:
        logger.exception("stream generation failed: %s", e)
        queue.put(None)


@web_app.post("/stream")
def stream(req: SynthesizeRequest, request: Request):
    return _stream(req.text, req.voice, request)


@web_app.post("/tts")
def text_to_speech(
    request: Request,
    text: str = Form(...),
    voice: str | None = Form(None),
    voice_url: str | None = Form(None),
):
    return _stream(text, voice or voice_url, request)


def _stream(text, voice, request: Request | None = None):
    if not text.strip():
        raise HTTPException(status_code=401, detail="Text cannot be empty")
    if voice not in voices:
        voice = next(iter(voices.keys())) if len(voices) > 0 else ""
        if not voice:
            raise HTTPException(status_code=402, detail="No voice found")

    print(f"stream [{len(text)}]➡️{voice}➡️{text if len(text) < 200 else text[:200] + '...'}⬅️")
    model_state = tts_model._cached_get_state_for_audio_prompt(voices[voice])

    return StreamingResponse(
        generate_data_stream(text, model_state, request),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


# --- App Routing ---

# Mount the iTalk backend logic at /italk prefix
web_app.include_router(italk_router, prefix="/italk")


@web_app.get("/")
@web_app.get("/index.html")
async def root():
    """Default: Serve iTalk index"""
    return FileResponse(Path(__file__).parent / "italk" / "index.html")


@web_app.get("/demo")
async def demo_page():
    """Serve demo page"""
    return FileResponse(Path(__file__).parent / "static" / "demo.html")


web_app.mount("/books", StaticFilesEx(directory="/Volumes/T7/books", html=True), name="books")
web_app.mount("/", StaticFilesEx(directory="./italk", html=True), name="italk")

@web_app.on_event("startup")
def startup():
    global voices, tts_model
    tts_model = TTSModel.load_model(temp=0.9, lsd_decode_steps=1)
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
    uvicorn.run(
        "server:web_app", host="0.0.0.0", port=9800, reload=False, reload_includes="./server.py",
    )
