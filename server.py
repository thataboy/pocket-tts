import io
import logging
import threading
import time
from os.path import getmtime
from pathlib import Path
from queue import Queue

import scipy.io.wavfile
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from pocket_tts import TTSModel
from pocket_tts.data.audio import stream_audio_chunks

logger = logging.getLogger(__name__)

VOICES_PATH = "./voices"

# Global model instance
tts_model = TTSModel.load_model(temp=0.9, lsd_decode_steps=1)

# hash of voice name -> safetensors Path
voices = {}

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
    voice_url: str | None = Form(None),
    voice_wav: UploadFile | None = File(None),
):
    return _stream(text, voice_url)


def _stream(text, voice):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if voice not in voices:
        voice = next(iter(voices.keys())) if len(voices) > 0 else ""
        if not voice:
            raise HTTPException(status_code=400, detail="No voice found")

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


@web_app.get("/")
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent / "pocket_tts" / "static" / "index.html"
    return FileResponse(static_path)


if __name__ == "__main__":
    process_voices()

    if voices:
        print("Warming up...")
        model_state = tts_model._cached_get_state_for_audio_prompt(next(iter(voices.values())))
        tts_model.generate_audio(model_state, "Hello, world.")

    uvicorn.run(web_app, host="0.0.0.0", port=9800, reload=False)
