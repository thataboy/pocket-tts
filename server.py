import io
import logging
import time
from os.path import getmtime
from pathlib import Path

import scipy.io.wavfile
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from pocket_tts import TTSModel
from pocket_tts.utils.utils import size_of_dict

logger = logging.getLogger(__name__)

VOICES_PATH = "./voices"

# Global model instance
tts_model = TTSModel.load_model()
voices = {}

web_app = FastAPI(
    title="Kyutai Pocket TTS API", description="Text-to-Speech generation API", version="1.0.0"
)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pod1-10007.internal.kyutai.org",
        "https://kyutai.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return {"ok": True}


class SynthesizeRequest(BaseModel):
    input: str = ""
    voice: str = ""


@web_app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    """
    Generate complete text in one go
    """
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if req.voice not in voices:
        raise HTTPException(status_code=400, detail="Invalid voice specified")

    t0 = time.perf_counter()

    # Use the appropriate model state
    audio_tensor = tts_model.generate_audio(
        voices[req.voice], req.input, frames_after_eos=2, copy_state=True
    )
    buffer = io.BytesIO()
    sample_rate = tts_model.sample_rate
    scipy.io.wavfile.write(buffer, sample_rate, audio_tensor.numpy())
    num_samples = audio_tensor.shape[-1]
    duration = num_samples / sample_rate
    elapsed = time.perf_counter() - t0
    rtf = elapsed / duration if duration > 0 else 0
    print(f"[{elapsed:.3f}s] len={len(req.input)} dur={duration:.2f}s rtf={rtf:.4f}")

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
            data = tts_model.get_state_for_audio_prompt(wav, truncate=True, export_path=sft)
        else:
            data = tts_model.get_state_for_audio_prompt(sft)
        voices[voice] = data

    print(f"{len(voices)} voices loaded; {size_of_dict(voices) // 1e6} MB.")


if __name__ == "__main__":
    process_voices()
    uvicorn.run(web_app, host="0.0.0.0", port=9800, reload=False)
