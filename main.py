from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
from datetime import datetime

app = FastAPI()

# Environment config
DEVICE = os.getenv("CHATTERBOX_DEVICE", "cpu")
OUTPUT_DIR = "/app/completed"
DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model placeholder
tts_model = None

def log_debug(message: str):
    if DEBUG:
        print(f"[DEBUG] {message}")

class TTSRequest(BaseModel):
    text: str = ""
    speaker: str = "default"
    filename: str = ""  # Desired base name without extension

def getSpeakerFilePath(speaker: str) -> str:
    speaker_file = f"{speaker}.wav"
    speaker_path = os.path.join("/app/audio_prompts", speaker_file)
    log_debug(f"Speaker file path: {speaker_path}")
    return speaker_path

def build_output_path(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    full_name = f"{timestamp}_{safe_base_name}.wav"
    output_path = os.path.join(OUTPUT_DIR, full_name)
    log_debug(f"Output path built: {output_path}")
    return output_path

@app.on_event("startup")
async def load_model_once():
    global tts_model
    log_debug("Loading TTS model at startup...")
    tts_model = ChatterboxTTS.from_pretrained(device=DEVICE)
    log_debug("TTS model loaded and ready.")

@app.post("/ttsdisk")
async def generate_tts_disk(request: TTSRequest):
    log_debug(f"Received TTS request (disk): {request}")
    wav = tts_model.generate(request.text)
    log_debug("Audio generated.")

    output_path = build_output_path(request.filename)
    ta.save(output_path, wav, tts_model.sr)
    log_debug(f"Audio saved to {output_path}")

    return {"audio_path": output_path}

@app.post("/tts")
async def generate_tts_stream(request: TTSRequest):
    log_debug(f"Received TTS request (stream): {request}")

    if not request.speaker:
        wav = tts_model.generate(request.text)
    else:
        speaker_path = getSpeakerFilePath(request.speaker)
        wav = tts_model.generate(request.text, audio_prompt_path=speaker_path)
    log_debug("Audio generated.")

    output_path = build_output_path(request.filename)
    ta.save(output_path, wav, tts_model.sr)
    log_debug(f"Audio saved to {output_path}")

    return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))
