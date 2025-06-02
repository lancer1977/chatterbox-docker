from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
from datetime import datetime

app = FastAPI()

# Read the device from an environment variable, fallback to 'cpu' if not set
DEVICE = os.getenv("CHATTERBOX_DEVICE", "cpu")
OUTPUT_DIR = "/app/completed"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    speaker: str = "default"
    filename: str  # Desired base name without extension

def build_output_path(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    full_name = f"{timestamp}_{safe_base_name}.wav"
    return os.path.join(OUTPUT_DIR, full_name)

@app.post("/ttsdisk")
async def generate_tts_disk(request: TTSRequest):
    model = ChatterboxTTS.from_pretrained(device=DEVICE)

    wav = model.generate(request.text)
    output_path = build_output_path(request.filename)
    ta.save(output_path, wav, model.sr)

    return {"audio_path": output_path}

@app.post("/tts")
async def generate_tts_stream(request: TTSRequest):
    model = ChatterboxTTS.from_pretrained(device=DEVICE)

    wav = model.generate(request.text)
    output_path = build_output_path(request.filename)
    ta.save(output_path, wav, model.sr)

    return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))
