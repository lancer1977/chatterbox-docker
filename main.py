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
SKIP_MODEL_LOAD = os.getenv("CHATTERBOX_SKIP_MODEL_LOAD", "0").lower() in ("1", "true", "yes")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model placeholder
tts_model = None

def log_debug(message: str):
    if DEBUG:
        print(f"[DEBUG] {message}")
# Try lower cfg_weight values (e.g. ~0.3) and increase exaggeration to around 0.7 or higher.
# Higher exaggeration tends to speed up speech; reducing cfg_weight helps compensate with slower, more deliberate pacing.
class TTSRequest(BaseModel):
    text: str = ""
    speaker: str = "default"
    filename: str = ""  # Desired base name without extension
    exaggeration: float = .5  # Exaggeration factor for the TTS model
    cfg_weight: float = 0.5  # Configuration weight for the TTS model
    temperature: float = 0.8  # Temperature for the TTS model
    def __str__(self):
        return f"TTSRequest(text={self.text}, speaker={self.speaker}, filename={self.filename}, exaggeration={self.exaggeration})"
    

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
    if SKIP_MODEL_LOAD:
        log_debug("Skipping TTS model load for smoke/health checks.")
        return
    log_debug("Loading TTS model at startup...")
    tts_model = ChatterboxTTS.from_pretrained(device=DEVICE)
    log_debug("TTS model loaded and ready.")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "debug": DEBUG,
        "model_loaded": tts_model is not None,
        "skip_model_load": SKIP_MODEL_LOAD,
    }

@app.get("/")
async def root():
    return {"name": "chatterbox-docker", "health": "/health", "tts": "/tts"}

@app.post("/tts")
async def generate_tts_stream(request: TTSRequest):
    log_debug(f"Received TTS request (stream): {request}")
    if not request.text.strip():
        return {"error": "Text for TTS cannot be empty."}

    if tts_model is None:
        return {"error": "TTS model is not loaded."}

    if not request.speaker:
        wav = tts_model.generate(request.text)
    else:
        speaker_path = getSpeakerFilePath(request.speaker)
        if not os.path.isfile(speaker_path):
            return {"error": f"Speaker file not found: {speaker_path}"}
        wav = tts_model.generate(request.text, audio_prompt_path=speaker_path, exaggeration=request.exaggeration, cfg_weight=request.cfg_weight, temperature=request.temperature)
    log_debug("Audio generated.")

    output_path = build_output_path(request.filename)
    ta.save(output_path, wav, tts_model.sr)
    log_debug(f"Audio saved to {output_path}")

    return FileResponse(output_path, media_type="audio/wav", filename=os.path.basename(output_path))
