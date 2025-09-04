from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
import os
from datetime import datetime

app = FastAPI()

# Environment config
FOLDER = os.getenv("CHATTERBOX_FOLDER", "/Users/crichmond/chatterbox")
DEVICE = os.getenv("CHATTERBOX_DEVICE", "mps")
OUTPUT_DIR = FOLDER + "/completed"
DEBUG = os.getenv("DEBUG", "1").lower() in ("1", "true", "yes")
map_location = torch.device(DEVICE)
torch_load_original = torch.load

def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.load = patched_torch_load
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

@app.on_event("startup")
async def load_model_once():
    global tts_model
    log_debug("Loading TTS model at startup:" + DEVICE)
    tts_model = ChatterboxTTS.from_pretrained(device=DEVICE)
    log_debug("TTS model loaded and ready.")    

def getSpeakerFilePath(speaker: str) -> str:
    speaker_file = f"{speaker}.wav"
    speaker_path = os.path.join(FOLDER, "audio_prompts", speaker_file)
    log_debug(f"Speaker file path: {speaker_path}")
    return speaker_path

def build_output_path(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    full_name = f"{timestamp}_{safe_base_name}.wav"
    output_path = os.path.join(OUTPUT_DIR, full_name)
    log_debug(f"Output path built: {output_path}")
    return output_path



 

@app.post("/tts")
async def generate_tts_stream(request: TTSRequest):
    log_debug(f"Received TTS request (stream): {request}")
    if not request.text.strip():
        return {"error": "Text for TTS cannot be empty."}



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
