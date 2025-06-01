from fastapi import FastAPI
from pydantic import BaseModel
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os

app = FastAPI()

# Read the device from an environment variable, fallback to 'cpu' if not set
DEVICE = os.getenv("CHATTERBOX_DEVICE", "cpu")

class TTSRequest(BaseModel):
    text: str
    speaker: str = "default"

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    model = ChatterboxTTS.from_pretrained(device=DEVICE)

    text = request.text
    wav = model.generate(text)
    ta.save("test-1.wav", wav, model.sr)

    return {"audio_path": "test-1.wav"}
