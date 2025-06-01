from fastapi import FastAPI
from pydantic import BaseModel 
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

app = FastAPI()
#
class TTSRequest(BaseModel):
    text: str
    speaker: str = "default"
    
    
#RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: gpu

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    model = ChatterboxTTS.from_pretrained(device="cuda")
    #model = ChatterboxTTS.from_pretrained(device="cuda")
    text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    wav = model.generate(text)
    ta.save("test-1.wav", wav, model.sr)

    # If you want to synthesize with a different voice, specify the audio prompt
    #AUDIO_PROMPT_PATH=request.speaker+".wav"
    #wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    #ta.save("test-2.wav", wav, model.sr)
    return {"audio_path": "test-1.wav"}


