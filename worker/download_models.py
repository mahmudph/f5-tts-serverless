import torch
from f5_tts.api import F5TTS
import whisperx

device = "cpu"

print("[1/2] Downloading F5-TTS model (F5TTS_v1_Base) ...")
F5TTS(model="F5TTS_v1_Base", device=device)
print("[1/2] Done.")

print("[2/2] Downloading WhisperX alignment model (language=id) ...")
whisperx.load_align_model(language_code="id", device=device)
print("[2/2] Done.")
