import io
import os
import sys
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(Path(r"C:\Users\SISAJusvinJ\CA\agents\.env"))

text = " ".join(sys.argv[1:]).strip()
if not text:
    print("usage: say.py <text>")
    sys.exit(1)

client = InferenceClient(
    model=os.getenv("HF_TTS_MODEL", "hexgrad/Kokoro-82M"),
    provider=os.getenv("HF_TTS_PROVIDER", "deepinfra"),
    token=os.getenv("HF_TOKEN"),
    timeout=60,
)
audio_bytes = client.text_to_speech(text, extra_body={"voice": os.getenv("HF_TTS_VOICE", "af_heart")})
data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
mono = data.mean(axis=1)
print(f"SPEAKING ({len(mono) / sr:.1f}s): {text[:80]}", flush=True)
sd.play(mono, sr)
sd.wait()
print("DONE", flush=True)
