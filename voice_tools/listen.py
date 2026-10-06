import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv(Path(r"C:\Users\SISAJusvinJ\CA\agents\.env"))

RATE = 16000
CHUNK = 1600  # 100 ms
MAX_SECONDS = float(os.getenv("LISTEN_MAX", "25"))
SILENCE_AFTER = float(os.getenv("LISTEN_SILENCE", "1.2"))
RMS_THRESH = float(os.getenv("LISTEN_RMS", "0.012"))

print("LISTENING... speak now.", flush=True)

audio = []
silence_time = 0.0
speech_started = False

with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=CHUNK) as stream:
    total = 0.0
    while total < MAX_SECONDS:
        block, _ier = stream.read(CHUNK)
        mono = block[:, 0]
        rms = float(np.sqrt(np.mean(mono * mono)))
        audio.append(mono.copy())
        total += CHUNK / RATE
        if rms > RMS_THRESH:
            speech_started = True
            silence_time = 0.0
        elif speech_started:
            silence_time += CHUNK / RATE
            if silence_time >= SILENCE_AFTER:
                break

samples = np.concatenate(audio) if audio else np.zeros(1, dtype="np.float32")

if not speech_started or len(samples) < RATE // 2:
    print("TRANSCRIPT: <no speech detected>")
    sys.exit(0)

from faster_whisper import WhisperModel

model = WhisperModel(
    os.getenv("WHISPER_MODEL", "base"),
    device="cpu",
    compute_type="int8",
)
segments, _ = model.transcribe(
    samples,
    language=os.getenv("WHISPER_LANGUAGE", "en"),
    beam_size=1,
    vad_filter=True,  # silero-based filtering inside whisper: kills hallucinations
    vad_parameters=dict(min_silence_duration_ms=400),
    condition_on_previous_text=False,
    no_speech_threshold=0.4,
    log_prob_threshold=-0.8,
    compression_ratio_threshold=2.2,
    temperature=0.0,  # float (no list) = no slow retry ladder
)
text = " ".join(s.text.strip() for s in segments).strip()
print(f"TRANSCRIPT: {text if text else '<empty>'}")
