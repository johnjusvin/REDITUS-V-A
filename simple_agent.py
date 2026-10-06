"""
Reditus Labs Voice Agent — fully self-hosted, open-source stack
===============================================================
LLM : OmniRouter (OpenAI-compatible endpoint, env-selectable model)
STT : sherpa-onnx Parakeet-TDT-0.6B-v2 int8 (primary)
      faster-whisper distil-large-v3 int8 (fallback, STT_BACKEND=whisper)
TTS : kokoro-onnx in-process (fp32 ONNX — measured 11x faster than int8 on
      non-VNNI CPUs), sentence-chunked so playback starts after sentence 1
VAD : Silero (local)          EOU: livekit.agents.inference.TurnDetector (local v1-mini)
No LiveKit Cloud inference is used anywhere in this file.

Measured on the dev host (i5-10210U, 4c/8t, CPU-only):
  Parakeet STT : RTF 0.108  (~0.8s for 7.4s of audio, 4 threads)
  Kokoro TTS   : ~1.2s for a 2s sentence, ~2.1s for the 3.6s greeting (fp32)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    Agent,
    AgentSession,
    APIConnectOptions,
    APIConnectionError,
    RunContext,
    stt,
    tts,
    utils,
)
from livekit.agents.llm import ChatMessage, StopResponse, function_tool
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.plugins import openai, silero

# Local audio end-of-turn detector: turn-detector-v1-mini runs fully in-process
# via livekit.local_inference (native model, ~108MB, no cloud call). Pinned to
# v1-mini so console/dev mode never resolves to the cloud v1 detector.
from livekit.agents.inference import TurnDetector

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("reditus-agent")

BASE_DIR = Path(__file__).parent

DEFAULT_PARAKEET_DIR = BASE_DIR / "models" / "parakeet" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
DEFAULT_KOKORO_MODEL = BASE_DIR / "models" / "kokoro" / "kokoro-v1.0.onnx"
DEFAULT_KOKORO_VOICES = BASE_DIR / "models" / "kokoro" / "voices-v1.0.bin"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set. Add it to the .env file next to this script "
            f"({BASE_DIR / '.env'})."
        )
    return value


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a valid number.") from None


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a valid integer.") from None


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_path(name: str, default: Path) -> Path:
    raw = (os.getenv(name) or "").strip()
    return Path(raw) if raw else default


_LOOP_STALL_WARNING = "event loop blocked"
_LOOP_STALL_COUNT = 0


class _LoopStallCounter(logging.Handler):
    """Counts 'event loop blocked' warnings from livekit.agents."""

    def emit(self, record: logging.LogRecord) -> None:
        global _LOOP_STALL_COUNT
        if _LOOP_STALL_WARNING in record.getMessage():
            _LOOP_STALL_COUNT += 1


def _configure_logging() -> None:
    level_name = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    # DEBUG for livekit.agents only when explicitly enabled
    agents_level = (
        logging.DEBUG if _env_bool("LIVEKIT_AGENTS_DEBUG", False) else logging.INFO
    )
    logging.getLogger("livekit.agents").setLevel(agents_level)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("livekit.agents").addHandler(_LoopStallCounter())


def validate_environment() -> None:
    """Fail fast with a clear message instead of dying mid-call."""
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        _require_env(name)
    _require_env("OMNIROUTER_BASE_URL")
    _require_env("OMNIROUTER_API_KEY")

    backend = (os.getenv("STT_BACKEND") or "parakeet").strip().lower()
    if backend not in ("parakeet", "whisper"):
        raise ConfigError(f"STT_BACKEND={backend!r} — must be 'parakeet' or 'whisper'.")

    if backend == "parakeet":
        pdir = _env_path("PARAKEET_MODEL_DIR", DEFAULT_PARAKEET_DIR)
        for fname in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
            if not (pdir / fname).is_file():
                raise ConfigError(f"Parakeet model file missing: {pdir / fname}")

    kokoro_model = _env_path("KOKORO_MODEL_PATH", DEFAULT_KOKORO_MODEL)
    kokoro_voices = _env_path("KOKORO_VOICES_PATH", DEFAULT_KOKORO_VOICES)
    if not kokoro_model.is_file():
        raise ConfigError(f"Kokoro model missing: {kokoro_model}")
    if not kokoro_voices.is_file():
        raise ConfigError(f"Kokoro voices missing: {kokoro_voices}")


# ---------------------------------------------------------------------------
# Shared audio helpers
# ---------------------------------------------------------------------------
def _to_16k_mono(buffer: AudioBuffer) -> np.ndarray:
    """Combine frames, mix to mono and resample to 16 kHz float32 with soxr
    (via rtc.AudioResampler) — no linear-interpolation artefacts."""
    frame = rtc.combine_audio_frames(buffer)
    if frame.sample_rate == 16000 and frame.num_channels == 1:
        return np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0

    resampler = rtc.AudioResampler(frame.sample_rate, 16000, num_channels=frame.num_channels)
    out_frames = resampler.push(frame) + resampler.flush()
    if not out_frames:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in out_frames])
    if frame.num_channels > 1:
        audio = audio.reshape(-1, frame.num_channels).mean(axis=1)
    return audio.astype(np.float32) / 32768.0


def _speech_event(lang: str, text: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=utils.shortuuid("stt-"),
        alternatives=[stt.SpeechData(language=lang, text=text)],
    )


# ---------------------------------------------------------------------------
# STT — primary: sherpa-onnx Parakeet-TDT-0.6B-v2 (int8)
# ---------------------------------------------------------------------------
class ParakeetSTT(stt.STT):
    """Non-streaming offline recognizer; the AgentSession feeds it complete
    VAD-segmented utterances. One recognizer instance per worker process."""

    def __init__(self, *, model_dir: Path, num_threads: int = 4):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        import sherpa_onnx

        self._hotwords_path: str | None = None
        hotwords_env = os.getenv("PARAKEET_HOTWORDS_FILE")
        if hotwords_env is not None:
            # explicit set (blank = disabled)
            hotwords_path = Path(hotwords_env.strip()) if hotwords_env.strip() else None
        else:
            hotwords_path = DEFAULT_PARAKEET_DIR / "hotwords.txt"
        if hotwords_path is not None:
            if not hotwords_path.is_file() and hotwords_path.name == "hotwords.txt":
                try:
                    hotwords_path.write_text(
                        "Reditus Labs:1.5\nReditus Loop:1.5\nReditus:1.5\n",
                        encoding="utf-8",
                    )
                except OSError:
                    hotwords_path = None
            if hotwords_path is not None and hotwords_path.is_file():
                self._hotwords_path = str(hotwords_path)
        logger.info(
            "Loading Parakeet TDT 0.6B v2 int8 from %s (threads=%d) [pid=%d]%s",
            model_dir, num_threads, os.getpid(),
            " with hotwords" if self._hotwords_path else "",
        )
        t0 = time.perf_counter()
        extra: dict[str, object] = {}
        decoding_method = "greedy_search"
        if self._hotwords_path:
            decoding_method = "modified_beam_search"
            extra = {
                "max_active_paths": 4,
                "hotwords_file": self._hotwords_path,
                "hotwords_score": 1.5,
                "modeling_unit": "bpe",
                "bpe_vocab": str(model_dir / "tokens.txt"),
            }
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=str(model_dir / "tokens.txt"),
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            joiner=str(model_dir / "joiner.int8.onnx"),
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method=decoding_method,
            model_type="nemo_transducer",
            provider="cpu",
            debug=False,
            **extra,
        )
        logger.info("Parakeet loaded in %.1fs [pid=%d]", time.perf_counter() - t0, os.getpid())

    @property
    def model(self) -> str:
        return "parakeet-tdt-0.6b-v2-int8"

    @property
    def provider(self) -> str:
        return "sherpa-onnx"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        def _process() -> tuple[str, float, float]:
            t0 = time.perf_counter()
            audio = _to_16k_mono(buffer)
            audio_s = len(audio) / 16000
            if audio_s < 0.1:
                return "", time.perf_counter() - t0, audio_s
            if audio_s > 15:
                logger.warning(
                    "STT chunk too long: %.1fs (VAD max_buffered_speech=15s)", audio_s
                )
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            self._recognizer.decode_stream(stream)
            return stream.result.text.strip(), time.perf_counter() - t0, audio_s

        text, dt, audio_s = await asyncio.to_thread(_process)
        if audio_s < 0.1:
            return _speech_event("en", "")
        text = _apply_transcript_replacements(text)
        logger.info("STT[parakeet]: %.0f ms for %.1fs audio", dt * 1000, audio_s)
        return _speech_event("en", text)


# ---------------------------------------------------------------------------
# STT — fallback: faster-whisper distil-large-v3 (int8, CPU)
# ---------------------------------------------------------------------------
class WhisperFallbackSTT(stt.STT):
    """Fallback backend. distil-large-v3 int8 — never large-v3 on CPU.
    No vad_filter: audio arrives VAD-segmented already."""

    def __init__(
        self,
        *,
        model_size: str = "distil-large-v3",
        cpu_threads: int = 4,
        beam_size: int = 1,
        initial_prompt: str = "Reditus Labs, Reditus Loop, vibe-coded",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper %s (cpu, int8, beam=%d, threads=%d) [pid=%d]",
            model_size, beam_size, cpu_threads, os.getpid(),
        )
        self._model = WhisperModel(
            model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads
        )
        self._model_size = model_size
        self._beam_size = beam_size
        self._initial_prompt = initial_prompt

    @property
    def model(self) -> str:
        return self._model_size

    @property
    def provider(self) -> str:
        return "faster-whisper"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        def _process() -> tuple[str, float, float]:
            t0 = time.perf_counter()
            audio = _to_16k_mono(buffer)
            audio_s = len(audio) / 16000
            if audio_s < 0.1:
                return "", time.perf_counter() - t0, audio_s
            if audio_s > 15:
                logger.warning(
                    "STT chunk too long: %.1fs (VAD max_buffered_speech=15s)", audio_s
                )
            segments, _info = self._model.transcribe(
                audio,
                language="en",
                beam_size=self._beam_size,
                condition_on_previous_text=False,
                initial_prompt=self._initial_prompt,
                temperature=0.0,  # float (no list) = no slow retry ladder
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            return text, time.perf_counter() - t0, audio_s

        text, dt, audio_s = await asyncio.to_thread(_process)
        if audio_s < 0.1:
            return _speech_event("en", "")
        text = _apply_transcript_replacements(text)
        logger.info("STT[whisper]: %.0f ms for %.1fs audio", dt * 1000, audio_s)
        return _speech_event("en", text)


def _build_stt() -> stt.STT:
    backend = (os.getenv("STT_BACKEND") or "parakeet").strip().lower()
    cores = os.cpu_count() or 4
    if backend == "whisper":
        return WhisperFallbackSTT(
            model_size=os.getenv("WHISPER_MODEL", "distil-large-v3"),
            cpu_threads=_env_int("STT_THREADS", max(2, cores - 2)),
        )
    return ParakeetSTT(
        model_dir=_env_path("PARAKEET_MODEL_DIR", DEFAULT_PARAKEET_DIR),
        num_threads=_env_int("STT_THREADS", max(2, cores - 2)),
    )


# ---------------------------------------------------------------------------
# TTS — kokoro-onnx, in-process, sentence-chunked
# ---------------------------------------------------------------------------
KOKORO_SAMPLE_RATE = 24000

_MD_NOISE = re.compile(r"[*_#`~>|]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[\"'“”‘’(\w])")


def _split_sentences(text: str) -> list[str]:
    text = _MD_NOISE.sub(" ", text).strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    merged: list[str] = []
    for part in parts:
        # keep tiny fragments ("Yes.", "Right?") with the previous sentence so
        # each synthesis call is worth its fixed overhead
        if merged and len(part) < 24:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged or [text]


# exact-text phrases synthesized once in prewarm and served from RAM
_CACHED_PHRASES: list[str] = [
    "Welcome to Reditus Labs! I'm your virtual assistant. What can I do for you today?",
    "Could I get your name?",
    "What's the best email or phone number?",
    "Could you spell that email for me?",
    "Are you still there? I can tell you about our services or book a consultation.",
    "It seems you had to go. Thank you for calling Reditus Labs. Goodbye!",
    "Sorry, my voice glitched for a second. Could you repeat that?",
]

_PHRASE_CACHE: dict[str, bytes] = {}


def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _phrase_cache_get(text: str) -> bytes | None:
    return _PHRASE_CACHE.get(_normalize_phrase(text))


def _extra_cached_phrases_from_env() -> list[str]:
    raw = (os.getenv("CACHED_TTS_PHRASES") or "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split("|") if p.strip()]


def _prime_phrase_cache(kokoro: "KokoroLocalTTS") -> None:
    import numpy as _np

    for phrase in _CACHED_PHRASES + _extra_cached_phrases_from_env():
        try:
            samples, _sr = kokoro._kokoro.create(
                phrase,
                voice=kokoro._voice,
                speed=kokoro._speed,
                lang=kokoro._lang,
            )
        except Exception as e:
            logger.warning("phrase cache: could not synthesize %r: %s", phrase[:40], e)
            continue
        pcm16 = (_np.clip(samples, -1.0, 1.0) * 32767).astype(_np.int16).tobytes()
        _PHRASE_CACHE[_normalize_phrase(phrase)] = pcm16
    logger.info("TTS phrase cache primed with %d phrases", len(_PHRASE_CACHE))


def _first_chunk_units(sentences: list[str]) -> list[str]:
    """Split the speech into emittable pieces so playback starts fast: the
    opening sentence is cut at its first comma (or dash), or after ~8 words.
    """
    if not sentences:
        return []
    units: list[str] = []
    first = sentences[0]
    comma = re.search(r"[,;—–]", first)
    words = first.split()
    if comma and len(first[: comma.start()].split()) <= 8:
        head = first[: comma.start() + 1].strip()
        tail = first[comma.start() + 1 :].strip(" ,")
    else:
        # ~3-4 words is the fastest first playback: one small create_stream call
        if len(words) > 4:
            head, tail = " ".join(words[:4]), " ".join(words[4:])
        else:
            head, tail = first, ""
    units.append(head)
    if tail:
        units.append(tail)
    units.extend(sentences[1:])
    return units


class KokoroLocalTTS(tts.TTS):
    """Local Kokoro-82M via kokoro-onnx. fp32 ONNX by default: int8 is ~11x
    SLOWER on CPUs without AVX512-VNNI (measured 13.3s vs 1.2s for the same
    2s utterance on an i5-10210U)."""

    def __init__(
        self,
        *,
        model_path: Path,
        voices_path: Path,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
        num_threads: int = 2,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=KOKORO_SAMPLE_RATE,
            num_channels=1,
        )
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads  # capped: leave cores for STT/VAD
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        logger.info(
            "Loading Kokoro ONNX %s (voice=%s, threads=%d) [pid=%d]",
            model_path.name, voice, num_threads, os.getpid(),
        )
        t0 = time.perf_counter()
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._kokoro = Kokoro.from_session(session, str(voices_path))
        self._voice = voice
        self._speed = speed
        self._lang = lang
        self._model_name = model_path.name
        # warm + measure RTF so we can warn when the CPU is too slow for realtime
        t1 = time.perf_counter()
        warm_samples, warm_sr = self._kokoro.create(
            "Reditus Labs builds digital products, intelligent systems, and voice experiences.",
            voice=voice, speed=speed, lang=lang,
        )
        warm_dt = time.perf_counter() - t1
        rtf = warm_dt / (len(warm_samples) / warm_sr) if len(warm_samples) else float("inf")
        logger.info(
            "Kokoro loaded+warmed in %.1fs; warmup RTF=%.2f (%.1fs audio in %.1fs) [pid=%d]",
            time.perf_counter() - t0, rtf, len(warm_samples) / warm_sr, warm_dt, os.getpid(),
        )
        if rtf > 0.5:
            logger.warning(
                "Kokoro RTF=%.2f (>0.5): realtime voice playback will feel sluggish on this CPU.",
                rtf,
            )

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "kokoro-onnx"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "KokoroLocalChunkedStream":
        return KokoroLocalChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class KokoroLocalChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        kokoro: KokoroLocalTTS = self._tts  # type: ignore[assignment]
        sentences = _split_sentences(self._input_text)
        if not sentences:
            return

        # a fully cached phrase skips synthesis entirely
        cached_pcm = _phrase_cache_get(self._input_text)
        if cached_pcm is not None:
            logger.info("TTS[kokoro]: cache hit for %r", self._input_text[:48])
            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=KOKORO_SAMPLE_RATE,
                num_channels=1,
                mime_type="audio/pcm",
            )
            output_emitter.push(cached_pcm)
            output_emitter.flush()
            return

        units = _first_chunk_units(sentences)

        started = False
        try:
            for i, sentence in enumerate(units):
                t0 = time.perf_counter()
                first_chunk_at: float | None = None
                unit_samples = 0
                unit_sr = KOKORO_SAMPLE_RATE
                async for samples, sr in kokoro._kokoro.create_stream(
                    sentence,
                    voice=kokoro._voice,
                    speed=kokoro._speed,
                    lang=kokoro._lang,
                ):
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter() - t0
                        logger.info(
                            "TTS[kokoro]: first chunk for unit %d/%d arrived after %.0f ms",
                            i + 1, len(units), first_chunk_at * 1000,
                        )
                    unit_samples += len(samples)
                    unit_sr = sr
                    if not started:
                        output_emitter.initialize(
                            request_id=utils.shortuuid(),
                            sample_rate=KOKORO_SAMPLE_RATE,
                            num_channels=1,
                            mime_type="audio/pcm",
                        )
                        started = True
                    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                    output_emitter.push(pcm16.tobytes())

                total = time.perf_counter() - t0
                logger.info(
                    "TTS[kokoro]: unit %d/%d, %.1fs audio in %.0f ms (first chunk %s)",
                    i + 1, len(units), unit_samples / unit_sr, total * 1000,
                    _ms(first_chunk_at),
                )
        except Exception as e:
            raise APIConnectionError(f"Kokoro synthesis failed: {e}", retryable=True) from e

        if started:
            output_emitter.flush()


# ---------------------------------------------------------------------------
# Per-turn latency metrics
# ---------------------------------------------------------------------------
def _log_turn_metrics(item) -> None:
    """One line per turn from ChatMessage.metrics (the 1.8.4 non-deprecated
    path; session 'metrics_collected' still works but logs a deprecation)."""
    if not isinstance(item, ChatMessage):
        return
    m = item.metrics or {}
    if item.role == "user":
        if "transcription_delay" in m or "end_of_turn_delay" in m:
            logger.info(
                "TURN[user] eou_delay=%s transcription_delay=%s",
                _ms(m.get("end_of_turn_delay")),
                _ms(m.get("transcription_delay")),
            )
    elif item.role == "assistant":
        if any(k in m for k in ("e2e_latency", "llm_node_ttft", "tts_node_ttfb")):
            logger.info(
                "TURN[agent] e2e=%s llm_ttft=%s tts_ttfb=%s playback=%s interrupted=%s",
                _ms(m.get("e2e_latency")),
                _ms(m.get("llm_node_ttft")),
                _ms(m.get("tts_node_ttfb")),
                _ms(m.get("playback_latency")),
                item.interrupted,
            )


def _ms(seconds: float | None) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds is not None else "-"


_FILLER_WORDS_DEFAULT = "uh,um,okay,yeah,hmm"

# set by the agent_state_changed listener in entrypoint
_LAST_AGENT_SPEECH_END_TS: float | None = None

_REPLACEMENTS_DEFAULT = "Renders->Reditus,Reditas->Reditus,Radius->Reditus"


def _apply_transcript_replacements(text: str) -> str:
    """Map common speech-recognition mishearings of 'Reditus' to the right name."""
    raw = os.getenv("TRANSCRIPT_REPLACEMENTS", _REPLACEMENTS_DEFAULT)
    for pair in raw.split(","):
        if "->" not in pair:
            continue
        src, _, dst = pair.partition("->")
        src, dst = src.strip(), dst.strip()
        if not src or not dst:
            continue
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
    return text


def _is_filler_only(text: str) -> bool:
    """True when every token in the transcript is a bare filler word."""
    raw = os.getenv("FILLER_WORDS", _FILLER_WORDS_DEFAULT)
    fillers = {w.strip().lower() for w in raw.split(",") if w.strip()}
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return bool(tokens) and all(tok in fillers for tok in tokens)


# ---------------------------------------------------------------------------
# The assistant
# ---------------------------------------------------------------------------
SERVICES_CATALOG = """What Reditus Labs offers:
1. Digital Products — design and build of web and mobile products, idea to launch.
2. Intelligent Systems — AI systems, agents and automation, including voice agents.
3. Software Engineering — custom software delivered with the Reditus Loop SDLC.
4. Technology Experiences — creative interactive experiences for brands and teams.
5. Project Rescue — taking over broken or vibe-coded projects, stabilizing them and shipping."""

INSTRUCTIONS = f"""You are the voice assistant of Reditus Labs, a Creative Engineering Lab.

{SERVICES_CATALOG}

Rules for every reply:
- At most 2 short sentences and roughly 25 words. This is a voice call.
- Plain spoken English only. No markdown, bullet lists, or emojis.
- When asked what services you offer, name the five services in one sentence
  with no descriptions, then ask which one they care about.
- Transcripts may mishear "Reditus" (for example "Renders", "Reditas",
  "Radius"). Always treat them as Reditus Labs.
- If asked about things outside Reditus Labs (prices, availability, other
  companies), say you can only help with Reditus Labs topics and offer the
  closest thing you can do.
- Only state facts from this description or your tools. If you don't know,
  say so and offer a consultation.

Consultation flow — collect ONE field at a time, in this order:
1. name, 2. contact (email or phone; if email, ask them to spell it),
3. a one-sentence project summary, 4. optionally which service.
Then read the contact details back and ask for confirmation. Only after an
explicit yes, call request_consultation. Never invent or assume a field.

When the user says goodbye or thanks to end the call, respond with a single
warm goodbye."""


class ReditusAssistant(Agent):
    def __init__(self):
        super().__init__(instructions=INSTRUCTIONS)
        self.requests_file = Path(
            os.getenv("CONSULTATION_LOG", str(BASE_DIR / "consultation_requests.jsonl"))
        )
        # LLM_BYPASS=1: skip the model call and answer from canned text. Lets you
        # measure the EOU -> STT -> TTS path without any LLM latency.
        self.llm_bypass = _env_bool("LLM_BYPASS", False)
        self._bypass_turns = 0

    _BYPASS_REPLIES = (
        "Got it. Reditus Labs builds digital products, intelligent systems, and software.",
        "Thanks. Which of those would you like to talk about?",
        "Could I get your name?",
        "Noted. I can set up a consultation with our team about that.",
    )

    async def on_user_turn_completed(self, turn_ctx, new_message: ChatMessage) -> None:
        if self.llm_bypass:
            idx = self._bypass_turns % len(self._BYPASS_REPLIES)
            self._bypass_turns += 1
            text = self._BYPASS_REPLIES[idx]
            logger.info("LLM_BYPASS: replying with canned text (no LLM call): %r", text)
            await self.session.say(text)
            raise StopResponse()

        # Filler-only utterances that arrive while the agent is speaking (or its
        # speech just ended) are echo/filler glitches: do not let them start a
        # reply. Checked here, before the reply is scheduled.
        try:
            raw = new_message.raw_text_content or ""
        except Exception:
            raw = ""
        recent_speaking = (
            self.session.agent_state == "speaking"
            or (
                _LAST_AGENT_SPEECH_END_TS is not None
                and time.time() - _LAST_AGENT_SPEECH_END_TS < 1.5
            )
        )
        if _is_filler_only(raw) and recent_speaking:
            logger.info(
                "Dropping filler-only user turn while the agent spoke; no LLM call will be made: %r",
                raw,
            )
            raise StopResponse()

        # fix common speech-recognition mishearings of "Reditus" before the LLM
        # (or the consultation tool) ever sees them
        replaced = False
        parts: list = []
        for part in new_message.content:
            if isinstance(part, str):
                fixed = _apply_transcript_replacements(part)
                if fixed != part:
                    replaced = True
                parts.append(fixed)
            else:
                parts.append(part)
        if replaced:
            new_message.content = parts

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {now}"

    @function_tool
    async def request_consultation(
        self,
        context: RunContext,
        name: str,
        contact: str,
        project_summary: str,
        service: str = "general",
    ) -> str:
        """Record a confirmed consultation request so the Reditus Labs team can follow up.
        Only call this AFTER reading the details back and the user confirmed them.

        Args:
            name: The visitor's name
            contact: Confirmed email address or phone number
            project_summary: One-sentence description of what they want to build or fix
            service: The service they are interested in, if any
        """
        # timestamp+uuid: unique across restarts without any state file
        reference = f"RL-{datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:4]}"
        record = {
            "reference": reference,
            "name": name,
            "contact": contact,
            "service": service,
            "project_summary": project_summary,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"

        def _append() -> None:
            with open(self.requests_file, "a", encoding="utf-8") as f:
                f.write(line)

        try:
            await asyncio.to_thread(_append)
        except OSError as e:
            logger.error("Could not save consultation request: %s", e)
            return (
                "The request could not be saved automatically. Tell the user the team will "
                "still follow up, and note the details verbally."
            )
        logger.info("Consultation recorded: %s (%s)", reference, name)
        return (
            f"Consultation request recorded with reference {reference}. "
            f"The Reditus Labs team will reach out to {name} at {contact}."
        )


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def _build_llm() -> openai.LLM:
    """OmniRouter via the OpenAI-compatible plugin. base_url/api_key are
    validated so a missing value can never silently route to api.openai.com."""
    base_url = _require_env("OMNIROUTER_BASE_URL")
    api_key = _require_env("OMNIROUTER_API_KEY")
    model = (os.getenv("OMNIROUTER_MODEL") or "").strip() or "minimax-m2.5"
    temperature = _env_float("LLM_TEMPERATURE", 0.2)
    max_tokens = _env_int("LLM_MAX_TOKENS", 120)

    logger.info(
        "LLM: model=%s via OmniRouter at %s (temperature=%.2f, max_tokens=%d)",
        model, base_url, temperature, max_tokens,
    )

    # passthrough for experiments like {"thinking":{"type":"disabled"}} or
    # {"chat_template_kwargs":{"enable_thinking":false}}; measured on
    # 2026-10-06: no consistent win for glm-5 with the agent's tools, so
    # it stays off by default.
    extra_body: dict | None = None
    raw_extra = (os.getenv("OMNIROUTER_EXTRA_BODY") or "").strip()
    if raw_extra:
        try:
            extra_body = json.loads(raw_extra)
            if not isinstance(extra_body, dict):
                raise ValueError("OMNIROUTER_EXTRA_BODY must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            raise ConfigError(f"OMNIROUTER_EXTRA_BODY is not valid JSON: {e}") from None

    return openai.LLM(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        extra_body=extra_body or NOT_GIVEN,
    )


# ---------------------------------------------------------------------------
# Worker wiring
# ---------------------------------------------------------------------------
def prewarm(proc: agents.JobProcess):
    """Load every model once per worker process (num_idle_processes env)."""
    # thread caps: keep STT/TTS from starving each other on a 4-core CPU
    cores = os.cpu_count() or 4
    stt_threads = _env_int("STT_THREADS", max(2, cores - 2))
    tts_threads = _env_int("TTS_THREADS", max(1, cores - 1))
    os.environ.setdefault("OMP_NUM_THREADS", str(stt_threads))
    logger.info("prewarm: pid=%d, os.cpu_count=%d, STT threads=%d, TTS threads=%d",
                os.getpid(), cores, stt_threads, tts_threads)

    logger.info("Loading VAD (min_speech=%.2fs, min_silence=%.2fs, max_buffered_speech=%.0fs) [pid=%d]",
                _env_float("VAD_MIN_SPEECH_DURATION", 0.05), _env_float("VAD_MIN_SILENCE_DURATION", 0.45),
                _env_float("VAD_MAX_BUFFERED_SPEECH", 15.0), os.getpid())
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=_env_float("VAD_MIN_SPEECH_DURATION", 0.05),
        min_silence_duration=_env_float("VAD_MIN_SILENCE_DURATION", 0.45),
        prefix_padding_duration=_env_float("VAD_PREFIX_PADDING", 0.5),
        max_buffered_speech=_env_float("VAD_MAX_BUFFERED_SPEECH", 15.0),
        activation_threshold=_env_float("VAD_ACTIVATION_THRESHOLD", 0.5),
    )
    # local streaming EOU detector (turn-detector-v1-mini, fully in-process);
    # EOT() load here so the first session doesn't pay the ~0.4s model load
    from livekit.local_inference import EOT

    EOT()
    proc.userdata["eou"] = TurnDetector(version="v1-mini")
    proc.userdata["stt"] = _build_stt()
    tts_instance = KokoroLocalTTS(
        model_path=_env_path("KOKORO_MODEL_PATH", DEFAULT_KOKORO_MODEL),
        voices_path=_env_path("KOKORO_VOICES_PATH", DEFAULT_KOKORO_VOICES),
        voice=os.getenv("KOKORO_VOICE", "af_heart"),
        speed=_env_float("KOKORO_SPEED", 1.0),
        lang=os.getenv("KOKORO_LANG", "en-us"),
        num_threads=tts_threads,
    )
    _prime_phrase_cache(tts_instance)
    proc.userdata["tts"] = tts_instance


async def _inactivity_watchdog(session: AgentSession) -> None:
    """Nudge after IDLE_PROMPT_SECONDS of silence; say goodbye after two nudges."""
    idle_limit = _env_float("IDLE_PROMPT_SECONDS", 10.0)
    last_activity = time.time()
    nudges = 0

    def _touch(*_args, **_kwargs) -> None:
        nonlocal last_activity
        last_activity = time.time()

    session.on("conversation_item_added", lambda ev: _touch())
    session.on("user_state_changed", lambda ev: _touch())
    session.on("agent_state_changed", lambda ev: _touch())

    try:
        while True:
            await asyncio.sleep(2.0)
            if session.agent_state == "speaking" or session.user_state == "speaking":
                _touch()
                continue
            if time.time() - last_activity < idle_limit:
                continue
            nudges += 1
            last_activity = time.time()
            if nudges <= 2:
                await session.say(
                    "Are you still there? I can tell you about our services or book a consultation.",
                    allow_interruptions=True,
                )
            else:
                await session.say(
                    "It seems you had to go. Thank you for calling Reditus Labs. Goodbye!",
                    allow_interruptions=False,
                )
                logger.info("inactivity watchdog: said goodbye, closing session")
                await asyncio.sleep(6.0)
                return
    except asyncio.CancelledError:
        pass


async def entrypoint(ctx: agents.JobContext):
    """Entry point for the Reditus voice agent."""

    session = AgentSession(
        stt=ctx.proc.userdata["stt"],
        llm=_build_llm(),
        tts=ctx.proc.userdata["tts"],
        vad=ctx.proc.userdata["vad"],
        turn_handling={
            # local streaming EOU detector (turn-detector-v1-mini, in-process);
            # no Cloud inference involved
            "turn_detection": ctx.proc.userdata["eou"],
            "endpointing": {
                "min_delay": _env_float("EOU_MIN_DELAY", 0.4),
                "max_delay": _env_float("EOU_MAX_DELAY", 1.0),
            },
            # barge-in is allowed, but a cough or a click must not cut the agent off:
            "interruption": {
                "enabled": True,
                # "vad" keeps barge-in fully local; "adaptive" would try the
                # LiveKit Cloud inference gateway (401 without cloud creds)
                "mode": "vad",
                "min_duration": _env_float("INTERRUPTION_MIN_DURATION", 0.8),
                "min_words": _env_int("INTERRUPTION_MIN_WORDS", 3),
                "resume_false_interruption": True,
                "false_interruption_timeout": _env_float("FALSE_INTERRUPTION_TIMEOUT", 1.5),
            },
            # start the LLM before the turn is fully confirmed (hides EOU latency)
            "preemptive_generation": {"enabled": True, "preemptive_tts": False},
        },
    )

    def _on_conversation_item(ev) -> None:
        _log_turn_metrics(ev.item)
        item = ev.item
        # don't let "okay", "yeah" cut in on the agent's own speech or land in history
        if (
            isinstance(item, ChatMessage)
            and item.role == "user"
            and session.agent_state == "speaking"
            and _is_filler_only(item.text_content or "")
        ):
            logger.info(
                "Dropped filler-only transcript while agent is speaking: %r",
                item.text_content,
            )
            session.history.items = [i for i in session.history.items if i is not item]

    session.on("conversation_item_added", _on_conversation_item)

    # never go silent: if TTS died this turn, try one spoken fallback
    async def _on_tts_error(_err) -> None:
        try:
            await session.say(
                "Sorry, my voice glitched for a second. Could you repeat that?",
                allow_interruptions=True,
            )
        except Exception:
            logger.exception("TTS fallback failed too — session may be silent")

    ctx.proc.userdata["tts"].on("error", lambda err: asyncio.create_task(_on_tts_error(err)))

    await session.start(room=ctx.room, agent=ReditusAssistant())

    watchdog = asyncio.create_task(_inactivity_watchdog(session))

    # keep the public fact that the agent spoke *recently*, so
    # on_user_turn_completed can tell filler-noise from a targeted answer
    def _track_agent_state(ev) -> None:
        global _LAST_AGENT_SPEECH_END_TS
        try:
            old = getattr(ev, "old_state", None)
            new = getattr(ev, "new_state", None)
        except AttributeError:
            return
        if old == "speaking" and new != "speaking":
            _LAST_AGENT_SPEECH_END_TS = time.time()
        elif new == "speaking":
            _LAST_AGENT_SPEECH_END_TS = None

    session.on("agent_state_changed", _track_agent_state)

    await session.generate_reply(
        instructions=(
            "Start the conversation by saying exactly this greeting, nothing more: "
            "'Welcome to Reditus Labs! I'm your virtual assistant. What can I do for you today?'"
        )
    )

    closed = asyncio.Event()
    session.on("close", lambda ev: closed.set())
    try:
        await asyncio.wait(
            [asyncio.create_task(closed.wait()), watchdog],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        watchdog.cancel()
        logger.info("event-loop stalls this session: %d", _LOOP_STALL_COUNT)


if __name__ == "__main__":
    _configure_logging()
    try:
        validate_environment()
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}")
        raise SystemExit(1)

    cores = os.cpu_count() or 4
    idle = _env_int("NUM_IDLE_PROCESSES", 0)
    logger.info(
        "starting worker: os.cpu_count=%d, NUM_IDLE_PROCESSES=%d, STT_BACKEND=%s",
        cores, idle, (os.getenv("STT_BACKEND") or "parakeet").strip().lower(),
    )
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # 0 for local testing (one model copy only), 1 for production
            num_idle_processes=idle,
            # first run may download the turn-detector ONNX + whisper fallback weights
            initialize_process_timeout=600.0,
        )
    )
