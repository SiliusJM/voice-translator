"""
Traductor Local de Voz a Voz
Preserva tu voz mientras traduce a otro idioma.
Se ejecuta completamente en GPU — sin APIs de pago requeridas.

Requisitos:  ver requirements.txt
Primera vez:  descarga Whisper large-v3 (~1.5 GB) y Chatterbox TTS (~1.5 GB).
"""

import os
import uuid
import logging
import warnings
import subprocess
import time
import hashlib
import threading
import gradio as gr
import torch
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Poner ffmpeg + ffprobe (static-ffmpeg) en PATH antes de que pydub/Gradio los busquen
_ffmpeg = _ffprobe = None
try:
    import static_ffmpeg
    import shutil
    import pydub
    static_ffmpeg.add_paths()
    _ffmpeg  = shutil.which("ffmpeg")
    _ffprobe = shutil.which("ffprobe")
    if _ffmpeg:
        pydub.AudioSegment.converter = _ffmpeg
        pydub.AudioSegment.ffmpeg    = _ffmpeg
    if _ffprobe:
        pydub.AudioSegment.ffprobe   = _ffprobe
except Exception:
    pass

# Carpeta de salida para los audios traducidos
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================================
# -- Proveedor TTS — Abstraccion para multiples motores de sintesis ------------
# ==============================================================================

class TTSProvider:
    """Clase base para proveedores de TTS."""
    name: str = "base"

    @property
    def sr(self) -> int:
        return 24000

    def generate(self, text: str, reference_wav: str, exaggeration: float = 0.5):
        """Genera audio a partir de texto. Devuelve numpy array float32."""
        raise NotImplementedError

    def test_connection(self, api_key: str = "") -> str:
        """Prueba la conexion con el proveedor. Devuelve mensaje de estado."""
        return "Conexion exitosa."


class ChatterboxProvider(TTSProvider):
    """Chatterbox TTS — clonacion de voz local en GPU."""
    name = "Chatterbox (Local GPU)"

    def __init__(self):
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from chatterbox.tts import ChatterboxTTS
            print("Cargando Chatterbox TTS (primera vez descarga ~1.5 GB)...")
            self._model = ChatterboxTTS.from_pretrained(device=DEVICE)
            # Warmup: la primera inferencia compila kernels CUDA y llena caches
            try:
                import numpy as _np
                import scipy.io.wavfile as _wav
                _wpath = str(OUTPUT_DIR / "_warmup_tmp.wav")
                _wav.write(_wpath, self._model.sr,
                           (_np.random.randn(self._model.sr).astype(_np.float32) * 0.01))
                print("  Warmup TTS (compilando kernels CUDA)...")
                with torch.inference_mode():
                    _ = self._model.generate(
                        "Testing one two three.",
                        audio_prompt_path=_wpath, exaggeration=0.3,
                    )
                Path(_wpath).unlink(missing_ok=True)
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print("  Warmup completado.")
            except Exception as e:
                print(f"  Warmup omitido: {e}")
            print("Chatterbox TTS listo.")
        return self._model

    @property
    def sr(self):
        return self._ensure_model().sr

    def generate(self, text: str, reference_wav: str, exaggeration: float = 0.5):
        import numpy as np
        model = self._ensure_model()
        with torch.inference_mode():
            wav = model.generate(text, audio_prompt_path=reference_wav, exaggeration=exaggeration)
        return wav.squeeze().cpu().numpy().astype(np.float32)

    def test_connection(self, api_key: str = "") -> str:
        try:
            self._ensure_model()
            return f"Chatterbox TTS cargado en **{DEVICE.upper()}**."
        except Exception as e:
            return f"Error cargando Chatterbox: {e}"


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs API — sintesis de voz en la nube con clonacion."""
    name = "ElevenLabs (API)"
    _voice_id: str | None = None
    _api_key: str = ""

    @property
    def sr(self):
        return 44100  # ElevenLabs devuelve 44100 Hz por defecto con output_format pcm_44100

    def set_api_key(self, key: str):
        self._api_key = key.strip()

    def _ensure_voice(self, reference_wav: str) -> str:
        """Crea (o reutiliza) una voz clonada a partir del audio de referencia."""
        import requests
        if self._voice_id:
            return self._voice_id

        if not self._api_key:
            raise RuntimeError("Configura tu clave API de ElevenLabs antes de sintetizar.")

        headers = {"xi-api-key": self._api_key}

        # Crear voz clonada con la referencia
        with open(reference_wav, "rb") as f:
            resp = requests.post(
                "https://api.elevenlabs.io/v1/voices/add",
                headers=headers,
                data={"name": f"voice-translator-{uuid.uuid4().hex[:6]}", "description": "Auto-cloned voice"},
                files={"files": ("reference.wav", f, "audio/wav")},
                timeout=60,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"ElevenLabs clone failed ({resp.status_code}): {resp.text[:200]}")

        self._voice_id = resp.json()["voice_id"]
        print(f"  [ElevenLabs] Voz clonada: {self._voice_id}")
        return self._voice_id

    def generate(self, text: str, reference_wav: str, exaggeration: float = 0.5):
        import requests
        import numpy as np

        voice_id = self._ensure_voice(reference_wav)
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": min(exaggeration, 1.0),
            },
            "output_format": "pcm_44100",
        }
        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ElevenLabs TTS failed ({resp.status_code}): {resp.text[:200]}")

        # PCM 16-bit little-endian -> float32 numpy
        audio_int16 = np.frombuffer(resp.content, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    def cleanup_voice(self):
        """Elimina la voz clonada temporal del servidor."""
        if not self._voice_id or not self._api_key:
            return
        try:
            import requests
            requests.delete(
                f"https://api.elevenlabs.io/v1/voices/{self._voice_id}",
                headers={"xi-api-key": self._api_key},
                timeout=10,
            )
            print(f"  [ElevenLabs] Voz temporal eliminada: {self._voice_id}")
        except Exception:
            pass
        self._voice_id = None

    def test_connection(self, api_key: str = "") -> str:
        if not api_key.strip():
            return "Ingresa tu clave API de ElevenLabs."
        try:
            import requests
            resp = requests.get(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": api_key.strip()},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                chars = data.get("subscription", {}).get("character_count", "?")
                limit = data.get("subscription", {}).get("character_limit", "?")
                return f"ElevenLabs conectado. Caracteres usados: {chars}/{limit}"
            elif resp.status_code == 401:
                return "Clave API de ElevenLabs invalida."
            else:
                return f"ElevenLabs respondio con codigo {resp.status_code}."
        except ImportError:
            return "Instala `requests` para usar ElevenLabs: pip install requests"
        except Exception as e:
            return f"Error: {e}"


# Registro de proveedores TTS disponibles
TTS_PROVIDERS: dict[str, TTSProvider] = {
    "chatterbox": ChatterboxProvider(),
    "elevenlabs": ElevenLabsProvider(),
}
TTS_PROVIDER_CHOICES = [
    ("Chatterbox (Local GPU) — Recomendado", "chatterbox"),
    ("ElevenLabs (API nube) — Requiere clave", "elevenlabs"),
]
DEFAULT_TTS_PROVIDER = "chatterbox"


def _get_tts_provider(name: str = DEFAULT_TTS_PROVIDER) -> TTSProvider:
    return TTS_PROVIDERS.get(name, TTS_PROVIDERS[DEFAULT_TTS_PROVIDER])


def test_tts_connection(provider_name: str, api_key: str = "") -> str:
    provider = _get_tts_provider(provider_name)
    return provider.test_connection(api_key)


# ==============================================================================
# -- Claves API ----------------------------------------------------------------
# ==============================================================================

GEMINI_KEY_FILE = Path(__file__).parent / ".gemini_key"
ELEVENLABS_KEY_FILE = Path(__file__).parent / ".elevenlabs_key"

def _load_gemini_key() -> str:
    if GEMINI_KEY_FILE.exists():
        return GEMINI_KEY_FILE.read_text(encoding="utf-8").strip()
    return os.environ.get("GEMINI_API_KEY", "")

def _save_gemini_key(key: str) -> str:
    GEMINI_KEY_FILE.write_text(key.strip(), encoding="utf-8")
    return "Clave Gemini guardada."

def _load_elevenlabs_key() -> str:
    if ELEVENLABS_KEY_FILE.exists():
        return ELEVENLABS_KEY_FILE.read_text(encoding="utf-8").strip()
    return os.environ.get("ELEVENLABS_API_KEY", "")

def _save_elevenlabs_key(key: str) -> str:
    ELEVENLABS_KEY_FILE.write_text(key.strip(), encoding="utf-8")
    return "Clave ElevenLabs guardada."


# Modelos Gemini conocidos
GEMINI_MODELS = [
    ("gemini-3.1-flash-lite-preview (15 RPM — RECOMENDADO)", "gemini-3.1-flash-lite-preview"),
    ("gemini-2.5-flash-lite (10 RPM)", "gemini-2.5-flash-lite"),
    ("gemini-2.5-flash (5 RPM)", "gemini-2.5-flash"),
    ("gemini-2.0-flash (10 RPM)", "gemini-2.0-flash"),
    ("gemini-1.5-flash (15 RPM)", "gemini-1.5-flash"),
]
GEMINI_DEFAULT = "gemini-3.1-flash-lite-preview"


def list_gemini_models(api_key: str):
    """Consulta la API y devuelve los modelos disponibles."""
    import google.generativeai as genai
    if not api_key.strip():
        return gr.Dropdown(choices=GEMINI_MODELS, value=GEMINI_DEFAULT), "Ingresa tu clave API primero."
    try:
        genai.configure(api_key=api_key.strip())
        api_models = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
            and "flash" in m.name.lower()
            and "tts" not in m.name.lower()
            and "dialog" not in m.name.lower()
            and "live" not in m.name.lower()
        ]
        detected_choices = [(f"{m} (detectado)", m) for m in sorted(api_models)]
        extra_known = [(lbl, v) for lbl, v in GEMINI_MODELS if v not in {m for m in api_models}]
        choices = detected_choices + extra_known
        best = next((m for m in api_models if "3.1-flash-lite" in m),
                     next((m for m in api_models if "flash-lite" in m),
                     next((m for m in api_models if "flash" in m), GEMINI_DEFAULT)))
        print(f"  [Gemini] Modelos detectados: {api_models}")
        return gr.Dropdown(choices=choices, value=best), f"{len(api_models)} modelos detectados. Seleccionado: **{best}**"
    except Exception as e:
        return gr.Dropdown(choices=GEMINI_MODELS, value=GEMINI_DEFAULT), f"No se pudo listar: {str(e)[:100]}"


def test_gemini_connection(api_key: str, model_name: str) -> str:
    """Prueba la conexion con el modelo Gemini seleccionado."""
    import google.generativeai as genai
    import google.api_core.exceptions
    if not api_key.strip():
        return "Ingresa tu clave API primero."
    try:
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Reply with just: OK")
        if response.text:
            return f"Conexion exitosa con **{model_name}**. Respuesta: `{response.text.strip()[:40]}`"
        return "Modelo respondio sin texto — puede estar limitado."
    except google.api_core.exceptions.ResourceExhausted as e:
        import re as _re
        wait = _re.search(r"retry in (\d+)", str(e))
        secs = wait.group(1) if wait else "?"
        return f"**{model_name}** — Cuota agotada. Espera {secs}s o elige otro modelo."
    except google.api_core.exceptions.PermissionDenied:
        return f"**{model_name}** — Sin permiso. Tu clave no tiene acceso a este modelo."
    except google.api_core.exceptions.InvalidArgument:
        return f"Nombre de modelo invalido: `{model_name}`."
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)[:120]}"


# Silenciar logs ruidosos
logging.getLogger("stanza").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

class _H11Filter(logging.Filter):
    def filter(self, record):
        if record.exc_info and record.exc_info[1]:
            if "Content-Length" in str(record.exc_info[1]):
                return False
        return True

logging.getLogger("uvicorn.error").addFilter(_H11Filter())
warnings.filterwarnings("ignore", category=FutureWarning)

# -- Deteccion de dispositivo --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
print(f"Dispositivo: {DEVICE}")

# Detectar GPU nombre y VRAM
_GPU_NAME = "Desconocida"
_GPU_VRAM_GB = 0.0
if DEVICE == "cuda":
    try:
        _GPU_NAME = torch.cuda.get_device_name(0)
        _GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU: {_GPU_NAME} ({_GPU_VRAM_GB:.1f} GB VRAM)")
    except Exception:
        # Fallback: mem_get_info
        try:
            _, total = torch.cuda.mem_get_info()
            _GPU_VRAM_GB = total / (1024 ** 3)
            print(f"  GPU: {_GPU_NAME} ({_GPU_VRAM_GB:.1f} GB VRAM) [fallback]")
        except Exception:
            pass

# -- Optimizaciones TF32 para GPUs modernas (Ampere+, RTX 30xx/40xx/50xx) -----
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("  TF32 habilitado (matmul + cuDNN)")

# -- Modo GPU agresivo (se activa con checkbox) --------------------------------
_gpu_agresivo_activo = False

def _activar_gpu_agresivo():
    """Activa cudnn.benchmark para auto-seleccionar algoritmos de convolucion."""
    global _gpu_agresivo_activo
    if _gpu_agresivo_activo or DEVICE != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    print("  [GPU Agresivo] cudnn.benchmark = True")
    _gpu_agresivo_activo = True
    print(f"  [GPU Agresivo] Modo activado en {_GPU_NAME}.")

WHISPER_MODEL_SIZE = "large-v3"

# -- Modelos lazy-load ---------------------------------------------------------
_whisper = None

def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print("Cargando modelo Whisper...")
        _whisper = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Whisper listo.")
    return _whisper


def get_tts():
    """Atajo para obtener el modelo Chatterbox (retrocompatibilidad)."""
    provider = _get_tts_provider("chatterbox")
    provider._ensure_model()
    return provider._model


# -- Idiomas soportados --------------------------------------------------------
LANGUAGES: dict[str, str] = {
    "en": "Ingles",
    "es": "Espa\u00f1ol",
    "fr": "Frances",
    "de": "Aleman",
    "pt": "Portugues",
    "it": "Italiano",
    "ar": "Arabe",
    "ja": "Japones",
    "zh": "Chino",
    "ru": "Ruso",
    "ko": "Coreano",
    "hi": "Hindi",
    "nl": "Holandes",
    "pl": "Polaco",
    "tr": "Turco",
    "cs": "Checo",
    "hu": "Hungaro",
}

CJK_LANGS = {"ja", "zh", "ko"}

# -- Cache de transcripcion ----------------------------------------------------
_transcription_cache: dict[str, tuple[str, str, list]] = {}

def _hash_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


# ==============================================================================
# -- Pipeline principal --------------------------------------------------------
# ==============================================================================

def transcribe(audio_path: str) -> tuple[str, str, list]:
    """Transcribe audio; devuelve (texto_completo, codigo_idioma, segmentos)."""
    model = get_whisper()
    segs_iter, info = model.transcribe(
        audio_path,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )
    seg_list = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segs_iter
        if s.no_speech_prob < 0.6 and s.text.strip()
    ]
    text = " ".join(s["text"] for s in seg_list).strip()
    return text, info.language, seg_list


def translate_batch_gemini(
    texts: list[str], from_code: str, to_code: str, api_key: str,
    model_name: str = "gemini-1.5-flash",
    prev_context: str = "",
) -> list[str]:
    """Traduce todos los segmentos en UNA sola llamada a Gemini Flash."""
    import google.generativeai as genai
    import google.api_core.exceptions
    import json

    if from_code == to_code:
        return texts, texts
    if not api_key:
        raise RuntimeError("Ingresa tu clave de API de Google AI Studio.")

    genai.configure(api_key=api_key)

    FALLBACK_MODELS = ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    candidates = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]

    model = None
    last_exc = None
    for candidate in candidates:
        try:
            model = genai.GenerativeModel(candidate)
            break
        except Exception as e:
            last_exc = e
            continue
    if model is None:
        raise RuntimeError(f"No se pudo inicializar ningun modelo Gemini: {last_exc}")

    lang_from = LANGUAGES.get(from_code, from_code)
    lang_to = LANGUAGES.get(to_code, to_code)

    payload = json.dumps(
        [{"i": i, "t": t} for i, t in enumerate(texts)],
        ensure_ascii=False,
    )
    context_hint = ""
    if prev_context:
        context_hint = (
            f"Previous translated text (for coherence only, do NOT include in output): "
            f"{prev_context[:300]}\n\n"
        )
    needs_romanization = to_code in CJK_LANGS
    if needs_romanization:
        roman_rules = {
            "ja": ("romaji", "Use extended vowels as double vowels: ou instead of \u014d, ei instead of \u0113, aa instead of \u0101, ii instead of \u012b, uu instead of \u016b. No macrons."),
            "zh": ("pinyin", "Standard pinyin without tone numbers or diacritics."),
            "ko": ("romanized Korean", "Standard revised romanization."),
        }
        roman_name, roman_detail = roman_rules.get(to_code, ("romanized", ""))
        prompt = (
            context_hint +
            f"Translate each item from {lang_from} to {lang_to}. "
            "Return ONLY a JSON array. Each item MUST have: i (index), t (translated in native script), "
            f"r ({roman_name} pronunciation for reading aloud). "
            f"{roman_detail} "
            "Natural fluent translation preserving tone. No markdown, no explanations.\n\n"
            f"Input: {payload}\n\nOutput:"
        )
    else:
        prompt = (
            context_hint +
            f"Translate each item from {lang_from} to {lang_to}. "
            "Return ONLY a JSON array with the same structure. "
            "Keep technical terms, product names, model numbers, and proper nouns EXACTLY as-is "
            "(e.g. RTX 3060 Ti, i5, GTX 780, etc.). "
            "Natural fluent translation preserving tone. No markdown, no explanations.\n\n"
            f"Input: {payload}\n\nOutput:"
        )

    response = None
    for attempt, candidate in enumerate(candidates):
        try:
            if attempt > 0:
                model = genai.GenerativeModel(candidate)
                print(f"  [Gemini] Intentando con modelo: {candidate}")
            response = model.generate_content(prompt)
            break
        except google.api_core.exceptions.ResourceExhausted:
            print(f"  [Gemini] Cuota agotada en {candidate}, intentando siguiente modelo...")
            last_exc = RuntimeError(f"Cuota agotada en {candidate}")
            continue
        except Exception as e:
            last_exc = e
            print(f"  [Gemini] Error en {candidate}: {e}")
            continue

    if response is None:
        raise RuntimeError(
            f"Todos los modelos Gemini fallaron. Ultimo error: {last_exc}\n"
            "Verifica tu clave en https://aistudio.google.com y espera unos minutos si excediste la cuota."
        )
    raw = response.text.strip()

    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]

    data = json.loads(raw)
    data.sort(key=lambda x: x["i"])

    result_map = {item["i"]: item["t"] for item in data}
    native_texts = [result_map.get(i, texts[i]) for i in range(len(texts))]
    if needs_romanization:
        roman_map = {item["i"]: item.get("r", item["t"]) for item in data}
        tts_texts = [roman_map.get(i, texts[i]) for i in range(len(texts))]
    else:
        tts_texts = native_texts
    return native_texts, tts_texts


def ensure_wav(audio_path: str) -> tuple[str, bool]:
    """Convierte cualquier formato de audio a WAV usando ffmpeg."""
    p = Path(audio_path)
    if p.suffix.lower() == ".wav":
        return audio_path, False
    out = str(OUTPUT_DIR / f"_in_{uuid.uuid4().hex[:8]}.wav")
    if _ffmpeg:
        subprocess.run(
            [_ffmpeg, "-y", "-i", audio_path, out],
            check=True, capture_output=True,
        )
        return out, True
    from pydub import AudioSegment
    seg = AudioSegment.from_file(audio_path)
    seg.export(out, format="wav")
    return out, True


def _build_atempo_filter(rate: float) -> str:
    """Construye cadena de filtros atempo para ffmpeg; cada filtro acepta solo 0.5-2.0."""
    filters = []
    r = rate
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r *= 2.0
    filters.append(f"atempo={r:.6f}")
    return ",".join(filters)


# Flag de cancelacion global
_cancel_flag = threading.Event()

# -- Maximo de caracteres por llamada TTS --------------------------------------
# Chunks mas cortos = menos bloqueo de GPU por fragmento = mas fluidez
_TTS_MAX_CHARS = 150


def _split_long_text(text: str, max_chars: int = _TTS_MAX_CHARS) -> list[str]:
    """Divide un texto largo en fragmentos mas cortos para TTS."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > max_chars:
        best_cut = -1
        for sep in ['. ', '! ', '? ', '; ', ', ', ' ']:
            idx = remaining.rfind(sep, 0, max_chars)
            if idx > max_chars // 4:
                best_cut = idx + len(sep)
                break
        if best_cut <= 0:
            best_cut = remaining.rfind(' ', 0, max_chars)
            if best_cut <= 0:
                best_cut = max_chars
        chunks.append(remaining[:best_cut].strip())
        remaining = remaining[best_cut:].strip()
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


def synthesize_translated(
    segments: list[dict],
    translated_texts: list[str],
    reference_wav: str,
    tgt_lang: str,
    exaggeration: float = 0.5,
    sync_duration: bool = False,
    tts_provider_name: str = DEFAULT_TTS_PROVIDER,
    gpu_agresivo: bool = False,
    elevenlabs_key: str = "",
) -> str:
    """
    Sintetiza CADA segmento individualmente y los concatena con silencios
    proporcionales al ritmo original.

    Optimizaciones:
    - Divide textos largos (>150 chars) en fragmentos mas cortos.
    - Pausas proporcionales: segmentos donde TTS habla mas rapido que el
      original reciben pausas extra para mantener un ritmo natural.
    - Relleno final limitado a 3s maximo.
    """
    import scipy.io.wavfile as wav_io
    import soundfile as sf
    import numpy as np
    from datetime import datetime

    _cancel_flag.clear()
    provider = _get_tts_provider(tts_provider_name)
    if isinstance(provider, ElevenLabsProvider) and elevenlabs_key:
        provider.set_api_key(elevenlabs_key)
    sample_rate = provider.sr

    # Activar modo GPU agresivo si lo pidieron
    if gpu_agresivo and DEVICE == "cuda":
        _activar_gpu_agresivo()

    # Preparar referencia de voz (max 15 s)
    MAX_REF_SECONDS = 15
    ref_data, ref_sr = sf.read(reference_wav, always_2d=False, dtype="float32")
    if ref_data.ndim > 1:
        ref_data = ref_data[:, 0]
    if len(ref_data) > MAX_REF_SECONDS * ref_sr:
        ref_data = ref_data[:MAX_REF_SECONDS * ref_sr]
    tmp_ref = str(OUTPUT_DIR / f"_ref_{uuid.uuid4().hex[:8]}.wav")
    sf.write(tmp_ref, ref_data, ref_sr, subtype="PCM_16")

    total = len([t for t in translated_texts if t.strip()])
    print(f"  [TTS] {total} segmentos a sintetizar ({provider.name})")

    # -- Pipeline de paso unico: genera + ensambla simultaneamente --
    # En vez de 3 pasos separados (generar → calcular pausas → concatenar),
    # generamos cada segmento y lo agregamos al buffer inmediatamente.
    # Esto reduce uso de memoria y elimina overhead de post-procesamiento.
    PACING_FACTOR = 0.45  # 0.0=sin pausas extra, 1.0=igualar duracion original
    original_duration = segments[-1]["end"] if segments else 0.0

    audio_pieces = []  # lista de numpy arrays que se concatenan al final
    t0 = time.time()
    done = 0
    prev_end = 0.0
    total_tts_audio = 0.0
    total_extra_pauses = 0.0

    try:
        for idx, (seg, translated) in enumerate(zip(segments, translated_texts)):
            if _cancel_flag.is_set():
                print("  [TTS] Cancelado por el usuario.")
                raise RuntimeError("Cancelado por el usuario.")

            # 1. Silencio original antes de este segmento
            gap = max(0.0, seg["start"] - prev_end)
            if gap > 0.02:
                audio_pieces.append(np.zeros(int(gap * sample_rate), dtype=np.float32))
            prev_end = seg["end"]

            if not translated.strip():
                continue

            done += 1
            elapsed = time.time() - t0

            # 2. Dividir textos largos en fragmentos mas cortos
            text_chunks = _split_long_text(translated)
            if len(text_chunks) > 1:
                print(f"  [TTS {done}/{total}] {seg['start']:.1f}s-{seg['end']:.1f}s "
                      f"({elapsed:.0f}s) | {translated[:70]!r} [{len(text_chunks)} fragmentos]")
            else:
                print(f"  [TTS {done}/{total}] {seg['start']:.1f}s-{seg['end']:.1f}s "
                      f"({elapsed:.0f}s) | {translated[:70]!r}")

            # 3. Generar audio TTS para cada fragmento
            chunk_wavs = []
            for chunk_text in text_chunks:
                if _cancel_flag.is_set():
                    raise RuntimeError("Cancelado por el usuario.")
                wav_chunk = provider.generate(chunk_text, tmp_ref, exaggeration)
                chunk_wavs.append(wav_chunk)

            if len(chunk_wavs) > 1:
                # Micro-pausa entre fragmentos para respiracion natural
                breath = np.zeros(int(sample_rate * 0.10), dtype=np.float32)
                pieces = [chunk_wavs[0]]
                for cw in chunk_wavs[1:]:
                    pieces.append(breath)
                    pieces.append(cw)
                wav_np = np.concatenate(pieces)
            else:
                wav_np = chunk_wavs[0]
            tts_dur = len(wav_np) / sample_rate
            total_tts_audio += tts_dur
            print(f"    -> {tts_dur:.1f}s generado")

            # 4. Agregar audio al buffer inmediatamente (pipeline overlap)
            audio_pieces.append(wav_np)

            # 5. Calcular y agregar pausa proporcional al ritmo
            orig_seg_dur = seg["end"] - seg["start"]
            time_saved = max(0.0, orig_seg_dur - tts_dur)
            extra = time_saved * PACING_FACTOR
            if extra > 0.02:
                audio_pieces.append(np.zeros(int(extra * sample_rate), dtype=np.float32))
                total_extra_pauses += extra
    finally:
        try:
            Path(tmp_ref).unlink()
        except OSError:
            pass

    if not audio_pieces:
        raise RuntimeError("No se genero audio: los segmentos estaban vacios.")

    if total_extra_pauses > 0.5:
        print(f"  [TTS] Pausas de ritmo: +{total_extra_pauses:.1f}s distribuidos entre segmentos "
              f"(factor {PACING_FACTOR})")

    # Concatenar todas las piezas en un solo array
    combined = np.concatenate(audio_pieces)
    del audio_pieces  # liberar memoria inmediatamente

    total_elapsed = time.time() - t0
    tts_dur_total = len(combined) / sample_rate
    print(f"  [TTS] Completado: {tts_dur_total:.1f}s de audio en {total_elapsed:.0f}s "
          f"({total_elapsed/max(tts_dur_total,0.1):.1f}x tiempo real)")

    # Relleno final: maximo 3s de silencio al final
    MAX_END_PAD = 3.0
    if tts_dur_total < original_duration:
        missing = original_duration - tts_dur_total
        pad_seconds = min(missing, MAX_END_PAD)
        if pad_seconds > 0.1:
            pad_samples = int(pad_seconds * sample_rate)
            combined = np.concatenate([combined, np.zeros(pad_samples, dtype=np.float32)])
            if missing > MAX_END_PAD:
                print(f"  [TTS] Relleno final: +{pad_seconds:.1f}s (limitado; faltaban {missing:.1f}s)")
            else:
                print(f"  [TTS] Relleno final: +{pad_seconds:.1f}s")
    elif tts_dur_total > original_duration:
        print(f"  [TTS] Audio mas largo: {tts_dur_total:.1f}s vs {original_duration:.1f}s (se mantiene)")

    final_dur = len(combined) / sample_rate
    print(f"  [TTS] Duracion final: {final_dur:.1f}s (original: {original_duration:.1f}s)")

    # Convertir a int16 para archivo mas pequeno y carga mas rapida en el navegador
    combined_int16 = np.clip(combined * 32767, -32768, 32767).astype(np.int16)
    del combined

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(OUTPUT_DIR / f"{timestamp}_traducido-{tgt_lang}.wav")
    wav_io.write(out_path, sample_rate, combined_int16)

    # Sincronizar duracion con atempo si se solicita y el audio es mas largo
    if sync_duration and _ffmpeg and segments and tts_dur_total > original_duration * 1.05:
        atempo = tts_dur_total / original_duration
        stretched = out_path.replace(".wav", "_sync.wav")
        filter_str = _build_atempo_filter(atempo)
        try:
            subprocess.run(
                [_ffmpeg, "-y", "-i", out_path, "-filter:a", filter_str, stretched],
                check=True, capture_output=True,
            )
            Path(out_path).unlink()
            shutil.move(stretched, out_path)
        except Exception:
            pass

    # Convertir WAV a MP3 para carga rapida en el navegador (~10x mas pequeno)
    # Se conserva el WAV original en outputs/ para maxima calidad
    mp3_path = out_path.replace(".wav", ".mp3")
    if _ffmpeg:
        try:
            subprocess.run(
                [_ffmpeg, "-y", "-i", out_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
                check=True, capture_output=True,
            )
            mp3_size = Path(mp3_path).stat().st_size / 1024
            wav_size = Path(out_path).stat().st_size / 1024
            print(f"  [TTS] MP3 para navegador: {mp3_size:.0f}KB (WAV: {wav_size:.0f}KB, "
                  f"{wav_size/max(mp3_size,1):.0f}x mas pequeno)")
            return mp3_path
        except Exception as e:
            print(f"  [TTS] No se pudo crear MP3, usando WAV: {e}")

    return out_path


def save_as_mp3(wav_path: str | None):
    """Convierte el WAV/MP3 de salida a MP3 y lo devuelve para descarga."""
    if not wav_path or not Path(wav_path).exists():
        raise gr.Error("Primero genera una traduccion.")
    # Si ya es MP3, buscar el WAV correspondiente para ofrecer como descarga
    if wav_path.endswith(".mp3"):
        return gr.File(value=wav_path, visible=True)
    if not _ffmpeg:
        raise gr.Error("ffmpeg no encontrado; reinstala static-ffmpeg.")
    mp3_path = wav_path.replace(".wav", ".mp3")
    if not Path(mp3_path).exists():
        subprocess.run(
            [_ffmpeg, "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
            check=True, capture_output=True,
        )
    return gr.File(value=mp3_path, visible=True)


def pipeline(
    audio_file: str | None,
    source_lang: str,
    target_lang: str,
    exaggeration: float,
    sync_duration: bool,
    gpu_agresivo: bool,
    api_key: str,
    gemini_model: str,
    tts_provider_name: str,
    elevenlabs_key: str = "",
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """Pipeline completo de traduccion voz-a-voz."""
    pipeline_start = time.time()
    if audio_file is None:
        raise gr.Error("Sube un archivo de audio primero.")
    if not api_key.strip():
        raise gr.Error("Ingresa tu clave de API de Google AI Studio en la seccion Configuracion.")
    if tts_provider_name == "elevenlabs" and not elevenlabs_key.strip():
        raise gr.Error("Ingresa tu clave API de ElevenLabs en la seccion Sintesis de Voz.")

    try:
        wav_file, is_tmp = ensure_wav(audio_file)
    except Exception as exc:
        raise gr.Error(f"No se pudo leer el audio: {exc}") from exc

    try:
        # 1 -- Transcripcion con Whisper (con cache)
        file_hash = _hash_file(wav_file)
        if file_hash in _transcription_cache:
            original_text, detected, segments = _transcription_cache[file_hash]
            print(f"  [Cache] Transcripcion reutilizada ({len(segments)} segmentos)")
            progress(0.35, desc="Paso 1/3 -- Transcripcion en cache!")
        else:
            progress(0.05, desc="Paso 1/3 -- Transcribiendo con Whisper...")
            try:
                original_text, detected, segments = transcribe(wav_file)
            except Exception as exc:
                raise gr.Error(f"Error en transcripcion: {exc}") from exc
            _transcription_cache[file_hash] = (original_text, detected, segments)

        if not original_text or not segments:
            raise gr.Error("No se detecto habla en el audio.")

        src = detected if source_lang == "auto" else source_lang
        src_label = LANGUAGES.get(src, src)
        tgt_label = LANGUAGES.get(target_lang, target_lang)

        # 2 -- Traduccion con Gemini
        n_segs = len(segments)
        progress(0.40, desc=f"Paso 2/3 -- Traduciendo {n_segs} segmentos ({src_label} -> {tgt_label})...")
        try:
            texts = [s["text"] for s in segments]
            display_texts, tts_texts = translate_batch_gemini(texts, src, target_lang, api_key.strip(), gemini_model)
        except Exception as exc:
            raise gr.Error(f"Error en traduccion (Gemini): {exc}") from exc

        # 3 -- Sintesis TTS por segmento
        provider = _get_tts_provider(tts_provider_name)
        progress(0.55, desc=f"Paso 3/3 -- Sintetizando {n_segs} segmentos con {provider.name}...")
        try:
            output_audio = synthesize_translated(
                segments, tts_texts, wav_file, target_lang, exaggeration,
                sync_duration, tts_provider_name, gpu_agresivo, elevenlabs_key,
            )
        except Exception as exc:
            raise gr.Error(f"Error en sintesis: {exc}") from exc

        total_time = time.time() - pipeline_start
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)
        time_str = f"{total_min}m {total_sec}s" if total_min > 0 else f"{total_sec}s"
        print(f"  [TOTAL] Pipeline completo en {total_time:.0f}s")
        progress(1.0, desc=f"Listo en {time_str}! Guardado en outputs/")

        if target_lang in CJK_LANGS and display_texts != tts_texts:
            trans_display = f"[{tgt_label}]  {' '.join(display_texts)}\n[Romanizado]  {' '.join(tts_texts)}"
        else:
            trans_display = f"[{tgt_label}]  {' '.join(display_texts)}"

        trans_display += f"\n\nTiempo total: {time_str} | {n_segs} segmentos"

        return (
            f"[{src_label}]  {original_text}",
            trans_display,
            output_audio,
        )
    finally:
        # Limpiar voz clonada de ElevenLabs si se uso
        provider = TTS_PROVIDERS.get(tts_provider_name)
        if isinstance(provider, ElevenLabsProvider):
            provider.cleanup_voice()
        if is_tmp:
            try:
                Path(wav_file).unlink()
            except OSError:
                pass


# ==============================================================================
# -- Pipeline para audio largo (>3 min) ----------------------------------------
# ==============================================================================

_LONG_AUDIO_THRESHOLD = 180   # 3 minutos
_TRANSLATION_BATCH_SIZE = 25  # segmentos por lote de traduccion
_TTS_GROUP_MAX_CHARS = 230    # max caracteres por grupo TTS (sweet spot RTX 5060)


def _get_audio_duration(wav_path: str) -> float:
    """Devuelve la duracion en segundos de un archivo de audio."""
    import soundfile as sf
    info = sf.info(wav_path)
    return info.duration


def _load_state(state_path: Path) -> dict:
    import json
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {}


def _save_state(state_path: Path, state: dict):
    import json
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _group_segments_for_tts(
    segments: list[dict],
    tts_texts: list[str],
    max_chars: int = _TTS_GROUP_MAX_CHARS,
    max_gap: float = 2.0,
) -> list[dict]:
    """
    Agrupa segmentos adyacentes en bloques para reducir llamadas TTS.

    Agrupacion dinamica segun pausa y puntuacion:
    - Pausa >=1.5s: pausa larga, corta grupo
    - Puntuacion fuerte (. ? !) + >=150 chars: oracion completa, corta
    - Pausa <0.8s: misma oracion, agrupa hasta ~250 chars
    - Pausa 0.8-1.5s: oracion cercana, agrupa hasta max_chars (230)

    Mantiene timestamps originales (start del primero, end del ultimo).
    Devuelve lista de grupos: {start, end, text, seg_indices}
    """
    groups = []
    current = None

    for idx, (seg, text) in enumerate(zip(segments, tts_texts)):
        text = text.strip()
        if not text:
            continue

        if current is None:
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "seg_indices": [idx],
            }
        else:
            gap = seg["start"] - current["end"]
            combined_len = len(current["text"]) + 1 + len(text)

            # El texto actual del grupo termina en puntuacion fuerte?
            ends_sentence = current["text"].rstrip().endswith((".", "?", "!"))

            # Limite dinamico segun pausa y puntuacion
            if gap >= 1.5:
                limit = 0                # pausa larga: forzar corte
            elif ends_sentence and len(current["text"]) >= 150:
                limit = 0                # oracion completa + buen tamaño: cortar
            elif gap < 0.8:
                limit = max_chars + 20   # ~250: misma oracion, agrupa un poco mas
            else:
                limit = max_chars        # ~230: oracion cercana

            if combined_len <= limit and gap <= max_gap:
                current["text"] += " " + text
                current["end"] = seg["end"]
                current["seg_indices"].append(idx)
            else:
                groups.append(current)
                current = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "seg_indices": [idx],
                }

    if current:
        groups.append(current)

    return groups


def _translate_batched_with_context(
    all_texts: list[str],
    from_code: str,
    to_code: str,
    api_key: str,
    model_name: str,
    batch_size: int = _TRANSLATION_BATCH_SIZE,
    progress_fn=None,
    progress_base: float = 0.0,
    progress_range: float = 0.1,
) -> tuple[list[str], list[str]]:
    """Traduce segmentos en lotes, pasando contexto del lote anterior para coherencia."""
    if len(all_texts) <= batch_size:
        return translate_batch_gemini(all_texts, from_code, to_code, api_key, model_name)

    all_display = []
    all_tts = []
    prev_context = ""
    n_batches = (len(all_texts) + batch_size - 1) // batch_size

    for bi, i in enumerate(range(0, len(all_texts), batch_size)):
        batch = all_texts[i:i + batch_size]
        if progress_fn:
            progress_fn(
                progress_base + progress_range * bi / n_batches,
                desc=f"Traduciendo lote {bi+1}/{n_batches} ({len(batch)} segmentos)...",
            )
        print(f"  [Traduccion] Lote {bi+1}/{n_batches}: {len(batch)} segmentos"
              + (f" (con contexto)" if prev_context else ""))
        display, tts = translate_batch_gemini(
            batch, from_code, to_code, api_key, model_name, prev_context
        )
        all_display.extend(display)
        all_tts.extend(tts)
        # Guardar ultimas 2 oraciones como contexto para el siguiente lote
        prev_context = " ".join(display[-2:]) if len(display) >= 2 else " ".join(display)

    return all_display, all_tts


def pipeline_largo(
    audio_file: str | None,
    source_lang: str,
    target_lang: str,
    exaggeration: float,
    sync_duration: bool,
    gpu_agresivo: bool,
    api_key: str,
    gemini_model: str,
    tts_provider_name: str,
    elevenlabs_key: str = "",
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """
    Pipeline optimizado para audio largo (>3 min).

    Arquitectura:
    1. Whisper COMPLETO (1 sola pasada, sin cortar audio)
    2. Traduccion en lotes con contexto entre lotes
    3. TTS por segmento con checkpoint/resume
    4. Ensamblaje final con pausas y ritmo
    """
    import soundfile as sf
    import numpy as np
    import scipy.io.wavfile as wav_io
    from datetime import datetime

    pipeline_start = time.time()
    if audio_file is None:
        raise gr.Error("Sube un archivo de audio primero.")
    if not api_key.strip():
        raise gr.Error("Ingresa tu clave de API de Google AI Studio.")
    if tts_provider_name == "elevenlabs" and not elevenlabs_key.strip():
        raise gr.Error("Ingresa tu clave API de ElevenLabs en la seccion Sintesis de Voz.")

    _cancel_flag.clear()

    try:
        wav_file, is_tmp = ensure_wav(audio_file)
    except Exception as exc:
        raise gr.Error(f"No se pudo leer el audio: {exc}") from exc

    try:
        audio_duration = _get_audio_duration(wav_file)
        dur_min = int(audio_duration // 60)
        dur_sec = int(audio_duration % 60)

        # Crear directorio de sesion para estado/resume
        file_hash = _hash_file(wav_file)
        session_dir = OUTPUT_DIR / f"_session_{file_hash[:12]}"
        session_dir.mkdir(parents=True, exist_ok=True)
        state_path = session_dir / "_state.json"
        state = _load_state(state_path)

        print(f"\n{'='*60}")
        print(f"  [LARGO] Audio: {dur_min}m {dur_sec}s ({audio_duration:.0f}s)")
        print(f"  [LARGO] Sesion: {session_dir.name}")
        if state:
            print(f"  [LARGO] Reanudando sesion anterior...")
        print(f"{'='*60}")

        # == 1. WHISPER COMPLETO (1 sola pasada) ==============================
        if "segments" in state and "original_text" in state:
            segments = state["segments"]
            original_text = state["original_text"]
            detected = state["detected_lang"]
            print(f"  [1/3] Transcripcion reanudada del estado ({len(segments)} segmentos)")
            progress(0.20, desc="Paso 1/3 -- Transcripcion recuperada!")
        else:
            progress(0.02, desc=f"Paso 1/3 -- Transcribiendo {dur_min}m {dur_sec}s con Whisper...")
            try:
                original_text, detected, segments = transcribe(wav_file)
            except Exception as exc:
                raise gr.Error(f"Error en transcripcion: {exc}") from exc
            state["original_text"] = original_text
            state["detected_lang"] = detected
            state["segments"] = segments
            _save_state(state_path, state)
            print(f"  [1/3] Whisper completo: {len(segments)} segmentos (guardado)")

        if not original_text or not segments:
            raise gr.Error("No se detecto habla en el audio.")

        src = detected if source_lang == "auto" else source_lang
        src_label = LANGUAGES.get(src, src)
        tgt_label = LANGUAGES.get(target_lang, target_lang)
        n_segs = len(segments)

        # == 2. TRADUCCION EN LOTES CON CONTEXTO =============================
        state_trans_key = f"translations_{target_lang}"
        if state_trans_key in state:
            display_texts = state[state_trans_key]["display"]
            tts_texts = state[state_trans_key]["tts"]
            print(f"  [2/3] Traduccion reanudada del estado ({len(display_texts)} segmentos)")
            progress(0.35, desc="Paso 2/3 -- Traduccion recuperada!")
        else:
            progress(0.22, desc=f"Paso 2/3 -- Traduciendo {n_segs} segmentos ({src_label} -> {tgt_label})...")
            try:
                texts = [s["text"] for s in segments]
                display_texts, tts_texts = _translate_batched_with_context(
                    texts, src, target_lang, api_key.strip(), gemini_model,
                    progress_fn=progress, progress_base=0.22, progress_range=0.13,
                )
            except Exception as exc:
                raise gr.Error(f"Error en traduccion: {exc}") from exc
            state[state_trans_key] = {"display": display_texts, "tts": tts_texts}
            _save_state(state_path, state)
            print(f"  [2/3] Traduccion completa: {len(display_texts)} segmentos (guardado)")

        # == 3. TTS POR SEGMENTO CON RESUME ===================================
        provider = _get_tts_provider(tts_provider_name)
        if isinstance(provider, ElevenLabsProvider) and elevenlabs_key:
            provider.set_api_key(elevenlabs_key)
        sample_rate = provider.sr
        if gpu_agresivo and DEVICE == "cuda":
            _activar_gpu_agresivo()

        # Preparar referencia de voz (max 15s)
        MAX_REF_SECONDS = 15
        ref_data, ref_sr = sf.read(wav_file, always_2d=False, dtype="float32")
        if ref_data.ndim > 1:
            ref_data = ref_data[:, 0]
        if len(ref_data) > MAX_REF_SECONDS * ref_sr:
            ref_data = ref_data[:MAX_REF_SECONDS * ref_sr]
        tmp_ref = str(session_dir / "_reference.wav")
        if not Path(tmp_ref).exists():
            sf.write(tmp_ref, ref_data, ref_sr, subtype="PCM_16")

        # Agrupar segmentos para reducir llamadas TTS
        groups = _group_segments_for_tts(segments, tts_texts)
        n_groups = len(groups)
        tts_state = state.get("tts_groups", {})
        done_from_cache = sum(1 for v in tts_state.values() if v.get("done"))
        if done_from_cache > 0:
            print(f"  [3/3] TTS: {done_from_cache}/{n_groups} grupos ya procesados (resume)")
        print(f"  [3/3] {n_segs} segmentos -> {n_groups} grupos TTS "
              f"(~{sum(len(g['text']) for g in groups)//n_groups} chars/grupo)")
        progress(0.38, desc=f"Paso 3/3 -- Sintetizando {n_groups} grupos...")

        PACING_FACTOR = 0.45
        original_duration = segments[-1]["end"] if segments else 0.0
        t0 = time.time()
        done = 0
        prev_end = 0.0
        audio_pieces = []
        total_tts_audio = 0.0
        total_extra_pauses = 0.0

        # Para ElevenLabs: pre-crear la voz clonada y luego paralelizar requests
        use_parallel = isinstance(provider, ElevenLabsProvider)
        if use_parallel:
            provider._ensure_voice(tmp_ref)
            print(f"  [ElevenLabs] Modo paralelo activado (4 requests simultaneas)")
        elif DEVICE == "cuda":
            free, total = torch.cuda.mem_get_info()
            print(f"  [GPU] VRAM: {free/1024**3:.1f} GB libre / {total/1024**3:.1f} GB total")

        def _generate_group_audio(group_data):
            """Genera audio para un grupo. Retorna (gi, wav_np, tts_dur)."""
            gi, group = group_data
            text_chunks = _split_long_text(group["text"], max_chars=_TTS_GROUP_MAX_CHARS + 70)
            chunk_wavs = []
            for chunk_text in text_chunks:
                if _cancel_flag.is_set():
                    raise RuntimeError("Cancelado por el usuario.")
                wav_chunk = provider.generate(chunk_text, tmp_ref, exaggeration)
                chunk_wavs.append(wav_chunk)
            if len(chunk_wavs) > 1:
                breath = np.zeros(int(sample_rate * 0.10), dtype=np.float32)
                pieces = [chunk_wavs[0]]
                for cw in chunk_wavs[1:]:
                    pieces.append(breath)
                    pieces.append(cw)
                wav_np = np.concatenate(pieces)
            else:
                wav_np = chunk_wavs[0]
            return gi, wav_np, len(wav_np) / sample_rate

        try:
            if use_parallel:
                # == MODO PARALELO (ElevenLabs API) ============================
                from concurrent.futures import ThreadPoolExecutor, as_completed

                # Separar grupos ya procesados vs pendientes
                pending = []
                for gi, group in enumerate(groups):
                    grp_key = str(gi)
                    grp_file = session_dir / f"grp_{gi:04d}.wav"
                    if grp_key in tts_state and tts_state[grp_key].get("done") and grp_file.exists():
                        continue
                    pending.append((gi, group))

                # Generar en paralelo (max 4 concurrent requests)
                generated = {}  # gi -> (wav_np, tts_dur)
                if pending:
                    n_pending = len(pending)
                    print(f"  [ElevenLabs] {n_pending} grupos pendientes, generando en paralelo...")
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {executor.submit(_generate_group_audio, item): item[0] for item in pending}
                        for future in as_completed(futures):
                            if _cancel_flag.is_set():
                                executor.shutdown(wait=False, cancel_futures=True)
                                raise RuntimeError("Cancelado por el usuario.")
                            gi_done, wav_np, tts_dur = future.result()
                            generated[gi_done] = (wav_np, tts_dur)
                            # Guardar para resume
                            grp_file = session_dir / f"grp_{gi_done:04d}.wav"
                            sf.write(str(grp_file), wav_np, sample_rate, subtype="FLOAT")
                            tts_state[str(gi_done)] = {"done": True, "duration": tts_dur}
                            done += 1
                            elapsed = time.time() - t0
                            n_s = len(groups[gi_done]["seg_indices"])
                            print(f"  [TTS {done}/{n_groups}] grupo {gi_done} "
                                  f"({elapsed:.0f}s) {n_s} segs | {groups[gi_done]['text'][:60]!r}"
                                  f" -> {tts_dur:.1f}s")
                            if done % 4 == 0:
                                state["tts_groups"] = tts_state
                                _save_state(state_path, state)
                                pct = 0.38 + 0.55 * (done / max(n_groups, 1))
                                progress(pct, desc=f"TTS {done}/{n_groups} (paralelo)...")

                # Ensamblar en orden
                for gi, group in enumerate(groups):
                    gap = max(0.0, group["start"] - prev_end)
                    if gap > 0.02:
                        audio_pieces.append(np.zeros(int(gap * sample_rate), dtype=np.float32))
                    prev_end = group["end"]

                    grp_key = str(gi)
                    grp_file = session_dir / f"grp_{gi:04d}.wav"

                    if gi in generated:
                        wav_np, tts_dur = generated[gi]
                    elif grp_key in tts_state and tts_state[grp_key].get("done") and grp_file.exists():
                        grp_audio, _ = sf.read(str(grp_file), dtype="float32")
                        wav_np = grp_audio
                        tts_dur = tts_state[grp_key]["duration"]
                    else:
                        continue

                    audio_pieces.append(wav_np)
                    total_tts_audio += tts_dur
                    orig_grp_dur = group["end"] - group["start"]
                    time_saved = max(0.0, orig_grp_dur - tts_dur)
                    extra = time_saved * PACING_FACTOR
                    if extra > 0.02:
                        audio_pieces.append(np.zeros(int(extra * sample_rate), dtype=np.float32))
                        total_extra_pauses += extra

                done = n_groups

            else:
                # == MODO SECUENCIAL (Chatterbox GPU) ==========================
                for gi, group in enumerate(groups):
                    if _cancel_flag.is_set():
                        state["tts_groups"] = tts_state
                        _save_state(state_path, state)
                        raise RuntimeError("Cancelado por el usuario.")

                    grp_key = str(gi)
                    grp_file = session_dir / f"grp_{gi:04d}.wav"

                    # Silencio original antes de este grupo
                    gap = max(0.0, group["start"] - prev_end)
                    if gap > 0.02:
                        audio_pieces.append(np.zeros(int(gap * sample_rate), dtype=np.float32))
                    prev_end = group["end"]

                    done += 1
                    elapsed = time.time() - t0

                    # Verificar si ya fue sintetizado (resume)
                    if grp_key in tts_state and tts_state[grp_key].get("done") and grp_file.exists():
                        grp_audio, _ = sf.read(str(grp_file), dtype="float32")
                        tts_dur = tts_state[grp_key]["duration"]
                        audio_pieces.append(grp_audio)
                        total_tts_audio += tts_dur
                        orig_grp_dur = group["end"] - group["start"]
                        time_saved = max(0.0, orig_grp_dur - tts_dur)
                        extra = time_saved * PACING_FACTOR
                        if extra > 0.02:
                            audio_pieces.append(np.zeros(int(extra * sample_rate), dtype=np.float32))
                            total_extra_pauses += extra
                        continue

                    # Generar TTS para el grupo (usar limite alto para no re-fragmentar)
                    text_chunks = _split_long_text(group["text"], max_chars=_TTS_GROUP_MAX_CHARS + 70)
                    n_s = len(group["seg_indices"])
                    frags = f" [{len(text_chunks)} fragmentos]" if len(text_chunks) > 1 else ""
                    print(f"  [TTS {done}/{n_groups}] {group['start']:.1f}s-{group['end']:.1f}s "
                          f"({elapsed:.0f}s) {n_s} segs | {group['text'][:80]!r}{frags}")

                    chunk_wavs = []
                    for ci, chunk_text in enumerate(text_chunks):
                        if _cancel_flag.is_set():
                            raise RuntimeError("Cancelado por el usuario.")
                        t_gen = time.time()
                        wav_chunk = provider.generate(chunk_text, tmp_ref, exaggeration)
                        gen_secs = time.time() - t_gen
                        chunk_dur = len(wav_chunk) / sample_rate
                        ratio = gen_secs / max(chunk_dur, 0.01)
                        print(f"    [gen {ci}] {gen_secs:.1f}s wall -> {chunk_dur:.1f}s audio ({ratio:.1f}x RT) | {len(chunk_text)} chars")
                        chunk_wavs.append(wav_chunk)

                        # Detectar throttling: si ratio > 2.0x, la GPU se esta sobrecalentando
                        if ratio > 2.0 and DEVICE == "cuda":
                            print(f"    ⚠️ GPU throttling detectado ({ratio:.1f}x RT). Pausando 3s para enfriar...")
                            torch.cuda.empty_cache()
                            time.sleep(3)

                    if len(chunk_wavs) > 1:
                        breath = np.zeros(int(sample_rate * 0.10), dtype=np.float32)
                        pieces = [chunk_wavs[0]]
                        for cw in chunk_wavs[1:]:
                            pieces.append(breath)
                            pieces.append(cw)
                        wav_np = np.concatenate(pieces)
                    else:
                        wav_np = chunk_wavs[0]
                    tts_dur = len(wav_np) / sample_rate
                    total_tts_audio += tts_dur
                    print(f"    -> {tts_dur:.1f}s generado")

                    # Guardar grupo para resume
                    sf.write(str(grp_file), wav_np, sample_rate, subtype="FLOAT")
                    tts_state[grp_key] = {"done": True, "duration": tts_dur}
                    if done % 3 == 0:
                        state["tts_groups"] = tts_state
                        _save_state(state_path, state)

                    # Limpiar cache CUDA cada 8 grupos para estabilizar VRAM
                    if done % 8 == 0 and DEVICE == "cuda":
                        torch.cuda.empty_cache()

                    audio_pieces.append(wav_np)

                    # Pacing basado en duracion original del grupo
                    orig_grp_dur = group["end"] - group["start"]
                    time_saved = max(0.0, orig_grp_dur - tts_dur)
                    extra = time_saved * PACING_FACTOR
                    if extra > 0.02:
                        audio_pieces.append(np.zeros(int(extra * sample_rate), dtype=np.float32))
                        total_extra_pauses += extra

                    # Progreso con estimacion
                    pct = 0.38 + 0.55 * (done / max(n_groups, 1))
                    remaining = (time.time() - t0) / done * (n_groups - done)
                    rem_min = int(remaining // 60)
                    rem_sec = int(remaining % 60)
                    progress(pct, desc=f"TTS {done}/{n_groups} (~{rem_min}m {rem_sec}s restantes)...")
        finally:
            state["tts_groups"] = tts_state
            _save_state(state_path, state)

        if not audio_pieces:
            raise RuntimeError("No se genero audio: los segmentos estaban vacios.")

        # == 4. ENSAMBLAJE FINAL ==============================================
        progress(0.94, desc="Ensamblando audio final...")

        if total_extra_pauses > 0.5:
            print(f"  [TTS] Pausas de ritmo: +{total_extra_pauses:.1f}s (factor {PACING_FACTOR})")

        combined = np.concatenate(audio_pieces)
        del audio_pieces

        total_elapsed = time.time() - t0
        tts_dur_total = len(combined) / sample_rate
        print(f"  [TTS] Completado: {tts_dur_total:.1f}s de audio en {total_elapsed:.0f}s "
              f"({total_elapsed/max(tts_dur_total,0.1):.1f}x tiempo real)")

        # Relleno final
        MAX_END_PAD = 3.0
        if tts_dur_total < original_duration:
            missing = original_duration - tts_dur_total
            pad_seconds = min(missing, MAX_END_PAD)
            if pad_seconds > 0.1:
                combined = np.concatenate([combined, np.zeros(int(pad_seconds * sample_rate), dtype=np.float32)])
                if missing > MAX_END_PAD:
                    print(f"  [TTS] Relleno final: +{pad_seconds:.1f}s (limitado; faltaban {missing:.1f}s)")
                else:
                    print(f"  [TTS] Relleno final: +{pad_seconds:.1f}s")
        elif tts_dur_total > original_duration:
            print(f"  [TTS] Audio mas largo: {tts_dur_total:.1f}s vs {original_duration:.1f}s (se mantiene)")

        final_dur = len(combined) / sample_rate
        print(f"  [TTS] Duracion final: {final_dur:.1f}s (original: {original_duration:.1f}s)")

        # Escribir WAV
        combined_int16 = np.clip(combined * 32767, -32768, 32767).astype(np.int16)
        del combined
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(OUTPUT_DIR / f"{timestamp}_traducido-{target_lang}_largo.wav")
        wav_io.write(out_path, sample_rate, combined_int16)

        # Sincronizar duracion
        if sync_duration and _ffmpeg and segments and tts_dur_total > original_duration * 1.05:
            atempo = tts_dur_total / original_duration
            stretched = out_path.replace(".wav", "_sync.wav")
            filter_str = _build_atempo_filter(atempo)
            try:
                subprocess.run(
                    [_ffmpeg, "-y", "-i", out_path, "-filter:a", filter_str, stretched],
                    check=True, capture_output=True,
                )
                Path(out_path).unlink()
                shutil.move(stretched, out_path)
            except Exception:
                pass

        # MP3 para navegador
        mp3_path = out_path.replace(".wav", ".mp3")
        if _ffmpeg:
            try:
                subprocess.run(
                    [_ffmpeg, "-y", "-i", out_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
                    check=True, capture_output=True,
                )
                mp3_size = Path(mp3_path).stat().st_size / 1024
                wav_size = Path(out_path).stat().st_size / 1024
                print(f"  [TTS] MP3: {mp3_size:.0f}KB (WAV: {wav_size:.0f}KB)")
                final_output = mp3_path
            except Exception:
                final_output = out_path
        else:
            final_output = out_path

        # Limpiar archivos de grupos
        for grp_f in session_dir.glob("grp_*.wav"):
            try:
                grp_f.unlink()
            except OSError:
                pass
        try:
            Path(tmp_ref).unlink()
        except OSError:
            pass

        state["completed"] = True
        _save_state(state_path, state)

        total_time = time.time() - pipeline_start
        t_min = int(total_time // 60)
        t_sec = int(total_time % 60)
        time_str = f"{t_min}m {t_sec}s" if t_min > 0 else f"{t_sec}s"
        print(f"\n{'='*60}")
        print(f"  [LARGO] Pipeline completo en {time_str} ({n_segs} segmentos -> {n_groups} grupos TTS)")
        print(f"{'='*60}")
        progress(1.0, desc=f"Listo en {time_str}! {n_groups} grupos TTS")

        if target_lang in CJK_LANGS and display_texts != tts_texts:
            trans_display = f"[{tgt_label}]  {' '.join(display_texts)}\n[Romanizado]  {' '.join(tts_texts)}"
        else:
            trans_display = f"[{tgt_label}]  {' '.join(display_texts)}"
        trans_display += f"\n\nTiempo total: {time_str} | {n_segs} segmentos ({n_groups} grupos TTS)"

        return (
            f"[{src_label}]  {original_text}",
            trans_display,
            final_output,
        )
    finally:
        # Limpiar voz clonada de ElevenLabs si se uso
        provider = TTS_PROVIDERS.get(tts_provider_name)
        if isinstance(provider, ElevenLabsProvider):
            provider.cleanup_voice()
        if is_tmp:
            try:
                Path(wav_file).unlink()
            except OSError:
                pass


def pipeline_auto(
    audio_file: str | None,
    source_lang: str,
    target_lang: str,
    exaggeration: float,
    sync_duration: bool,
    gpu_agresivo: bool,
    api_key: str,
    gemini_model: str,
    tts_provider_name: str,
    elevenlabs_key: str = "",
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """Selecciona automaticamente pipeline normal o largo segun duracion."""
    if audio_file is None:
        raise gr.Error("Sube un archivo de audio primero.")

    try:
        wav_tmp, is_tmp = ensure_wav(audio_file)
        duration = _get_audio_duration(wav_tmp)
        if is_tmp:
            Path(wav_tmp).unlink(missing_ok=True)
    except Exception:
        duration = 0

    if duration > _LONG_AUDIO_THRESHOLD:
        print(f"  [AUTO] Audio largo ({duration:.0f}s > {_LONG_AUDIO_THRESHOLD}s) -> pipeline largo")
        return pipeline_largo(
            audio_file, source_lang, target_lang, exaggeration,
            sync_duration, gpu_agresivo, api_key, gemini_model,
            tts_provider_name, elevenlabs_key, progress,
        )
    else:
        return pipeline(
            audio_file, source_lang, target_lang, exaggeration,
            sync_duration, gpu_agresivo, api_key, gemini_model,
            tts_provider_name, elevenlabs_key, progress,
        )

# -- Interfaz Gradio -----------------------------------------------------------
# ==============================================================================

src_choices = [("Detectar automaticamente", "auto")] + [(name, code) for code, name in LANGUAGES.items()]
tgt_choices = [(name, code) for code, name in LANGUAGES.items()]

_APP_CSS = """
/* Audio player: timestamps siempre visibles */
.timestamps {
    font-size: 14px !important;
    padding: 4px 0 !important;
}
.timestamps time {
    min-width: 55px !important;
    font-variant-numeric: tabular-nums !important;
}
"""

_APP_HEAD = """
<script>
/* Tooltip flotante de tiempo sobre el waveform del audio */
function _addAudioTips() {
    document.querySelectorAll('audio').forEach(function(aud) {
        var wrap = aud.parentElement;
        for (var i = 0; i < 6 && wrap; i++) {
            if (wrap.querySelector('canvas')) break;
            wrap = wrap.parentElement;
        }
        if (!wrap || wrap._hasTip) return;
        var cv = wrap.querySelector('canvas');
        if (!cv) return;
        var wv = cv.parentElement;
        wrap._hasTip = true;
        wv.style.position = 'relative';
        var tip = document.createElement('div');
        tip.style.cssText = 'position:absolute;top:-30px;background:rgba(30,30,30,.92);'
            + 'color:#fff;padding:4px 10px;border-radius:6px;font-size:13px;'
            + 'display:none;pointer-events:none;z-index:9999;white-space:nowrap;'
            + 'transform:translateX(-50%);box-shadow:0 2px 8px rgba(0,0,0,.3);'
            + 'font-variant-numeric:tabular-nums;';
        wv.appendChild(tip);
        wv.addEventListener('mousemove', function(e) {
            var r = wv.getBoundingClientRect();
            var pct = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
            var dur = aud.duration;
            if (dur && isFinite(dur)) {
                var t = pct * dur;
                var m = Math.floor(t / 60);
                var s = Math.floor(t % 60);
                tip.textContent = m + ':' + (s < 10 ? '0' : '') + s;
                tip.style.left = (e.clientX - r.left) + 'px';
                tip.style.display = 'block';
            }
        });
        wv.addEventListener('mouseleave', function() { tip.style.display = 'none'; });
    });
}
setInterval(_addAudioTips, 2000);
</script>
"""

with gr.Blocks(title="Traductor de Voz Local") as demo:
    gr.Markdown(
        "# Traductor de Voz Local\n"
        "Sube tu audio y lo traduce a otro idioma **conservando tu tono de voz**.\n\n"
        "> Traduccion con **Google Gemini Flash** (API gratuita). "
        "Obtene tu clave gratis en [aistudio.google.com](https://aistudio.google.com). "
        "Los audios se guardan automaticamente en la carpeta `outputs/`."
    )

    # -- Configuracion: Traduccion (Gemini) --
    with gr.Accordion("Traduccion -- Gemini API", open=not bool(_load_gemini_key())):
        with gr.Row():
            api_key_tb = gr.Textbox(
                label="Clave API de Google AI Studio",
                placeholder="AIzaSy...",
                type="password",
                value=_load_gemini_key(),
                scale=3,
            )
            save_key_btn = gr.Button("Guardar clave", scale=1)
        with gr.Row():
            gemini_model_dd = gr.Dropdown(
                label="Modelo Gemini",
                choices=GEMINI_MODELS,
                value=GEMINI_DEFAULT,
                scale=2,
                allow_custom_value=True,
            )
            test_gemini_btn = gr.Button("Probar conexion", variant="secondary", scale=1)
            detect_btn = gr.Button("Detectar modelos", variant="secondary", scale=1)
        gemini_status = gr.Markdown("")

    # -- Configuracion: Sintesis de Voz (TTS) --
    with gr.Accordion("Sintesis de Voz -- TTS", open=False):
        with gr.Row():
            tts_provider_dd = gr.Dropdown(
                label="Proveedor TTS",
                choices=TTS_PROVIDER_CHOICES,
                value=DEFAULT_TTS_PROVIDER,
                scale=2,
            )
            test_tts_btn = gr.Button("Probar TTS", variant="secondary", scale=1)
        with gr.Row():
            elevenlabs_key_tb = gr.Textbox(
                label="Clave API de ElevenLabs (solo si usas ElevenLabs)",
                placeholder="sk_...",
                type="password",
                value=_load_elevenlabs_key(),
                scale=3,
            )
            save_el_key_btn = gr.Button("Guardar clave", scale=1)
        tts_status = gr.Markdown("")

    # -- Panel principal --
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                label="Sube tu archivo de audio",
                sources=["upload"],
                type="filepath",
                elem_id="audio-input",
            )
            with gr.Row():
                src_dd = gr.Dropdown(
                    label="Idioma de origen",
                    choices=src_choices,
                    value="auto",
                )
                tgt_dd = gr.Dropdown(
                    label="Idioma destino",
                    choices=tgt_choices,
                    value="en",
                )
            with gr.Accordion("Opciones avanzadas", open=False):
                exag_sl = gr.Slider(
                    minimum=0.25, maximum=1.5, value=0.5, step=0.05,
                    label="Expresividad de voz (0.25=neutro, 0.5=natural, 1.0=expresivo)",
                )
                sync_dur_cb = gr.Checkbox(
                    value=False,
                    label="Sincronizar duracion con el original (atempo)",
                    info="Si el audio traducido es mas largo, lo acelera para igualar duracion. Puede sonar robotico.",
                )
                gpu_agresivo_cb = gr.Checkbox(
                    value=False,
                    label="Modo GPU agresivo (experimental)",
                    info="Activa cudnn.benchmark para buscar algoritmos optimos. Requiere GPU con >=10GB VRAM.",
                )
            with gr.Accordion("Flujo del pipeline", open=False):
                # Mensaje dinamico segun GPU detectada
                if _GPU_VRAM_GB >= 10:
                    _gpu_rec = (
                        f"**Tu GPU ({_GPU_NAME}, {_GPU_VRAM_GB:.0f}GB):** "
                        "Puedes activarlo. Tienes suficiente VRAM. "
                        "La primera traduccion sera mas lenta (calibra algoritmos), las siguientes mas rapidas."
                    )
                elif _GPU_VRAM_GB >= 6:
                    _gpu_rec = (
                        f"**Tu GPU ({_GPU_NAME}, {_GPU_VRAM_GB:.0f}GB):** "
                        "No se recomienda activarlo. Tu VRAM esta justo en el limite "
                        "y el modelo TTS ya consume ~4-5GB. Podria causar errores de memoria o empeorar el rendimiento. "
                        "Dejalo desactivado para uso estable."
                    )
                elif DEVICE == "cuda":
                    _gpu_rec = (
                        f"**Tu GPU ({_GPU_NAME}, {_GPU_VRAM_GB:.0f}GB):** "
                        "NO activar. Tu VRAM es insuficiente y causara errores CUDA Out of Memory."
                    )
                else:
                    _gpu_rec = "**Sin GPU CUDA detectada.** El modo agresivo no aplica."

                gr.Markdown(
                    "| Paso | Que hace | 30s audio | 2 min audio | Cuello de botella |\n"
                    "|------|----------|-----------|-------------|-------------------|\n"
                    "| **1. Whisper** | Transcripcion con timestamps | ~5s | ~20s | Bajo |\n"
                    "| **2. Gemini** | Traduccion (1 llamada API) | ~2s | ~3s | Minimo |\n"
                    "| **3. TTS** | Sintesis voz por segmento | ~25-50s | ~1.5-2.5 min | **PRINCIPAL** ~1.0x |\n"
                    "| **4. Mezcla** | Pausas + escritura WAV int16 | <1s | <1s | Ninguno |\n\n"
                    "**Rendimiento actual:** ~1.0x tiempo real (2 min de audio = ~2 min de TTS).\n\n"
                    "**Notas:**\n"
                    "- Cada segmento se sintetiza **individualmente** preservando pausas originales.\n"
                    "- Las pausas se ajustan **proporcionalmente** al ritmo original.\n"
                    "- Textos largos (>150 chars) se **dividen** en fragmentos para mejor calidad TTS.\n"
                    "- Para **japones, chino, coreano**: romanizacion automatica para TTS.\n"
                    "- Si traduces el **mismo audio** a otro idioma, la transcripcion se reutiliza del cache.\n"
                    "- El TTS tarda menos si la GPU ya esta caliente (segunda traduccion mas rapida).\n\n"
                    "---\n\n"
                    "**Audio largo (>3 minutos):**\n\n"
                    "Se activa automaticamente un **pipeline optimizado**:\n"
                    "- **Whisper transcribe TODO** el audio en 1 sola pasada (sin cortes)\n"
                    "- La traduccion se hace **en lotes con contexto** entre cada lote (sin perder coherencia)\n"
                    "- El TTS sintetiza **segmento por segmento** con checkpoint\n"
                    "- Si se interrumpe, al reanudar **salta lo ya procesado** (transcripcion, traduccion, y cada segmento TTS)\n"
                    "- Ideal para videos de **5-30+ minutos**\n\n"
                    "---\n\n"
                    "**Modo GPU agresivo (experimental):**\n\n"
                    "| Optimizacion | Que hace | Requisito |\n"
                    "|-------------|----------|-----------|"
                    "\n| `cudnn.benchmark` | Auto-selecciona el mejor algoritmo de convolucion por cada shape | >=10GB VRAM |\n\n"
                    "**Cuando activarlo:**\n"
                    "- Tienes una GPU con **10GB VRAM o mas** (RTX 3080, 4070 Ti, 4080, 4090, etc.)\n"
                    "- Quieres exprimir el ultimo ~5-10% de velocidad\n"
                    "- La primera traduccion sera **mas lenta** (calibra algoritmos), las siguientes mas rapidas\n\n"
                    "**Riesgos si tu GPU NO cumple el requisito:**\n"
                    "- **CUDA Out of Memory**: la app se cierra con error de memoria. Solucion: desactiva la opcion y reinicia.\n"
                    "- **Rendimiento peor**: en GPUs con poca VRAM, el benchmark recompila en cada shape distinto, causando latencia extra.\n"
                    "- **Inestabilidad**: en GPUs pre-Turing puede causar crashes.\n\n"
                    + _gpu_rec
                )
            with gr.Row():
                btn = gr.Button("Traducir", variant="primary", scale=3)
                stop_btn = gr.Button("Cancelar", variant="stop", scale=1)

        with gr.Column(scale=1):
            orig_box = gr.Textbox(label="Original (transcrito)", interactive=False)
            trans_box = gr.Textbox(label="Texto traducido", interactive=False)
            audio_out = gr.Audio(
                label="Resultado (guardado en outputs/)",
                type="filepath",
                interactive=False,
                elem_id="audio-output",
            )
            with gr.Row():
                mp3_btn = gr.Button("Guardar como MP3", variant="secondary")
            mp3_file = gr.File(label="Descarga MP3", visible=False)

    # -- Eventos --
    save_key_btn.click(fn=_save_gemini_key, inputs=[api_key_tb], outputs=[gemini_status])
    save_el_key_btn.click(fn=_save_elevenlabs_key, inputs=[elevenlabs_key_tb], outputs=[tts_status])
    test_gemini_btn.click(fn=test_gemini_connection, inputs=[api_key_tb, gemini_model_dd], outputs=[gemini_status])
    detect_btn.click(fn=list_gemini_models, inputs=[api_key_tb], outputs=[gemini_model_dd, gemini_status])
    test_tts_btn.click(fn=test_tts_connection, inputs=[tts_provider_dd, elevenlabs_key_tb], outputs=[tts_status])

    event = btn.click(
        fn=pipeline_auto,
        inputs=[audio_in, src_dd, tgt_dd, exag_sl, sync_dur_cb, gpu_agresivo_cb,
                api_key_tb, gemini_model_dd, tts_provider_dd, elevenlabs_key_tb],
        outputs=[orig_box, trans_box, audio_out],
    )
    stop_btn.click(fn=lambda: _cancel_flag.set(), inputs=None, outputs=None, cancels=[event])
    mp3_btn.click(fn=save_as_mp3, inputs=[audio_out], outputs=[mp3_file])


if __name__ == "__main__":
    print("Cargando modelos antes de abrir la app...")
    print("  [1/2] Whisper (transcripcion)...")
    get_whisper()
    print("  [2/2] Chatterbox TTS (clonacion de voz, primera vez descarga ~1.5 GB)...")
    get_tts()
    key = _load_gemini_key()
    if key:
        print(f"  Clave Gemini configurada: {'*' * 8}{key[-4:] if len(key) > 4 else '****'}")
    else:
        print("  AVISO: No hay clave Gemini. Ingresala en la seccion de Configuracion.")
    print("Todo listo. Abriendo app...")
    demo.queue().launch(
        inbrowser=True,
        allowed_paths=[str(OUTPUT_DIR)],
        css=_APP_CSS,
        head=_APP_HEAD,
    )
