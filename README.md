# Traductor Local de Voz a Voz

Traduce audio de un idioma a otro **conservando tu tono de voz original** usando clonacion de voz local con GPU.

Pipeline: **Whisper** (transcripcion) → **Gemini Flash** (traduccion) → **Chatterbox TTS** (sintesis con tu voz)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Required-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Caracteristicas

- **Clonacion de voz local** — Tu voz traducida suena como tu, no como un robot
- **17 idiomas** soportados (Ingles, Espanol, Frances, Japones, Chino, Coreano, etc.)
- **Dos pipelines** optimizados:
  - Audio corto (<3 min): procesamiento directo por segmento
  - Audio largo (>3 min): Whisper completo + traduccion por lotes + TTS agrupado con resume
- **Resume automatico** — Si se interrumpe, continua desde donde quedo
- **Romanizacion CJK** — Japones (romaji), Chino (pinyin), Coreano (romanizado)
- **ElevenLabs** opcional — Sintesis en la nube con requests paralelos (4 simultaneos)
- **Interfaz web** con Gradio — Sube tu audio y descarga la traduccion

## Requisitos

| Componente | Minimo | Recomendado |
|---|---|---|
| **GPU** | NVIDIA RTX 3060 (8GB VRAM) | RTX 4060+ (8GB+) |
| **Python** | 3.11 | 3.11 |
| **RAM** | 8 GB | 16 GB |
| **Almacenamiento** | ~5 GB (modelos) | ~5 GB |
| **OS** | Windows 10/11 | Windows 10/11 |

> **Nota:** Se requiere GPU NVIDIA con CUDA. No es compatible con AMD ni Intel.

## Instalacion

### Automatica (recomendada)

```bash
# 1. Clona el repositorio
git clone https://github.com/SiliusJM/voice-translator.git
cd voice-translator

# 2. Ejecuta el instalador (detecta CUDA automaticamente)
install.bat
```

El instalador:
1. Verifica Python 3.11 y GPU NVIDIA
2. Crea un entorno virtual `.venv`
3. Instala dependencias + PyTorch con CUDA
4. Verifica que la GPU se detecta correctamente

### Manual

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Uso

### Iniciar la app

```bash
run.bat
```

O manualmente:

```bash
.venv\Scripts\activate
python app.py
```

Se abre en `http://127.0.0.1:7860`

### Configuracion necesaria

1. **Clave API de Gemini** (gratis):
   - Ve a [aistudio.google.com](https://aistudio.google.com)
   - Crea una clave API
   - Pegala en la seccion "Traduccion — Gemini API" de la app

2. **ElevenLabs** (opcional, de pago):
   - Solo si quieres usar TTS en la nube en vez de local
   - Obtiene tu clave en [elevenlabs.io](https://elevenlabs.io)

### Primer uso

El primer arranque descarga automaticamente:
- Whisper large-v3 (~1.5 GB)
- Chatterbox TTS (~1.5 GB)

Esto tarda varios minutos la primera vez. Las ejecuciones siguientes arrancan rapido.

## Arquitectura y Optimizaciones

### Pipeline corto (<3 min)

```
Audio → Whisper (transcripcion) → Gemini Flash (traduccion) → Chatterbox TTS (por segmento) → MP3
```

### Pipeline largo (>3 min)

```
Audio → Whisper COMPLETO → Traduccion en lotes (con contexto) → TTS agrupado → Resume/Checkpoint → MP3
```

### Optimizaciones implementadas

#### 1. Whisper completo (sin chunks)
- **Problema:** Dividir audio por tiempo (90s) cortaba oraciones a la mitad
- **Solucion:** Una sola pasada de Whisper sobre el audio completo
- **Impacto:** Eliminacion total de cortes de oracion

#### 2. Traduccion con contexto entre lotes
- **Problema:** Traducir segmentos individuales perdia coherencia narrativa
- **Solucion:** Lotes de 25 segmentos, pasando las ultimas 2 oraciones traducidas como contexto al siguiente lote
- **Impacto:** Traduccion mas coherente y natural

#### 3. Agrupacion dinamica de segmentos para TTS
- **Problema:** 46 segmentos individuales = 46 llamadas TTS con overhead de CUDA cada una
- **Solucion:** Agrupacion inteligente basada en:
  - Pausa >= 1.5s → Corte (cambio de tema)
  - Puntuacion fuerte (. ? !) + >= 150 chars → Corte (oracion completa)
  - Pausa < 0.8s → Agrupa hasta 250 chars (misma oracion)
  - Pausa 0.8-1.5s → Agrupa hasta 230 chars
- **Impacto:** 46 segmentos → ~22 grupos, reduce overhead ~50%

#### 4. Limites de caracteres por grupo (sweet spot)
- **Problema:** Grupos de 280-350 chars causaban regresion severa en Chatterbox (no-lineal con longitud)
- **Solucion:** `_TTS_GROUP_MAX_CHARS = 230` como sweet spot para RTX 5060 (8GB)
- **Impacto:** 14 min → ~6 min para audio de 5 min

#### 5. Anti-refragmentacion
- **Problema:** `_split_long_text` con limite de 150 chars re-fragmentaba grupos de 230 chars en 2+ fragmentos
- **Solucion:** Limite de split en pipeline largo = `_TTS_GROUP_MAX_CHARS + 70` (300 chars)
- **Impacto:** Mantiene la agrupacion intacta

#### 6. Micro-pausas de respiracion
- **Solucion:** 100ms de silencio entre fragmentos dentro del mismo grupo
- **Impacto:** Audio mas natural sin afectar velocidad

#### 7. Sistema de resume con checkpoints
- Cada grupo TTS se guarda como WAV individual (`grp_XXXX.wav`)
- Estado guardado en JSON (`_state.json`) con transcripcion, traduccion y progreso TTS
- Si se interrumpe, reanuda desde el ultimo grupo completado

#### 8. ElevenLabs paralelo
- **Problema:** Requests HTTP secuenciales = lento con API cloud
- **Solucion:** `ThreadPoolExecutor` con 4 workers simultaneos
- **Impacto:** ~4x mas rapido que secuencial para ElevenLabs

#### 9. Optimizaciones de GPU
- TF32 habilitado (matmul + cuDNN) para RTX 30xx/40xx/50xx
- Warmup TTS al arrancar (compila kernels CUDA)
- `cudnn.benchmark` opcional (checkbox GPU Agresivo)
- `torch.compile` removido intencionalmente (causa regresion 2x en 8GB VRAM)

#### 10. Pacing y ritmo
- Factor de pacing 0.45: pausa proporcional al tiempo ahorrado por segmento
- Relleno final limitado a 3s maximo
- Atempo opcional: acelera audio si excede duracion original (ffmpeg)

### Rendimiento tipico (RTX 5060, 8GB VRAM)

| Audio | Segmentos | Grupos TTS | Tiempo | Ratio |
|---|---|---|---|---|
| 4m 55s | 46 | ~22 | ~5-7 min | 1.0-1.3x |
| 4m 55s | 46 | ~22 | ~12-13 min | 2.8x (GPU throttled) |

> **Nota:** El rendimiento de Chatterbox TTS varia segun la temperatura de la GPU,
> procesos en segundo plano y VRAM disponible. Reiniciar la app entre traducciones
> largas puede mejorar el rendimiento. La app incluye logs de timing por llamada TTS
> para diagnosticar cuellos de botella.

### Diagnostico de rendimiento: Thermal Throttling

Cuando el rendimiento empieza bien (~1.0x) pero se degrada con el tiempo (~2.5x+), **el problema no es el codigo — es la temperatura de la GPU**.

**Patron tipico de thermal throttling:**

| Grupo | Ratio |
|---|---|
| 1–5 | ~1.0x |
| 6–10 | ~1.5x |
| 10+ | ~2.5x |

Si ves este patron en los logs de timing, tu GPU se esta sobrecalentando.

#### Checklist de test controlado

1. **Reinicio limpio** — Reinicia tu PC (no solo la app)
2. **Antes de correr**, ejecuta `nvidia-smi` en terminal y verifica:
   - 0% GPU usage
   - VRAM baja (<500 MB ideal)
3. **Corre SOLO la app** — Sin Chrome, Discord, VSCode pesado, ni otros programas GPU
4. **Observa el patron** — Si el ratio sube con cada grupo, es throttling confirmado

#### Soluciones

| Solucion | Dificultad | Efectividad |
|---|---|---|
| Cerrar programas en segundo plano | Facil | Alta |
| Elevar laptop / base con ventiladores | Facil | Media-Alta |
| Reducir `_TTS_GROUP_MAX_CHARS` (230 → 200) | Facil | Media |
| Limitar potencia GPU (MSI Afterburner / NVIDIA Panel) | Media | Alta |
| La app limpia cache CUDA automaticamente cada 8 grupos | Automatico | Media |
| La app detecta throttling y pausa 3s para enfriar la GPU | Automatico | Media |

> **Importante:** El checkbox "GPU Agresivo" (`cudnn.benchmark`) puede **empeorar** el throttling
> porque genera mas calor. Usalo solo si tienes buena refrigeracion.

#### Lectura clave

> *"Cuando el rendimiento cae con el tiempo, casi nunca es codigo — es hardware."*

## Estructura del proyecto

```
voice-translator/
├── app.py              # Aplicacion principal (todo el pipeline)
├── install.bat         # Instalador automatico (detecta CUDA)
├── run.bat             # Lanzador de la app
├── requirements.txt    # Dependencias Python
├── .gitignore          # Archivos excluidos de Git
├── README.md           # Este archivo
└── outputs/            # Audios generados (no se sube a Git)
```

## Modelos utilizados

| Modelo | Proposito | Tamano | Ejecucion |
|---|---|---|---|
| [faster-whisper large-v3](https://github.com/SYSTRAN/faster-whisper) | Transcripcion | ~1.5 GB | GPU local |
| [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) | Clonacion de voz | ~1.5 GB | GPU local |
| [Gemini Flash](https://aistudio.google.com) | Traduccion | API | Nube (gratis) |
| [ElevenLabs](https://elevenlabs.io) (opcional) | TTS alternativo | API | Nube (pago) |

## Creditos

Desarrollado por **Siliu Xix**

---

*Si tienes problemas o sugerencias, abre un Issue en el repositorio.*
