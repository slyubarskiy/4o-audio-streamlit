"""
Audio processing configuration module
Centralizes audio processing settings and backend selection
"""

import os
import logging

logger = logging.getLogger(__name__)

# Target audio format specifications
# GPT-4o audio models are optimized for 16kHz mono PCM16 audio
TARGET_SAMPLE_RATE = 16000  # Optimal sample rate for GPT-4o audio
TARGET_BIT_DEPTH = "PCM_16"  # 16-bit signed PCM, little-endian
FORCE_MONO = True  # Convert multi-channel audio to mono (required for GPT-4o)

# Resampling backend selection
# Options: "ffmpeg" (faster, lower memory) or "librosa" (higher quality, more control)
RESAMPLING_BACKEND = os.getenv("AUDIO_RESAMPLING_BACKEND", "ffmpeg")  # Default to ffmpeg

# FFmpeg configuration
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")  # Path to ffmpeg executable
FFMPEG_QUALITY = "high"  # Options: "low", "medium", "high"

# FFmpeg quality presets
FFMPEG_QUALITY_PRESETS = {
    "low": {
        "audio_codec": "pcm_s16le",
        "bitrate": None,  # Use default for PCM
        "compression_level": None,
    },
    "medium": {
        "audio_codec": "pcm_s16le",
        "bitrate": None,
        "compression_level": None,
    },
    "high": {
        "audio_codec": "pcm_s16le",  # Lossless PCM for highest quality
        "bitrate": None,
        "compression_level": None,
    }
}

# Librosa configuration
LIBROSA_RESAMPLE_TYPE = "kaiser_best"  # Options: "kaiser_best", "kaiser_fast", "scipy", "polyphase"

# Timeout settings
FFMPEG_TIMEOUT = 30  # seconds
PROCESSING_TIMEOUT = 60  # seconds for entire processing pipeline

# Logging settings
LOG_AUDIO_METRICS = True
LOG_PROCESSING_TIME = True

# Fallback behavior
FALLBACK_TO_LIBROSA = True  # If ffmpeg fails, try librosa
FALLBACK_TO_ORIGINAL = True  # If all processing fails, use original audio

def get_ffmpeg_quality_params():
    """Get FFmpeg quality parameters based on current preset"""
    preset = FFMPEG_QUALITY_PRESETS.get(FFMPEG_QUALITY, FFMPEG_QUALITY_PRESETS["high"])
    return preset

def validate_backend(backend: str = None) -> str:
    """
    Validate and return the audio processing backend to use

    Args:
        backend: Override backend selection (optional)

    Returns:
        Valid backend name ("ffmpeg" or "librosa")
    """
    if backend is None:
        backend = RESAMPLING_BACKEND

    if backend not in ["ffmpeg", "librosa"]:
        logger.warning(f"Invalid backend '{backend}', defaulting to 'librosa'")
        return "librosa"

    if backend == "ffmpeg":
        # Check if ffmpeg is available
        import subprocess
        try:
            result = subprocess.run(
                [FFMPEG_PATH, "-version"],
                capture_output=True,
                timeout=5,
                check=False
            )
            if result.returncode != 0:
                logger.warning("FFmpeg not available, falling back to librosa")
                return "librosa" if FALLBACK_TO_LIBROSA else backend
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            logger.warning("FFmpeg not found or not accessible, falling back to librosa")
            return "librosa" if FALLBACK_TO_LIBROSA else backend

    return backend

def get_backend_info() -> dict:
    """Get information about the current audio processing configuration"""
    backend = validate_backend()
    return {
        "backend": backend,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "target_bit_depth": TARGET_BIT_DEPTH,
        "force_mono": FORCE_MONO,
        "ffmpeg_available": validate_backend("ffmpeg") == "ffmpeg",
        "ffmpeg_quality": FFMPEG_QUALITY if backend == "ffmpeg" else None,
        "librosa_resample_type": LIBROSA_RESAMPLE_TYPE if backend == "librosa" else None,
        "fallback_enabled": FALLBACK_TO_LIBROSA or FALLBACK_TO_ORIGINAL,
    }

# Log configuration on module load
logger.info(f"Audio configuration loaded: backend={RESAMPLING_BACKEND}, "
           f"target_sr={TARGET_SAMPLE_RATE}, force_mono={FORCE_MONO}")