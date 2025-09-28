"""
Shared components for GPT-4o Audio Transcription System.

This package contains reusable utilities and business logic components
that are shared between v1 (Streamlit) and v2 (FastAPI) implementations.
"""

from api_error_utils import (
    APIError,
    RetryableError,
    NonRetryableError,
    classify_error,
    api_retry,
    circuit_breaker,
    with_timeout,
    cache_on_error,
    validate_response,
    batch_api_call,
)

from api_metrics_logger import (
    TranscriptionMetrics,
    APIMetricsLogger,
    measure_audio_metrics,
    track_api_call,
    log_token_usage,
    create_performance_report,
)

from audio_config import (
    get_ffmpeg_quality_params,
    validate_backend,
    get_backend_info,
)

from audio_format_utils import (
    detect_wav_format,
    get_soundfile_subtype,
    preserve_format_params,
)

from audio_processor import (
    resample_audio_ffmpeg,
    resample_audio_librosa,
    resample_audio_auto,
)

__all__ = [
    # Error utilities
    'APIError',
    'RetryableError',
    'NonRetryableError',
    'classify_error',
    'api_retry',
    'circuit_breaker',
    'with_timeout',
    'cache_on_error',
    'validate_response',
    'batch_api_call',
    # Metrics
    'TranscriptionMetrics',
    'APIMetricsLogger',
    'measure_audio_metrics',
    'track_api_call',
    'log_token_usage',
    'create_performance_report',
    # Audio config
    'get_ffmpeg_quality_params',
    'validate_backend',
    'get_backend_info',
    # Audio format
    'detect_wav_format',
    'get_soundfile_subtype',
    'preserve_format_params',
    # Audio processor
    'resample_audio_ffmpeg',
    'resample_audio_librosa',
    'resample_audio_auto',
]