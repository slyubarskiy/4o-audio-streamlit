# Shared Components

Reusable utilities and business logic components shared between v1 (Streamlit) and v2 (FastAPI) implementations.

## Components

### Error Handling (`api_error_utils.py`)
- Exponential backoff retry logic
- Circuit breaker pattern implementation
- Timeout management
- Response validation
- Error classification (retryable vs non-retryable)

### Metrics Tracking (`api_metrics_logger.py`)
- Transcription metrics collection
- Performance analysis and reporting
- Cost estimation
- Real-time metrics dashboard

### Audio Processing
- **`audio_config.py`** - Audio backend configuration
- **`audio_processor.py`** - Audio resampling and processing
- **`audio_format_utils.py`** - WAV format detection and preservation

## Usage

These components are automatically available to both v1 and v2 applications through the UV workspace configuration.

```python
from shared_components import (
    api_retry,
    APIMetricsLogger,
    resample_audio_auto,
    # ... other imports
)
```

## Development

When adding new shared functionality:
1. Add the module to this package
2. Update `__init__.py` with appropriate exports
3. Ensure compatibility with both Streamlit and FastAPI contexts
4. Add appropriate tests in `tests/shared/`

## Dependencies

See `pyproject.toml` for the minimal dependency set required by shared components.