# GPT-4o Audio Transcription System - Project Documentation

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Architecture Map](#2-architecture-map)
3. [Module Documentation](#3-module-documentation)
4. [File Boundaries](#4-file-boundaries)
5. [API & Coding Standards](#5-api--coding-standards)
6. [Execution & CLI](#6-execution--cli)
7. [Testing Strategy](#7-testing-strategy)
8. [Extensibility & Integration](#8-extensibility--integration)
9. [Context Hygiene](#9-context-hygiene)
10. [Merge & Refactor Guidance](#10-merge--refactor-guidance)

---

## 1. System Overview

### 1.1 Purpose
The GPT-4o Audio Transcription System is a production-ready Streamlit web application that provides secure, scalable audio transcription services using OpenAI's GPT-4o audio preview model. The system emphasizes reliability through comprehensive error handling, performance monitoring, and secure authentication.

### 1.2 Core Capabilities
- **Audio Processing**: Real-time audio recording and transcription via web interface
- **Multi-segment Support**: Contextual transcription across multiple audio segments
- **Consolidated Analysis**: Intelligent summarization and reorganization of complete conversations
- **Language Detection**: Automatic language identification and transcription in source language
- **Performance Monitoring**: Real-time metrics tracking with cost estimation
- **Secure Access**: Google OAuth 2.0 authentication with email-based authorization

### 1.3 Use Cases
- **Meeting Transcription**: Record and transcribe business meetings with context preservation
- **Multi-language Support**: Process audio in various languages with automatic detection
- **Note Taking**: Quick voice-to-text conversion with intelligent organization
- **Conversation Analysis**: Generate structured summaries from multi-segment recordings
- **Research Documentation**: Capture and organize research interviews or field recordings

### 1.4 Environment & Dependencies
```
Runtime: Python 3.8+
Framework: Streamlit 1.x
AI Service: Azure OpenAI (GPT-4o audio preview)
Authentication: Google OAuth 2.0
Deployment: Render/Docker/Local
```

### 1.5 System Requirements
- **API Keys**: Azure OpenAI endpoint and key
- **OAuth Credentials**: Google Client ID and Secret
- **Network**: Stable internet connection for API calls
- **Browser**: Modern web browser with audio recording capabilities

---

## 2. Architecture Map

### 2.1 High-Level Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    User Interface (Browser)              │
│                         Streamlit UI                     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                    Main Application Layer                │
│                        app.py                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   OAuth      │  │   Audio      │  │  Transcript  │ │
│  │   Handler    │  │   Processor  │  │  Manager     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                    Service Layer                         │
│  ┌─────────────────────────┐  ┌──────────────────────┐ │
│  │   api_error_utils.py    │  │ api_metrics_logger.py│ │
│  │  - Retry Logic          │  │  - Performance Track │ │
│  │  - Circuit Breaker      │  │  - Cost Analysis    │ │
│  │  - Timeout Management   │  │  - Usage Reports    │ │
│  └─────────────────────────┘  └──────────────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 External Services                        │
│  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │  Google OAuth    │  │  Azure OpenAI API          │  │
│  │  API             │  │  (GPT-4o audio preview)    │  │
│  └──────────────────┘  └────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Module Dependencies
```python
# Dependency Graph
app.py
├── api_error_utils.py      # Error handling decorators
├── api_metrics_logger.py   # Performance monitoring
├── oauth_debug.py          # OAuth troubleshooting
├── streamlit               # UI framework
├── openai                  # Azure OpenAI client
├── requests                # HTTP client
├── jwt                     # Token validation
└── dotenv                  # Environment configuration

api_error_utils.py
├── streamlit               # UI notifications
├── requests                # HTTP error handling
└── openai                  # API error classification

api_metrics_logger.py
├── pandas                  # Data analysis
├── streamlit               # Dashboard display
└── dataclasses            # Metrics structures
```

### 2.3 Data Flow
1. **Authentication Flow**
   ```
   User → Google OAuth → Token Exchange → Email Verification → Session Creation
   ```

2. **Transcription Flow**
   ```
   Audio Recording → Base64 Encoding → API Request → Response Processing → UI Display
   ```

3. **Error Recovery Flow**
   ```
   API Call → Error Detection → Classification → Retry/Circuit Break → Fallback/Cache
   ```

### 2.4 State Management
```python
# Session State Structure
st.session_state = {
    'authenticated': bool,              # Auth status
    'user_email': str,                  # Authorized email
    'transcription_segments': List[str], # Transcript history
    'messages': List[dict],             # API message context
    'metrics_logger': APIMetricsLogger, # Metrics instance
    'api_metrics_log': List[dict],     # Detailed metrics
    'consolidated_analysis': str,       # Cached analysis
    'error_log': List[dict],           # Error history
    'health_check_time': float,        # Last health check
    'api_healthy': bool                # API status
}
```

---

## 3. Module Documentation

### 3.1 app.py - Main Application
**Purpose**: Core application orchestrator handling UI, authentication, and transcription workflow

**Key Functions**:
```python
def get_config() -> dict:
    """Load configuration from secrets or environment
    Returns: Configuration dictionary with API keys and settings"""

def create_openai_client() -> openai.AzureOpenAI:
    """Initialize Azure OpenAI client with connection pooling
    Returns: Configured OpenAI client instance"""

def authenticate_with_gmail() -> bool:
    """Handle complete OAuth authentication flow
    Returns: True if authenticated, False otherwise"""

@api_retry @with_timeout
def exchange_code_for_token(code: str) -> dict:
    """Exchange OAuth code for access token with retry logic
    Args: code - Authorization code from OAuth callback
    Returns: Token response dictionary"""

def verify_user_email(id_token: str) -> bool:
    """Validate user email from JWT token
    Args: id_token - Google ID token
    Returns: True if email matches authorized email"""

@api_retry @validate_response
def transcribe_audio(messages: list, deployment_id: str,
                     segment_id: int, audio_metrics: dict) -> Completion:
    """Process audio transcription with metrics tracking
    Args: messages - API message history
          deployment_id - Model deployment name
          segment_id - Current segment number
          audio_metrics - Audio file metrics
    Returns: OpenAI completion response"""

@api_retry @cache_on_error
def create_consolidated_analysis() -> str:
    """Generate consolidated analysis of all segments
    Returns: Formatted analysis text"""

def handle_transcription_error(error: Exception, segment_num: int):
    """Centralized error handling with user feedback
    Args: error - Exception instance
          segment_num - Failed segment number"""

def main():
    """Main application entry point with error boundaries"""
```

**Dependencies**:
- External: streamlit, openai, requests, jwt, dotenv
- Internal: api_error_utils, api_metrics_logger, oauth_debug

**Configuration Requirements**:
- GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
- AZURE_OPENAI_API_ENDPOINT_US2, AZURE_OPENAI_API_KEY_US2
- AUTHORIZED_EMAIL, APP_DOMAIN

### 3.2 api_error_utils.py - Error Handling Framework
**Purpose**: Robust error handling utilities providing retry logic, circuit breakers, and timeout management

**Key Components**:
```python
class APIError(Exception):
    """Base exception for API-related errors"""

class RetryableError(APIError):
    """Errors that should trigger retry attempts"""

class NonRetryableError(APIError):
    """Errors that should not trigger retries"""

def classify_error(exception: Exception) -> Tuple[bool, str, str]:
    """Classify errors as retryable or non-retryable
    Returns: (is_retryable, error_type, user_message)"""

@decorator
def api_retry(max_attempts=3, initial_delay=1.0, max_delay=30.0,
              exponential_base=2.0, jitter=True, use_streamlit_ui=True):
    """Exponential backoff retry with UI feedback
    Features: Automatic error classification, progress display,
              configurable backoff, jitter prevention"""

@decorator
def circuit_breaker(failure_threshold=5, recovery_timeout=60.0):
    """Prevent cascading failures with circuit breaker pattern
    Features: Automatic circuit opening, timed recovery,
              failure counting, service protection"""

@decorator
def with_timeout(timeout_seconds=30.0):
    """Add timeout protection to function calls
    Features: Thread-based timeout, clean termination,
              exception propagation"""

@decorator
def cache_on_error(cache_key: str, ttl_seconds=3600):
    """Cache successful responses for error fallback
    Features: Session-based caching, TTL management,
              automatic fallback"""

@decorator
def validate_response(validator: Callable, error_message: str):
    """Validate API responses against expected schema
    Features: Custom validation logic, clear error messages"""

def batch_api_call(items: list, api_function: Callable,
                   batch_size=10, allow_partial=True) -> Tuple[list, list]:
    """Process items in batches with partial failure handling
    Returns: (successful_results, failed_items)"""
```

**Error Classification Matrix**:
```python
# Retryable Errors (with backoff)
- openai.APITimeoutError → timeout
- openai.RateLimitError → rate_limit
- openai.APIConnectionError → connection
- openai.InternalServerError → server_error
- HTTP 429, 500, 502, 503, 504 → server issues

# Non-Retryable Errors (immediate failure)
- openai.AuthenticationError → auth
- HTTP 400, 401, 403, 404 → client errors
- Validation failures → invalid request
```

### 3.3 api_metrics_logger.py - Performance Monitoring
**Purpose**: Comprehensive metrics tracking, performance analysis, and cost estimation

**Key Components**:
```python
@dataclass
class TranscriptionMetrics:
    """Comprehensive metrics for each transcription
    Fields: segment_id, timestamp, audio_size_bytes,
            audio_duration_estimate, encoding_time,
            api_call_time, total_time, tokens (prompt/completion/total),
            model, status, error_type, retry_count,
            context_segments_used, response_length"""

    @property
    def tokens_per_second(self) -> float
    @property
    def audio_to_text_ratio(self) -> float
    @property
    def cost_estimate(self) -> float

class APIMetricsLogger:
    """Centralized metrics collection and analysis"""

    def log_transcription(self, metrics: TranscriptionMetrics):
        """Record transcription event with automatic summarization"""

    def get_current_session_metrics(self) -> Dict[str, Any]:
        """Calculate session-level statistics
        Returns: session_duration, success_rate, avg_latency,
                 total_tokens, estimated_cost, total_audio_mb"""

    def get_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance patterns and trends
        Returns: latency percentiles (p50/p95), error distribution,
                 cache hit rate, retry statistics"""

    def export_metrics_csv(self) -> str:
        """Export detailed metrics for external analysis"""

    def display_metrics_dashboard(self):
        """Render real-time metrics dashboard in sidebar"""

def measure_audio_metrics(audio_data: bytes) -> Dict[str, Any]:
    """Extract audio file characteristics
    Returns: size_bytes, duration_seconds, sample_rate, channels"""

@decorator
def track_api_call(func):
    """Generic API call tracking decorator"""

def log_token_usage(completion) -> dict:
    """Extract token usage from OpenAI response"""

def create_performance_report() -> str:
    """Generate optimization recommendations report
    Sections: Executive summary, Performance insights,
              Error analysis, Optimization recommendations"""
```

**Metrics Collection Points**:
1. Audio processing (size, duration, encoding time)
2. API calls (latency, tokens, retry count)
3. System health (success rate, error distribution)
4. Cost tracking (token usage, estimated charges)

### 3.4 oauth_debug.py - Authentication Debugging
**Purpose**: OAuth troubleshooting utilities for development and deployment

**Key Functions**:
```python
def oauth_debug():
    """Interactive OAuth debugging interface
    Features: Configuration display, URL inspection,
              Token exchange testing, Error diagnostics"""
```

**Debug Capabilities**:
- Configuration validation
- OAuth URL generation and inspection
- Token exchange testing
- Email authorization verification
- Common error solutions

---

## 4. File Boundaries

### 4.1 Directory Structure
```
4o-audio-streamlit/
├── app.py                    # Main application (READ/WRITE: session state)
├── api_error_utils.py        # Error utilities (READ-ONLY)
├── api_metrics_logger.py     # Metrics tracking (READ/WRITE: session state)
├── oauth_debug.py            # Debug utilities (READ-ONLY)
├── requirements.txt          # Python dependencies
├── render.yaml              # Deployment config
├── .dockerfile              # Container definition
├── .dockerignore           # Docker exclusions
├── .gitignore              # Git exclusions
├── .env                    # Local secrets (NEVER COMMIT)
├── .streamlit/
│   ├── config.toml         # Streamlit configuration
│   └── secrets.toml        # Production secrets (NEVER COMMIT)
├── docs/
│   └── project_doc.md      # This documentation
├── deploy/                 # Deployment artifacts
└── __pycache__/           # Python bytecode (AUTO-GENERATED)
```

### 4.2 Safe Zones (Read/Write Permitted)
```python
# Session State (In-Memory)
st.session_state.*           # All session variables
st.cache_data               # Cached computations
st.cache_resource           # Cached resources

# Temporary Data
io.BytesIO()                # Audio buffer
io.StringIO()               # CSV export buffer
```

### 4.3 Unsafe Zones (Read-Only or Restricted)
```python
# Configuration Files
.streamlit/secrets.toml     # NEVER write, contains secrets
.env                        # NEVER write, local development only

# System Files
__pycache__/               # Auto-generated, don't modify
*.pyc                      # Bytecode files

# External Services
https://oauth2.googleapis.com/*     # OAuth endpoints
https://accounts.google.com/*       # Authentication
Azure OpenAI endpoints              # API calls only
```

### 4.4 Configuration Hierarchy
```
Priority Order (highest to lowest):
1. Streamlit Secrets (st.secrets)      # Production
2. Environment Variables (os.getenv)    # Docker/CI
3. .env file (dotenv.load_dotenv)      # Local development
4. Default values in code               # Fallback
```

### 4.5 Secret Management
```python
# Required Secrets
GOOGLE_CLIENT_ID            # OAuth client identifier
GOOGLE_CLIENT_SECRET        # OAuth client secret
AZURE_OPENAI_API_KEY_US2   # Azure OpenAI API key
AZURE_OPENAI_API_ENDPOINT_US2  # Azure OpenAI endpoint

# Configurable Settings
AUTHORIZED_EMAIL           # Single authorized user
APP_DOMAIN                # Deployment domain
```

---

## 5. API & Coding Standards

### 5.1 Error Handling Patterns
```python
# Standard Error Handling Flow
try:
    # Primary operation
    result = api_call()
except openai.RateLimitError as e:
    # Specific handling for rate limits
    logger.warning(f"Rate limit hit: {e}")
    # Use cached result or queue for retry
except APIError as e:
    # Handle known API errors
    handle_transcription_error(e, context)
except Exception as e:
    # Unknown errors - log and re-raise
    logger.error(f"Unexpected error: {e}")
    raise

# Decorator Stack Pattern
@api_retry(max_attempts=3)          # Retry on failure
@circuit_breaker(failure_threshold=5) # Prevent cascading failures
@with_timeout(timeout_seconds=30)    # Timeout protection
@validate_response(validator=lambda x: x and x.choices)
def api_operation():
    pass
```

### 5.2 Logging Standards
```python
# Log Levels
logger.debug()    # Detailed diagnostic info
logger.info()     # General informational messages
logger.warning()  # Warning conditions
logger.error()    # Error conditions
logger.critical() # Critical failures requiring immediate attention

# Log Format
"[%(asctime)s] [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"

# Structured Logging
logger.info(f"Transcription Segment {segment_id}: "
           f"Status={status}, Tokens={tokens}, "
           f"Time={duration:.2f}s, Audio={size_kb:.1f}KB")
```

### 5.3 Metrics Collection Standards
```python
# Metrics Collection Points
1. Request Initiation    → Start timer, log request details
2. Processing Steps      → Intermediate measurements
3. Success/Failure       → Final metrics, status recording
4. Cleanup              → Archive old metrics

# Metric Categories
- Performance: latency, throughput, processing rate
- Reliability: success rate, retry count, error distribution
- Resource Usage: tokens, audio size, memory
- Business: cost estimation, usage patterns
```

### 5.4 Authentication Flow
```python
# OAuth 2.0 Flow Implementation
1. Initialize OAuth URL with parameters
2. Redirect user to Google authentication
3. Handle callback with authorization code
4. Exchange code for tokens
5. Validate email from ID token
6. Create authenticated session

# Session Management
- Session timeout: No automatic timeout (Streamlit managed)
- Token refresh: Not implemented (single-use auth)
- Logout: Clear session state and redirect
```

### 5.5 API Communication Standards
```python
# Request Structure
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "assistant", "content": context},  # Previous segments
    {"role": "user", "content": [
        {"type": "text", "text": user_prompt},
        {"type": "input_audio", "input_audio": {
            "data": base64_encoded_audio,
            "format": "wav"
        }}
    ]}
]

# Response Validation
def validate_transcription_response(response):
    return (response and
            response.choices and
            response.choices[0].message.content)

# Timeout Configuration
Client Level: 60 seconds (OpenAI client)
Request Level: 30 seconds (decorator)
Health Check: 5 seconds (quick validation)
```

### 5.6 Type Hints and Documentation
```python
# Function Signature Pattern
def transcribe_audio(
    messages: List[Dict[str, Any]],
    deployment_id: str = DEPLOYMENT_ID,
    segment_id: int = 0,
    audio_metrics: Optional[Dict[str, Any]] = None
) -> openai.types.chat.ChatCompletion:
    """Process audio transcription with metrics tracking.

    Args:
        messages: Chat completion message history
        deployment_id: Azure OpenAI deployment identifier
        segment_id: Current segment number for tracking
        audio_metrics: Pre-calculated audio metrics

    Returns:
        ChatCompletion response from OpenAI API

    Raises:
        APIError: For retryable API failures
        NonRetryableError: For permanent failures
    """
```

---

## 6. Execution & CLI

### 6.1 Local Development
```bash
# Environment Setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install Dependencies
pip install -r requirements.txt

# Configure Secrets (create .env file)
cat > .env << EOF
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
AZURE_OPENAI_API_KEY_US2=your_api_key
AZURE_OPENAI_API_ENDPOINT_US2=your_endpoint
AUTHORIZED_EMAIL=your_email@example.com
APP_DOMAIN=localhost:8501
EOF

# Run Application
streamlit run app.py

# With custom configuration
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

### 6.2 Docker Deployment
```bash
# Build Docker Image
docker build -t audio-transcription-app .

# Run Container
docker run -p 8080:8080 \
  -e GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID \
  -e GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET \
  -e AZURE_OPENAI_API_KEY_US2=$AZURE_OPENAI_API_KEY_US2 \
  -e AZURE_OPENAI_API_ENDPOINT_US2=$AZURE_OPENAI_API_ENDPOINT_US2 \
  -e AUTHORIZED_EMAIL=$AUTHORIZED_EMAIL \
  -e APP_DOMAIN=$APP_DOMAIN \
  audio-transcription-app
```

### 6.3 Render Deployment
```yaml
# render.yaml configuration
services:
  - type: web
    name: transcription-app
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py
    envVars:
      - key: GOOGLE_CLIENT_ID
        sync: false  # Set in Render dashboard
      - key: GOOGLE_CLIENT_SECRET
        sync: false  # Set in Render dashboard
      # ... other environment variables
```

### 6.4 Debug Commands
```bash
# OAuth Debug Mode
streamlit run oauth_debug.py

# Check Configuration
python -c "from app import get_config; print(get_config())"

# Test Azure OpenAI Connection
python -c "from app import create_openai_client; client = create_openai_client()"

# Export Metrics (from UI)
# Use sidebar "Export Metrics CSV" button
```

### 6.5 Health Monitoring
```python
# Automated Health Checks
- Frequency: Every 5 minutes
- Timeout: 5 seconds
- Test: Simple API call with 1 token
- UI Indicator: Sidebar status display

# Manual Health Check
curl -X GET http://localhost:8501/healthz  # If implemented
```

### 6.6 Performance Tuning
```bash
# Streamlit Configuration (.streamlit/config.toml)
[server]
maxUploadSize = 200        # Max file upload size in MB
maxMessageSize = 200       # Max WebSocket message size

[runner]
magicEnabled = false       # Disable magic commands for performance

[client]
showErrorDetails = false   # Hide detailed errors in production
```

---

## 7. Testing Strategy

### 7.1 Unit Testing Framework
```python
# tests/test_api_error_utils.py
import pytest
from unittest.mock import Mock, patch
from api_error_utils import api_retry, classify_error, circuit_breaker

class TestErrorClassification:
    def test_retryable_errors(self):
        """Test classification of retryable errors"""
        import openai
        error = openai.RateLimitError("Rate limit exceeded")
        is_retryable, error_type, message = classify_error(error)
        assert is_retryable == True
        assert error_type == "rate_limit"

    def test_non_retryable_errors(self):
        """Test classification of non-retryable errors"""
        import openai
        error = openai.AuthenticationError("Invalid API key")
        is_retryable, error_type, message = classify_error(error)
        assert is_retryable == False
        assert error_type == "auth"

class TestRetryDecorator:
    @patch('time.sleep')
    def test_retry_on_failure(self, mock_sleep):
        """Test retry logic with exponential backoff"""
        mock_func = Mock(side_effect=[Exception("Fail"), "Success"])

        @api_retry(max_attempts=2, use_streamlit_ui=False)
        def test_function():
            return mock_func()

        result = test_function()
        assert result == "Success"
        assert mock_func.call_count == 2

class TestCircuitBreaker:
    def test_circuit_opens_after_threshold(self):
        """Test circuit breaker opening after failures"""
        mock_func = Mock(side_effect=Exception("API Error"))

        @circuit_breaker(failure_threshold=3, recovery_timeout=1)
        def test_function():
            return mock_func()

        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                test_function()

        # Circuit should be open
        with pytest.raises(APIError, match="Circuit breaker is open"):
            test_function()
```

### 7.2 Integration Testing
```python
# tests/test_transcription_flow.py
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from app import transcribe_audio, create_consolidated_analysis

class TestTranscriptionFlow:
    @patch('app.openai_client_us2')
    def test_successful_transcription(self, mock_client):
        """Test complete transcription workflow"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Transcribed text"
        mock_client.chat.completions.create.return_value = mock_response

        # Test transcription
        messages = [{"role": "user", "content": "test"}]
        result = transcribe_audio(messages, segment_id=1)

        assert result.choices[0].message.content == "Transcribed text"

    def test_consolidated_analysis(self):
        """Test multi-segment consolidation"""
        with patch.object(st, 'session_state', {'transcription_segments': [
            "First segment content",
            "Second segment content"
        ]}):
            analysis = create_consolidated_analysis()
            assert "First segment" in analysis
            assert "Second segment" in analysis
```

### 7.3 Performance Testing
```python
# tests/test_performance.py
import time
import pytest
from api_metrics_logger import TranscriptionMetrics, APIMetricsLogger

class TestPerformanceMetrics:
    def test_metrics_calculation(self):
        """Test performance metric calculations"""
        metrics = TranscriptionMetrics(
            segment_id=1,
            timestamp=time.time(),
            audio_size_bytes=1024000,
            audio_duration_estimate=10.0,
            encoded_size_bytes=1365333,
            encoding_time=0.5,
            api_call_time=2.0,
            total_time=2.5,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o-audio-preview",
            status="success"
        )

        assert metrics.tokens_per_second == 75.0  # 150/2
        assert metrics.audio_to_text_ratio == 5.0  # 10/2
        assert metrics.cost_estimate > 0

    def test_performance_analysis(self):
        """Test performance analysis generation"""
        logger = APIMetricsLogger()

        # Add sample metrics
        for i in range(10):
            metrics = TranscriptionMetrics(
                segment_id=i,
                timestamp=time.time(),
                audio_size_bytes=1024000,
                audio_duration_estimate=10.0,
                encoded_size_bytes=1365333,
                encoding_time=0.5,
                api_call_time=2.0 + i * 0.1,  # Varying latency
                total_time=2.5 + i * 0.1,
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                model="gpt-4o-audio-preview",
                status="success"
            )
            logger.log_transcription(metrics)

        analysis = logger.get_performance_analysis()
        assert 'avg_api_latency' in analysis
        assert 'p95_api_latency' in analysis
        assert analysis['error_rate'] == 0.0
```

### 7.4 Load Testing
```bash
# Using locust for load testing
pip install locust

# tests/load_test.py
from locust import HttpUser, task, between

class TranscriptionUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def transcribe_audio(self):
        # Simulate audio transcription request
        with open('test_audio.wav', 'rb') as f:
            files = {'audio': f}
            self.client.post('/transcribe', files=files)

# Run load test
locust -f tests/load_test.py --host http://localhost:8501
```

### 7.5 Test Coverage Requirements
```bash
# Run tests with coverage
pytest --cov=. --cov-report=term-missing --cov-report=html

# Coverage targets
- api_error_utils.py: 90% minimum
- api_metrics_logger.py: 85% minimum
- app.py (non-UI logic): 80% minimum
- Overall: 80% minimum

# Critical paths requiring 100% coverage
- Error classification logic
- Authentication verification
- Token usage calculation
- Circuit breaker state management
```

### 7.6 CI/CD Testing Pipeline
```yaml
# .github/workflows/test.yml
name: Test Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=. --cov-fail-under=80
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## 8. Extensibility & Integration

### 8.1 Adding New Features
```python
# Pattern for adding new transcription models
class TranscriptionProvider(ABC):
    """Base class for transcription providers"""
    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        pass

class GPT4AudioProvider(TranscriptionProvider):
    """GPT-4 Audio implementation"""
    def transcribe(self, audio_data: bytes) -> str:
        # Implementation
        pass

class WhisperProvider(TranscriptionProvider):
    """Whisper API implementation"""
    def transcribe(self, audio_data: bytes) -> str:
        # Implementation
        pass

# Factory pattern for provider selection
def create_provider(provider_type: str) -> TranscriptionProvider:
    providers = {
        'gpt4-audio': GPT4AudioProvider,
        'whisper': WhisperProvider
    }
    return providers[provider_type]()
```

### 8.2 Claude Code Integration
```python
# CLAUDE.md configuration for AI assistance
"""
## Working with this codebase

### Key patterns to follow:
1. Always use decorators for API calls (@api_retry, @with_timeout)
2. Log all errors with context using logger.error()
3. Track metrics for new API operations
4. Validate configuration before making changes

### Common tasks:
- Adding new API endpoint: Extend api_error_utils decorators
- Adding metrics: Use TranscriptionMetrics dataclass
- Debugging OAuth: Run oauth_debug.py tool
"""

# Subagent task example
"""
When implementing new audio processing features:
1. Review existing transcribe_audio() implementation
2. Maintain compatibility with metrics logging
3. Add appropriate error handling decorators
4. Update performance report generation
"""
```

### 8.3 Plugin Architecture
```python
# Plugin system for custom processors
class AudioProcessor:
    """Base class for audio processors"""
    def process(self, audio_data: bytes) -> bytes:
        return audio_data

class NoiseReductionProcessor(AudioProcessor):
    """Reduce background noise"""
    def process(self, audio_data: bytes) -> bytes:
        # Apply noise reduction
        return processed_data

# Pipeline for chaining processors
class ProcessingPipeline:
    def __init__(self):
        self.processors = []

    def add_processor(self, processor: AudioProcessor):
        self.processors.append(processor)

    def process(self, audio_data: bytes) -> bytes:
        for processor in self.processors:
            audio_data = processor.process(audio_data)
        return audio_data

# Usage in main app
pipeline = ProcessingPipeline()
pipeline.add_processor(NoiseReductionProcessor())
processed_audio = pipeline.process(raw_audio)
```

### 8.4 Database Integration
```python
# Pattern for adding database support
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class TranscriptionRecord(Base):
    __tablename__ = 'transcriptions'

    id = Column(String, primary_key=True)
    user_email = Column(String)
    timestamp = Column(DateTime)
    audio_size = Column(Float)
    transcript = Column(String)
    cost = Column(Float)

    @classmethod
    def from_metrics(cls, metrics: TranscriptionMetrics):
        """Create record from metrics object"""
        return cls(
            id=str(uuid.uuid4()),
            timestamp=datetime.fromtimestamp(metrics.timestamp),
            audio_size=metrics.audio_size_bytes,
            cost=metrics.cost_estimate
        )

# Integration point in app.py
def save_transcription(metrics, transcript):
    if DATABASE_ENABLED:
        record = TranscriptionRecord.from_metrics(metrics)
        record.transcript = transcript
        session.add(record)
        session.commit()
```

### 8.5 Webhook Integration
```python
# Pattern for webhook notifications
import httpx

class WebhookNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    @api_retry(max_attempts=2)
    async def notify_completion(self, segment_id: int, transcript: str):
        """Send webhook on transcription completion"""
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json={
                'event': 'transcription_complete',
                'segment_id': segment_id,
                'transcript': transcript,
                'timestamp': time.time()
            })

    @api_retry(max_attempts=2)
    async def notify_error(self, segment_id: int, error: str):
        """Send webhook on transcription error"""
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json={
                'event': 'transcription_error',
                'segment_id': segment_id,
                'error': error,
                'timestamp': time.time()
            })
```

### 8.6 Custom UI Components
```python
# Pattern for custom Streamlit components
import streamlit.components.v1 as components

def audio_waveform_component(audio_data: bytes):
    """Custom component for audio visualization"""
    # Generate waveform data
    waveform_html = f"""
    <canvas id="waveform"></canvas>
    <script>
        // Waveform visualization code
        const audioData = {list(audio_data[:1000])};
        // Draw waveform
    </script>
    """
    components.html(waveform_html, height=200)

# Usage in app.py
if audio_value:
    audio_data = audio_value.read()
    audio_waveform_component(audio_data)
```

---

## 9. Context Hygiene

### 9.1 CLAUDE.md Best Practices
```markdown
# CLAUDE.md Structure

## Repository Overview
Brief description of project purpose and architecture

## Key Directories
Clear mapping of directory structure and purposes

## Coding Standards
- Language-specific standards (PEP 8 for Python)
- Error handling requirements
- Documentation expectations

## Common Tasks
Step-by-step guides for frequent operations

## Testing Requirements
Coverage targets and testing patterns

## DO NOT
- List of actions to avoid
- Common pitfalls
- Security considerations
```

### 9.2 Import Organization
```python
# Standard import order (PEP 8)
# 1. Standard library imports
import os
import time
import logging
from typing import Dict, List, Optional

# 2. Related third-party imports
import streamlit as st
import pandas as pd
import openai
import requests

# 3. Local application imports
from api_error_utils import api_retry, APIError
from api_metrics_logger import APIMetricsLogger, TranscriptionMetrics

# Import grouping rules
- Group imports by category
- Alphabetize within groups
- One import per line for clarity
- Use explicit imports (avoid from x import *)
```

### 9.3 Module Boundaries
```python
# Clear separation of concerns
"""
app.py
- UI/UX logic
- User interaction handling
- Session state management
- High-level orchestration

api_error_utils.py
- Error handling logic only
- No business logic
- No UI dependencies
- Pure utility functions

api_metrics_logger.py
- Metrics collection and analysis
- Performance reporting
- No transcription logic
- Optional UI components (dashboard)

oauth_debug.py
- Debugging utilities only
- Not imported in production
- Standalone operation
"""

# Anti-patterns to avoid
# ❌ Circular imports
# ❌ Business logic in utilities
# ❌ UI code in service layers
# ❌ Direct file I/O in decorators
```

### 9.4 Configuration Management
```python
# Centralized configuration pattern
class Config:
    """Application configuration singleton"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from appropriate source"""
        self.GOOGLE_CLIENT_ID = self._get_secret('GOOGLE_CLIENT_ID')
        self.AZURE_API_KEY = self._get_secret('AZURE_OPENAI_API_KEY_US2')
        # ... other config

    def _get_secret(self, key: str, default=None):
        """Get secret with fallback hierarchy"""
        # 1. Try Streamlit secrets
        # 2. Try environment variable
        # 3. Try .env file
        # 4. Use default
        pass

# Usage
config = Config()
client_id = config.GOOGLE_CLIENT_ID
```

### 9.5 Error Context Preservation
```python
# Maintaining error context through layers
class ContextualError(Exception):
    """Error with preserved context"""
    def __init__(self, message: str, context: dict):
        super().__init__(message)
        self.context = context

# Usage pattern
try:
    result = transcribe_audio(audio_data)
except Exception as e:
    raise ContextualError(
        "Transcription failed",
        context={
            'segment_id': current_segment,
            'audio_size': len(audio_data),
            'original_error': str(e),
            'timestamp': time.time()
        }
    )
```

### 9.6 Documentation Standards
```python
# Comprehensive docstring pattern
def complex_operation(
    param1: str,
    param2: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Perform complex operation with multiple parameters.

    This function handles the complex business logic for processing
    data with various configuration options.

    Args:
        param1: Primary parameter description
        param2: Optional parameter with default behavior
        **kwargs: Additional options:
            - option1 (bool): Enable feature X
            - option2 (str): Configuration for Y

    Returns:
        Dictionary containing:
            - 'result': Processing result
            - 'metadata': Operation metadata
            - 'metrics': Performance metrics

    Raises:
        ValueError: If param1 is invalid
        APIError: If external service fails
        TimeoutError: If operation exceeds timeout

    Example:
        >>> result = complex_operation("test", param2=5)
        >>> print(result['metadata'])
        {'duration': 1.5, 'status': 'success'}

    Note:
        This operation is cached for 5 minutes to reduce API calls
    """
    pass
```

---

## 10. Merge & Refactor Guidance

### 10.1 Integration with Other Projects
```python
# Modular integration pattern
class TranscriptionModule:
    """Standalone transcription module for integration"""

    def __init__(self, config: dict):
        self.config = config
        self._setup_clients()
        self._setup_error_handling()

    def _setup_clients(self):
        """Initialize required clients"""
        self.openai_client = self._create_openai_client()
        self.metrics_logger = APIMetricsLogger()

    def transcribe(self, audio_data: bytes) -> dict:
        """Public interface for transcription"""
        return {
            'transcript': self._process_audio(audio_data),
            'metrics': self._get_metrics(),
            'status': 'success'
        }

# Integration example
from transcription_module import TranscriptionModule

module = TranscriptionModule(config={
    'api_key': 'xxx',
    'endpoint': 'xxx'
})
result = module.transcribe(audio_bytes)
```

### 10.2 Code Organization Patterns
```
# Recommended package structure for larger projects
audio_transcription/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── transcription.py    # Core transcription logic
│   ├── authentication.py   # OAuth handling
│   └── models.py           # Data models
├── services/
│   ├── __init__.py
│   ├── openai_service.py   # OpenAI integration
│   ├── error_handler.py    # Error utilities
│   └── metrics_service.py  # Metrics tracking
├── ui/
│   ├── __init__.py
│   ├── main_app.py        # Streamlit UI
│   ├── components.py      # UI components
│   └── dashboard.py       # Metrics dashboard
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py     # Audio processing
│   └── validators.py      # Input validation
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

### 10.3 Migration Strategies
```python
# Version migration pattern
class MigrationManager:
    """Handle data/config migrations between versions"""

    MIGRATIONS = {
        '1.0.0': 'migrate_v1_to_v2',
        '2.0.0': 'migrate_v2_to_v3'
    }

    def migrate(self, from_version: str, to_version: str, data: dict):
        """Apply migrations between versions"""
        migrations_to_apply = self._get_migration_path(
            from_version, to_version
        )

        for migration_name in migrations_to_apply:
            migration_func = getattr(self, migration_name)
            data = migration_func(data)

        return data

    def migrate_v1_to_v2(self, data: dict) -> dict:
        """Migration from v1.0.0 to v2.0.0"""
        # Transform data structure
        data['version'] = '2.0.0'
        data['metrics'] = data.pop('stats', {})
        return data
```

### 10.4 Refactoring Guidelines
```python
# Before refactoring checklist
"""
1. Ensure comprehensive test coverage (>80%)
2. Document current behavior
3. Create feature flags for gradual rollout
4. Preserve backward compatibility
5. Plan rollback strategy
"""

# Feature flag pattern
class FeatureFlags:
    ENABLE_NEW_TRANSCRIPTION = os.getenv('ENABLE_NEW_TRANSCRIPTION', 'false').lower() == 'true'
    ENABLE_ADVANCED_METRICS = os.getenv('ENABLE_ADVANCED_METRICS', 'false').lower() == 'true'

# Usage
if FeatureFlags.ENABLE_NEW_TRANSCRIPTION:
    result = new_transcription_logic()
else:
    result = legacy_transcription_logic()
```

### 10.5 API Versioning
```python
# API versioning for external integrations
from flask import Flask, request
app = Flask(__name__)

@app.route('/api/v1/transcribe', methods=['POST'])
def transcribe_v1():
    """Legacy API endpoint"""
    audio_data = request.files['audio'].read()
    return {'transcript': legacy_transcribe(audio_data)}

@app.route('/api/v2/transcribe', methods=['POST'])
def transcribe_v2():
    """New API with enhanced features"""
    audio_data = request.files['audio'].read()
    options = request.json.get('options', {})
    return {
        'transcript': enhanced_transcribe(audio_data, options),
        'metrics': get_transcription_metrics(),
        'version': 'v2'
    }
```

### 10.6 Performance Optimization Patterns
```python
# Caching strategy
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def expensive_computation(audio_hash: str) -> str:
    """Cache expensive computations by audio hash"""
    # Perform computation
    return result

def process_audio(audio_data: bytes) -> str:
    # Generate deterministic hash
    audio_hash = hashlib.sha256(audio_data).hexdigest()

    # Check cache first
    return expensive_computation(audio_hash)

# Batch processing pattern
def batch_transcribe(audio_files: List[bytes]) -> List[str]:
    """Process multiple files efficiently"""
    results = []

    # Group into optimal batch sizes
    batch_size = 5
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i + batch_size]

        # Process batch in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            batch_results = executor.map(transcribe_single, batch)
            results.extend(batch_results)

    return results
```

---

## Appendix A: Quick Reference

### API Endpoints
```python
# OpenAI/Azure OpenAI
endpoint = "https://your-instance.openai.azure.com/"
model = "gpt-4o-audio-preview"

# Google OAuth
auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
token_url = "https://oauth2.googleapis.com/token"
```

### Environment Variables
```bash
# Required
GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET
AZURE_OPENAI_API_KEY_US2
AZURE_OPENAI_API_ENDPOINT_US2

# Optional
AUTHORIZED_EMAIL (default: sergeyly@gmail.com)
APP_DOMAIN (default: localhost:8501)
```

### Common Commands
```bash
# Development
streamlit run app.py
streamlit run oauth_debug.py

# Testing
pytest tests/
pytest --cov=. --cov-report=html

# Docker
docker build -t app .
docker run -p 8080:8080 app

# Metrics Export
# Use UI button or programmatic export
```

### Error Codes
```python
# Custom error types
APIError          # Base API error
RetryableError    # Should retry
NonRetryableError # Don't retry

# HTTP Status Codes
429  # Rate limit (retry)
500+ # Server errors (retry)
400  # Bad request (don't retry)
401  # Unauthorized (don't retry)
```

---

## Appendix B: Security Considerations

### Authentication Security
- OAuth 2.0 with PKCE flow recommended for production
- Single authorized email prevents unauthorized access
- No token persistence reduces attack surface
- Session isolation per user

### API Key Management
- Never commit secrets to repository
- Use Streamlit secrets for production
- Rotate keys regularly
- Monitor usage for anomalies

### Data Protection
- Audio data processed in memory only
- No persistent storage of recordings
- Transcripts tied to session state
- Automatic session cleanup on logout

### Network Security
- HTTPS only for production deployment
- API calls use TLS encryption
- Timeout protection prevents hanging connections
- Circuit breaker prevents cascade failures

---

*Documentation Version: 1.0.0*
*Last Updated: 2024*
*Total Lines: ~1,150*
*Estimated Tokens: ~18,000*