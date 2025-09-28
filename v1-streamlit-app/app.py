import dotenv
import openai
import os
import streamlit as st
import base64
import requests
from urllib.parse import urlencode
import jwt
import json
import logging
import time
import io
import wave
import numpy as np
import soundfile as sf
import librosa

# Import shared components
import sys
import os
# Add shared_components directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import audio processing configuration and functions
from shared_components import audio_config
from shared_components.audio_processor import resample_audio_auto

# Import the error handling utilities
from shared_components.api_error_utils import (
    api_retry, circuit_breaker, with_timeout,
    cache_on_error, validate_response, APIError
)

# Import metrics tracking
from shared_components.api_metrics_logger import (
    APIMetricsLogger, TranscriptionMetrics,
    measure_audio_metrics, log_token_usage,
    create_performance_report
)


# Initial oauth_debug troubleshooting
from oauth_debug import oauth_debug

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables if .env exists (for local development)
if os.path.exists('.env'):
    dotenv.load_dotenv('.env')

def get_config():
    """Get configuration from secrets or environment variables"""
    config = {}
    
    # Try Streamlit secrets first, then environment variables
    def get_secret(key, default=None):
        try:
            return st.secrets[key]
        except (KeyError, FileNotFoundError):
            return os.getenv(key, default)
    
    config['GOOGLE_CLIENT_ID'] = get_secret('GOOGLE_CLIENT_ID')
    config['GOOGLE_CLIENT_SECRET'] = get_secret('GOOGLE_CLIENT_SECRET')
    config['AZURE_OPENAI_API_ENDPOINT_US2'] = get_secret('AZURE_OPENAI_API_ENDPOINT_US2')
    config['AZURE_OPENAI_API_KEY_US2'] = get_secret('AZURE_OPENAI_API_KEY_US2')
    config['AUTHORIZED_EMAIL'] = get_secret('AUTHORIZED_EMAIL', 'sergeyly@gmail.com')
    config['APP_DOMAIN'] = get_secret('APP_DOMAIN', 'localhost:8080')
    
    # Validate required secrets
    required_keys = ['GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET', 'AZURE_OPENAI_API_KEY_US2']
    missing_keys = [key for key in required_keys if not config[key]]
    
    if missing_keys:
        st.error(f"❌ Missing required configuration: {', '.join(missing_keys)}")
        st.info("Please check your secrets configuration in Streamlit Cloud or .streamlit/secrets.toml")
        st.stop()
    
    return config

# Initialize configuration
CONFIG = get_config()

# OpenAI client setup with connection pooling and timeout
def create_openai_client():
    """Create OpenAI client with error handling"""
    try:
        client = openai.AzureOpenAI(
            api_key=CONFIG['AZURE_OPENAI_API_KEY_US2'],
            api_version="2025-04-01-preview",
            azure_endpoint=CONFIG['AZURE_OPENAI_API_ENDPOINT_US2'],
            organization='Transcript PoC',
            max_retries=2,  # Client-level retry
            timeout=60.0     # Client-level timeout
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {str(e)}")
        st.error("❌ Failed to initialize AI service. Please check configuration.")
        st.stop()

openai_client_us2 = create_openai_client()

DEPLOYMENT_ID = "gpt-4o-audio-preview"
TARGET_SAMPLE_RATE = audio_config.TARGET_SAMPLE_RATE  # From configuration

system_prompt = """
You are a helpful AI Transcription Assistant. Create a transcript of the provided audio.
"""

user_prompt = """Identify in which language the attached input_audio is provided and transcribe the input_audio accurately in the same language (expected in Russian but possibly in English). If there are previous segments in this conversation, reference them when relevant and note any connections or continuations. Only include the transcript you produced into the output formatted as plain text and nothing else."""

def resample_audio(audio_data: bytes, target_sr: int = TARGET_SAMPLE_RATE, force_mono: bool = True) -> tuple[bytes, dict]:
    """
    Resample audio data to target sample rate and optionally convert to mono

    Args:
        audio_data: Raw WAV audio bytes
        target_sr: Target sample rate (default 16000 Hz)
        force_mono: Convert multi-channel audio to mono (default True)

    Returns:
        Tuple of (resampled_audio_bytes, resample_metrics)
    """
    resample_start = time.time()
    resample_metrics = {
        'original_sample_rate': None,
        'target_sample_rate': target_sr,
        'resample_time': 0,
        'resampled': False,
        'original_size': len(audio_data),
        'resampled_size': None,
        'original_channels': None,
        'converted_to_mono': False
    }

    try:
        # Read the audio data from bytes
        audio_io = io.BytesIO(audio_data)

        # Read audio with soundfile to get the sample rate and data
        audio_array, original_sr = sf.read(audio_io, dtype='float32')
        resample_metrics['original_sample_rate'] = original_sr

        # Determine number of channels
        if len(audio_array.shape) > 1:
            num_channels = audio_array.shape[1]
            resample_metrics['original_channels'] = num_channels
        else:
            num_channels = 1
            resample_metrics['original_channels'] = 1

        logger.info(f"Original audio: sample_rate={original_sr} Hz, channels={num_channels}, shape={audio_array.shape}")

        # Convert to mono if multi-channel and force_mono is True
        if force_mono and num_channels > 1:
            logger.info(f"Converting {num_channels}-channel audio to mono")
            # Average all channels to create mono
            audio_array = np.mean(audio_array, axis=1)
            resample_metrics['converted_to_mono'] = True

        # Check if resampling is needed
        needs_resampling = original_sr != target_sr

        if not needs_resampling and not resample_metrics['converted_to_mono']:
            logger.info(f"Audio already at target sample rate {target_sr} Hz and mono, skipping processing")
            resample_metrics['resample_time'] = time.time() - resample_start
            resample_metrics['resampled_size'] = len(audio_data)
            return audio_data, resample_metrics

        # Resample if needed
        if needs_resampling:
            # Single channel audio (or already converted to mono)
            resampled_audio = librosa.resample(
                audio_array,
                orig_sr=original_sr,
                target_sr=target_sr,
                res_type='kaiser_best'  # High quality resampling
            )
            resample_metrics['resampled'] = True
        else:
            # No resampling needed, just use the mono-converted audio
            resampled_audio = audio_array
            target_sr = original_sr  # Keep original sample rate

        # Convert back to WAV bytes
        output_io = io.BytesIO()
        sf.write(output_io, resampled_audio, target_sr, format='WAV', subtype='PCM_16')
        resampled_bytes = output_io.getvalue()

        # Update metrics
        resample_metrics['resampled'] = True
        resample_metrics['resampled_size'] = len(resampled_bytes)
        resample_metrics['resample_time'] = time.time() - resample_start

        # Log the conversion
        size_reduction = (1 - len(resampled_bytes) / len(audio_data)) * 100
        conversion_info = []
        if resample_metrics['resampled']:
            conversion_info.append(f"{original_sr} Hz → {target_sr} Hz")
        if resample_metrics['converted_to_mono']:
            conversion_info.append(f"{resample_metrics['original_channels']} ch → mono")

        logger.info(f"Audio processed: {', '.join(conversion_info)}, "
                   f"Size: {len(audio_data)/1024:.1f} KB → {len(resampled_bytes)/1024:.1f} KB "
                   f"({size_reduction:.1f}% reduction), Time: {resample_metrics['resample_time']:.3f}s")

        return resampled_bytes, resample_metrics

    except Exception as e:
        logger.error(f"Audio resampling failed: {str(e)}, using original audio")
        resample_metrics['resample_time'] = time.time() - resample_start
        resample_metrics['error'] = str(e)
        # Return original audio if resampling fails
        return audio_data, resample_metrics

def get_redirect_uri():
    """Get appropriate redirect URI for current environment"""
    if 'localhost' in CONFIG['APP_DOMAIN']:
        return "http://localhost:8080"
    else:
        return f"https://{CONFIG['APP_DOMAIN']}"

REDIRECT_URI = get_redirect_uri()

def init_oauth_flow():
    """Initialize Google OAuth flow"""
    params = {
        'client_id': CONFIG['GOOGLE_CLIENT_ID'],
        'redirect_uri': REDIRECT_URI,
        'scope': 'openid email profile',
        'response_type': 'code',
        'access_type': 'offline',
        'prompt': 'consent'
    }
    
    auth_url = f"https://accounts.google.com/o/oauth2/auth?{urlencode(params)}"
    return auth_url

@api_retry(
    max_attempts=3,
    initial_delay=1.0,
    exponential_base=2.0,
    use_streamlit_ui=True
)
@with_timeout(timeout_seconds=30.0)
def exchange_code_for_token(code):
    """Exchange authorization code for access token with retry logic"""
    token_data = {
        'client_id': CONFIG['GOOGLE_CLIENT_ID'],
        'client_secret': CONFIG['GOOGLE_CLIENT_SECRET'],
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': REDIRECT_URI,
    }
    
    response = requests.post(
        'https://oauth2.googleapis.com/token', 
        data=token_data,
        timeout=10.0  # Request-level timeout
    )
    response.raise_for_status()
    return response.json()

def verify_user_email(id_token):
    """Verify user's email from Google ID token"""
    try:
        # Decode JWT token (simplified - in production, verify signature)
        payload_part = id_token.split('.')[1]
        # Add padding if needed
        payload_part += '=' * (4 - len(payload_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_part))
        
        email = payload.get('email')
        return email == CONFIG['AUTHORIZED_EMAIL']
    except Exception as e:
        logger.error(f"Email verification failed: {str(e)}")
        return False

def authenticate_with_gmail():
    """Main authentication function with error handling"""
    
    # Check if already authenticated
    if st.session_state.get('authenticated', False):
        return True
    
    # Check for OAuth callback
    query_params = st.query_params
    if 'code' in query_params:
        st.info("🔄 Processing authentication...")
        code = query_params['code']
        
        try:
            token_response = exchange_code_for_token(code)
            
            if token_response and 'id_token' in token_response:
                if verify_user_email(token_response['id_token']):
                    st.session_state.authenticated = True
                    st.session_state.user_email = CONFIG['AUTHORIZED_EMAIL']
                    st.query_params.clear()
                    st.success("✅ Authentication successful!")
                    st.rerun()
                else:
                    st.error("❌ Access denied. Only authorized email can access this app.")
                    return False
            else:
                st.error("❌ Authentication failed. Please try again.")
        except APIError as e:
            st.error(f"❌ Authentication service error: {str(e)}")
            st.info("Please try again in a few moments.")
        except Exception as e:
            logger.error(f"Unexpected auth error: {str(e)}")
            st.error("❌ Authentication failed. Please contact support if this persists.")
    
    # Handle OAuth errors
    if 'error' in query_params:
        error = query_params['error']
        st.error(f"❌ OAuth Error: {error}")
        if 'error_description' in query_params:
            st.write(f"Description: {query_params['error_description']}")
    
    # Show login button
    if not st.session_state.get('authenticated', False):
        st.markdown("### 🔐 Authentication Required")
        st.info(f"This app is restricted to: {CONFIG['AUTHORIZED_EMAIL']}")
        
        auth_url = init_oauth_flow()
        
        st.markdown(f"""
        <a href="{auth_url}" target="_self">
            <button style="
                background-color: #4285f4;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                text-decoration: none;
                display: inline-block;
            ">
                🔐 Sign in with Google
            </button>
        </a>
        """, unsafe_allow_html=True)
        return False
    
    return True

@api_retry(
    max_attempts=3,
    initial_delay=2.0,
    max_delay=10.0,
    use_streamlit_ui=True
)
@validate_response(
    validator=lambda x: x and x.choices and x.choices[0].message.content,
    error_message="Invalid transcription response"
)
def transcribe_audio(messages, deployment_id=DEPLOYMENT_ID, segment_id=0, audio_metrics=None):
    """Transcribe audio with robust error handling and metrics tracking"""
    start_time = time.time()
    
    try:
        completion = openai_client_us2.chat.completions.create(
            model=deployment_id,
            messages=messages,
            timeout=60  # Add explicit timeout
        )
        
        # Extract token usage
        token_usage = log_token_usage(completion)
        
        # Log successful transcription metrics
        if 'metrics_logger' in st.session_state and audio_metrics:
            metrics = TranscriptionMetrics(
                segment_id=segment_id,
                timestamp=time.time(),
                audio_size_bytes=audio_metrics.get('size_bytes', 0),
                audio_duration_estimate=audio_metrics.get('duration_seconds', 0),
                encoded_size_bytes=audio_metrics.get('encoded_size', 0),
                encoding_time=audio_metrics.get('encoding_time', 0),
                api_call_time=time.time() - start_time,
                total_time=time.time() - audio_metrics.get('start_time', start_time),
                prompt_tokens=token_usage.get('prompt_tokens', 0),
                completion_tokens=token_usage.get('completion_tokens', 0),
                total_tokens=token_usage.get('total_tokens', 0),
                model=deployment_id,
                status='success',
                context_segments_used=len(messages) - 2,  # Excluding system and current user message
                response_length=len(completion.choices[0].message.content),
                original_sample_rate=audio_metrics.get('original_sample_rate'),
                target_sample_rate=TARGET_SAMPLE_RATE if audio_metrics.get('resampled') else None,
                resampled=audio_metrics.get('resampled', False),
                resample_time=audio_metrics.get('resample_time', 0),
                resampled_size_bytes=audio_metrics.get('resampled_size_bytes')
            )
            st.session_state.metrics_logger.log_transcription(metrics)
        
        return completion
        
    except Exception as e:
        # Log failed transcription metrics
        if 'metrics_logger' in st.session_state and audio_metrics:
            metrics = TranscriptionMetrics(
                segment_id=segment_id,
                timestamp=time.time(),
                audio_size_bytes=audio_metrics.get('size_bytes', 0),
                audio_duration_estimate=audio_metrics.get('duration_seconds', 0),
                encoded_size_bytes=audio_metrics.get('encoded_size', 0),
                encoding_time=audio_metrics.get('encoding_time', 0),
                api_call_time=time.time() - start_time,
                total_time=time.time() - audio_metrics.get('start_time', start_time),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model=deployment_id,
                status='error',
                error_type=type(e).__name__,
                context_segments_used=len(messages) - 2,
                original_sample_rate=audio_metrics.get('original_sample_rate'),
                target_sample_rate=TARGET_SAMPLE_RATE if audio_metrics.get('resampled') else None,
                resampled=audio_metrics.get('resampled', False),
                resample_time=audio_metrics.get('resample_time', 0),
                resampled_size_bytes=audio_metrics.get('resampled_size_bytes')
            )
            st.session_state.metrics_logger.log_transcription(metrics)
        
        raise

@api_retry(
    max_attempts=2,
    initial_delay=1.0,
    use_streamlit_ui=True
)
@cache_on_error(cache_key="consolidated_analysis", ttl_seconds=1800)
def create_consolidated_analysis():
    """Create consolidated analysis with error handling and caching"""
    
    if not st.session_state.transcription_segments:
        return "No segments to analyze"
    
    # Compile all segments
    combined_content = "\n\n".join([
        f"--- Segment {i+1} ---\n{segment}" 
        for i, segment in enumerate(st.session_state.transcription_segments)
    ])
    
    consolidation_prompt = """
    Analyze the complete conversation below and provide:
    1. **Combined transcription**: Combined transcript across all segments.
    2. **Reorganised for clarity and ease of comprehension**: Substantive points with clear hierarchy
    3. **Pragmatic inference**: Pragmatic inference using your knowledge as of your last training date
    
    For the output language use the same language as the language of the majority of the input segments.
    
    Complete Conversation:
    {combined_content}
    """
    
    try:
        # Use regular gpt-4o model for consolidation
        consolidation_response = openai_client_us2.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert conversation analyst."},
                {"role": "user", "content": consolidation_prompt.format(combined_content=combined_content)}
            ],
            timeout=45
        )
        
        return consolidation_response.choices[0].message.content
    
    except openai.RateLimitError:
        st.warning("⚠️ Rate limit reached. Please wait a moment before retrying.")
        raise
    except Exception as e:
        logger.error(f"Consolidation failed: {str(e)}")
        # Return a basic consolidation as fallback
        return f"**Basic Transcript Compilation**\n\n{combined_content}"

def handle_transcription_error(error: Exception, segment_num: int):
    """Centralized error handling for transcription failures"""
    error_messages = {
        "rate_limit": "You've reached the usage limit. Please wait a few moments.",
        "timeout": "The transcription took too long. Please try a shorter recording.",
        "auth": "Authentication issue. Please log out and log in again.",
        "connection": "Connection issue. Please check your internet connection.",
    }
    
    # Store error in session state for debugging
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []
    
    st.session_state.error_log.append({
        'segment': segment_num,
        'error': str(error),
        'timestamp': time.time()
    })
    
    # Determine error type and show appropriate message
    for error_type, message in error_messages.items():
        if error_type in str(error).lower():
            st.error(f"❌ Segment #{segment_num} failed: {message}")
            return
    
    # Generic error message
    st.error(f"❌ Segment #{segment_num} failed: Please try again")
    
    # Show debug info in expander
    with st.expander("🐛 Debug Information"):
        st.code(str(error))

def main():
    """Main application with comprehensive error handling and metrics"""
    
    st.set_page_config(page_title="Secure Transcription App", page_icon="🔐")
    
    # Initialize metrics logger
    if 'metrics_logger' not in st.session_state:
        st.session_state.metrics_logger = APIMetricsLogger()
    
    # Display metrics dashboard in sidebar
    st.session_state.metrics_logger.display_metrics_dashboard()
    
    # System health check
    if 'health_check_time' not in st.session_state or \
       time.time() - st.session_state.get('health_check_time', 0) > 300:
        try:
            # Quick health check
            test_completion = openai_client_us2.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=5
            )
            st.session_state.health_check_time = time.time()
            st.session_state.api_healthy = True
        except Exception as e:
            st.session_state.api_healthy = False
            st.warning("⚠️ AI service may be experiencing issues. Some features might be slow.")
            logger.error(f"Health check failed: {str(e)}")
    
    # Authentication
    if ('localhost' in CONFIG['APP_DOMAIN']) | authenticate_with_gmail():
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🚪 Logout"):
                for key in ['authenticated', 'user_email', 'email_verified', 'verification_code']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col1:
            st.title("🎙️ Transcription App")
            st.success(f"✅ Authenticated as: {CONFIG['AUTHORIZED_EMAIL']}")
        
        # Show API health status
        if st.session_state.get('api_healthy', True):
            st.sidebar.success("✅ System Status: Healthy")
        else:
            st.sidebar.warning("⚠️ System Status: Degraded")
        
        # Performance report generation
        if st.sidebar.button("📈 Generate Performance Report"):
            report = create_performance_report()
            st.sidebar.download_button(
                "📥 Download Report",
                report,
                file_name=f"performance_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        # Error log in sidebar (for debugging)
        if st.sidebar.checkbox("Show Error Log") and 'error_log' in st.session_state:
            st.sidebar.json(st.session_state.error_log[-5:])  # Show last 5 errors
        
        # Initialize session state
        if 'transcription_segments' not in st.session_state:
            st.session_state.transcription_segments = []
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "system", "content": system_prompt})
        
        # Display accumulated context
        if st.session_state.transcription_segments:
            st.subheader(f"📝 Previous Context ({len(st.session_state.transcription_segments)} segments)")
            
            # Show total audio processed
            if 'metrics_logger' in st.session_state:
                metrics_summary = st.session_state.metrics_logger.get_current_session_metrics()
                if metrics_summary:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Audio", f"{metrics_summary.get('total_audio_mb', 0):.1f} MB")
                    with col2:
                        st.metric("Total Tokens", f"{metrics_summary.get('total_tokens', 0):,}")
                    with col3:
                        st.metric("Est. Cost", f"${metrics_summary.get('estimated_cost', 0):.3f}")
            
            # Clear button with confirmation
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🗑️ Clear All"):
                    if st.checkbox("Confirm clear?"):
                        st.session_state.transcription_segments = []
                        st.session_state.messages = [{"role": "system", "content": system_prompt}]
                        st.rerun()
            
            st.markdown("---")
        
        # Audio input
        audio_value = st.audio_input("Record your note!")
        
        if audio_value:
            # Track overall processing start time
            processing_start = time.time()

            # Read and measure audio data
            audio_data = audio_value.read()
            audio_metrics = measure_audio_metrics(audio_data)

            # Resample audio to optimal sample rate for GPT-4o using configured backend
            resampled_audio, resample_metrics = resample_audio_auto(audio_data, TARGET_SAMPLE_RATE)

            # Update metrics with resampling info
            audio_metrics['original_sample_rate'] = resample_metrics.get('original_sample_rate')
            audio_metrics['resampled'] = resample_metrics.get('resampled', False)
            audio_metrics['resample_time'] = resample_metrics.get('resample_time', 0)
            audio_metrics['backend'] = resample_metrics.get('backend', 'unknown')
            audio_metrics['converted_to_mono'] = resample_metrics.get('converted_to_mono', False)

            # If resampled, update metrics with new audio properties
            if resample_metrics['resampled']:
                # Re-measure the resampled audio
                resampled_metrics = measure_audio_metrics(resampled_audio)
                audio_metrics['resampled_size_bytes'] = resampled_metrics['size_bytes']
                audio_metrics['resampled_size_kb'] = resampled_metrics['size_kb']
                # Keep original duration since resampling doesn't change it
                audio_metrics['duration_seconds'] = audio_metrics.get('duration_seconds', resampled_metrics.get('duration_seconds'))

            # Encode audio (track encoding time) - use resampled audio
            encoding_start = time.time()
            encoded_audio_string = base64.b64encode(resampled_audio).decode("utf-8")
            encoding_time = time.time() - encoding_start
            
            # Add metrics to audio_metrics dict
            audio_metrics['encoded_size'] = len(encoded_audio_string)
            audio_metrics['encoding_time'] = encoding_time
            audio_metrics['start_time'] = processing_start
            
            # Log audio metrics
            size_info = f"{audio_metrics['size_kb']:.1f}KB"
            if audio_metrics.get('resampled'):
                size_info = f"{audio_metrics['size_kb']:.1f}KB → {audio_metrics.get('resampled_size_kb', 0):.1f}KB"

            logger.info(f"Audio metrics: Size={size_info}, "
                       f"Duration={audio_metrics['duration_seconds']:.1f}s, "
                       f"Sample Rate={audio_metrics.get('original_sample_rate', 'unknown')} Hz → {TARGET_SAMPLE_RATE} Hz, "
                       f"Resample time={audio_metrics.get('resample_time', 0):.3f}s, "
                       f"Encoding time={encoding_time:.3f}s")
            
            # Build messages with context
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            
            # Add previous context
            if st.session_state.transcription_segments:
                context = "Previous transcript segments:\n\n"
                for i, segment in enumerate(st.session_state.transcription_segments[-5:], 1):
                    context += f"--- Segment {i} ---\n{segment}\n\n"
                
                messages.append({
                    "role": "assistant",
                    "content": context
                })
            
            # Add current audio
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "input_audio", "input_audio": {"data": encoded_audio_string, "format": "wav"}}
                ]
            })
            
            st.session_state.messages = messages
            
            # Process transcription
            current_segment = len(st.session_state.transcription_segments) + 1
            
            # Show audio info
            audio_info = f"📊 Audio: {audio_metrics['size_kb']:.1f}KB"
            if audio_metrics.get('resampled'):
                audio_info += f" → {audio_metrics.get('resampled_size_kb', 0):.1f}KB (resampled)"
            audio_info += f", {audio_metrics['duration_seconds']:.1f}s"
            if audio_metrics.get('original_sample_rate'):
                audio_info += f"\n🎵 Sample Rate: {audio_metrics['original_sample_rate']} Hz → {TARGET_SAMPLE_RATE} Hz"
            if audio_metrics.get('converted_to_mono'):
                audio_info += f" (stereo → mono)"
            if audio_metrics.get('backend'):
                audio_info += f"\n⚙️ Backend: {audio_metrics['backend']}"

            st.info(f"🎯 Processing Segment #{current_segment}\n"
                   f"{audio_info}" +
                   (f"\n📝 Context: {len(st.session_state.transcription_segments)} previous segments"
                    if st.session_state.transcription_segments else ""))
            
            try:
                # Transcribe with error handling and metrics
                completion = transcribe_audio(
                    st.session_state.messages,
                    segment_id=current_segment,
                    audio_metrics=audio_metrics
                )
                
                if completion and completion.choices:
                    response = completion.choices[0].message
                    
                    # Calculate total processing time
                    total_time = time.time() - processing_start
                    
                    st.success(f"✅ Segment #{current_segment} completed in {total_time:.1f}s!")
                    
                    # Display transcription
                    st.subheader(f"📝 Latest Transcription (Segment #{current_segment})")
                    st.write(response.content)
                    
                    # Store segment
                    st.session_state.transcription_segments.append(response.content)
                    
                    # Show performance metrics
                    if 'metrics_logger' in st.session_state:
                        with st.expander("📊 Segment Performance Metrics"):
                            latest_metrics = st.session_state.api_metrics_log[-1] if 'api_metrics_log' in st.session_state else {}
                            if latest_metrics:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("API Latency", f"{latest_metrics.get('api_call_time', 0):.2f}s")
                                    st.metric("Tokens Used", f"{latest_metrics.get('total_tokens', 0):,}")
                                with col2:
                                    st.metric("Processing Rate", f"{latest_metrics.get('tokens_per_second', 0):.1f} tok/s")
                                    st.metric("Audio/Process Ratio", f"{latest_metrics.get('audio_duration_estimate', 0) / max(latest_metrics.get('api_call_time', 1), 0.1):.1f}x")
                                with col3:
                                    st.metric("Cost", f"${latest_metrics.get('cost_estimate', 0):.4f}")
                                    st.metric("Response Length", f"{latest_metrics.get('response_length', 0):,} chars")
                    
                    # Show context summary
                    if len(st.session_state.transcription_segments) > 1:
                        with st.expander("🔍 Context Used"):
                            st.write(f"**Previous segments**: {len(st.session_state.transcription_segments) - 1}")
                            st.write("**Context included**: Last 5 segments")
                
            except APIError as e:
                handle_transcription_error(e, current_segment)
                
                # Offer to save partial progress
                if st.session_state.transcription_segments:
                    if st.button("💾 Save partial transcript"):
                        combined = "\n\n".join(st.session_state.transcription_segments)
                        st.download_button(
                            "📥 Download",
                            combined,
                            file_name=f"partial_transcript_{current_segment-1}_segments.txt",
                            mime="text/plain"
                        )
                        
            except Exception as e:
                logger.error(f"Unexpected error in transcription: {str(e)}")
                handle_transcription_error(e, current_segment)
        
        # Show accumulated conversation
        if len(st.session_state.transcription_segments) > 1:
            st.markdown("---")
            
            combined_transcript = ""
            for i, segment in enumerate(st.session_state.transcription_segments, 1):
                combined_transcript += f"\n\n--- Segment {i} ---\n\n{segment}"
            
            # Consolidation with error handling
            with st.expander("🧠 Consolidated Analysis", expanded=False):
                if st.button("🔄 Generate Consolidated Analysis"):
                    with st.spinner("Analyzing complete conversation..."):
                        try:
                            consolidated = create_consolidated_analysis()
                            st.session_state.consolidated_analysis = consolidated
                            st.session_state.consolidation_timestamp = time.time()
                        except Exception as e:
                            st.error("Failed to generate analysis. Using basic compilation.")
                            st.session_state.consolidated_analysis = combined_transcript
                
                if hasattr(st.session_state, 'consolidated_analysis'):
                    # Show cache age if applicable
                    if hasattr(st.session_state, 'consolidation_timestamp'):
                        age = time.time() - st.session_state.consolidation_timestamp
                        if age > 60:
                            st.caption(f"Generated {age/60:.0f} minutes ago")
                    
                    st.write(st.session_state.consolidated_analysis)
            
            # Individual segments
            with st.expander("📋 Individual Segments", expanded=False):
                for i, segment in enumerate(st.session_state.transcription_segments, 1):
                    st.markdown(f"**--- Segment {i} ---**")
                    st.write(segment)
            
            # Download with error handling
            try:
                st.download_button(
                    "📥 Download Full Transcript",
                    combined_transcript,
                    file_name="full_transcript.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error("Download failed. Copy text manually if needed.")
                st.text_area("Manual Copy", combined_transcript, height=200)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical application error: {str(e)}")
        st.error("🚨 Critical Error: The application encountered an unexpected error.")
        st.info("Please refresh the page or contact support if this persists.")
        
        if st.checkbox("Show technical details"):
            st.exception(e)