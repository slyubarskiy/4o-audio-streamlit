# MINIMAL CHANGES to add context to your existing code

# **Transcription Script** (Your existing code with MINIMAL additions)
import dotenv
import openai
import os
import streamlit as st
import base64 

import requests
from urllib.parse import urlencode
import jwt
import json

import dotenv

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
    config['APP_DOMAIN'] = get_secret('APP_DOMAIN', 'localhost:8501')
    
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

# OpenAI client setup

openai_client_us2 = openai.AzureOpenAI(
    api_key=CONFIG['AZURE_OPENAI_API_KEY_US2'],
    api_version="2025-04-01-preview",
    azure_endpoint=CONFIG['AZURE_OPENAI_API_ENDPOINT_US2'],
    organization='Transcript PoC'
)


DEPLOYMENT_ID = "gpt-4o-audio-preview"

system_prompt = """
You are a helpful AI Transcription Assistant. Create a transcript of the provided audio
"""

# ENHANCED: Add context awareness to your user prompt
user_prompt = """### TASKS:
Separate the outputs of each of the 3 subtasks below with the horizontal line "\n---------------------------------------\n" and enclose the entire output in the new lines "\n#####################################################################\n":
1. Transcribe the input_audio, enriching it with labels conveying sentiments, emotions and emphasis labels in square brackets in-line.
2. Append to it your paralinguistic analysis in russian (non-verbal cues such as tone, pauses, etc.) output in the form of implications of emotions, emphasis or attitude to understand how things are said and provide insight into emotions, attitudes conveyed through speech patterns).
3. Append to it the reorganised insights from this transcript, its labels and paralinguistic analysis for clarity and ease of comprehension by someone with User.

IMPORTANT: If there are previous segments in this conversation, reference them when relevant and note any connections or continuations.
"""


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

def exchange_code_for_token(code):
    """Exchange authorization code for access token"""
    token_data = {
        'client_id': CONFIG['GOOGLE_CLIENT_ID'],
        'client_secret': CONFIG['GOOGLE_CLIENT_SECRET'],
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': REDIRECT_URI,
    }
    
    try:
        response = requests.post('https://oauth2.googleapis.com/token', data=token_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Token exchange failed: {e}")
        return None

# Determine redirect URI based on environment
def get_redirect_uri():
    """Get appropriate redirect URI for current environment"""
    if 'localhost' in CONFIG['APP_DOMAIN']:
        return "http://localhost:8501"
    else:
        return f"https://{CONFIG['APP_DOMAIN']}"

REDIRECT_URI = get_redirect_uri()

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
    except:
        return False

def authenticate_with_gmail():
    """Main authentication function"""
    
    # Debug information (only show in development)
    if 'localhost' in REDIRECT_URI and st.sidebar.button("🔧 Debug OAuth Config"):
        st.sidebar.write("**OAuth Configuration:**")
        st.sidebar.write(f"Client ID: {CONFIG['GOOGLE_CLIENT_ID'][:20]}...")
        st.sidebar.write(f"Redirect URI: {REDIRECT_URI}")
        st.sidebar.write(f"Authorized Email: {CONFIG['AUTHORIZED_EMAIL']}")
    
    # Check if already authenticated
    if st.session_state.get('authenticated', False):
        return True
    
    # Check for OAuth callback
    query_params = st.query_params
    if 'code' in query_params:
        st.info("🔄 Processing authentication...")
        code = query_params['code']
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
    
def create_consolidated_analysis():
    """Separate function for consolidated analysis using gpt-4o"""
    
    # Compile all segments
    combined_content = "\n\n".join([
        f"--- Segment {i+1} ---\n{segment}" 
        for i, segment in enumerate(st.session_state.transcription_segments)
    ])
    
    consolidation_prompt = """
    Analyze the complete conversation below and provide:
    1. **Comprehensive Summary**: Key themes and progression
    2. **Combined Paralinguistic Analysis**: Overall emotional journey 
    3. **User-Optimized Structure**: Main points with clear hierarchy
    
    Complete Conversation:
    {combined_content}
    """
    
    # Use regular gpt-4o model for consolidation
    consolidation_response = openai_client_us2.chat.completions.create(
        model="gpt-4o",  # Use standard model
        messages=[
            {"role": "system", "content": "You are an expert conversation analyst."},
            {"role": "user", "content": consolidation_prompt.format(combined_content=combined_content)}
        ],
#        response_format={"type": "json_object"}  # More likely to work with standard model
    )
    
    return consolidation_response.choices[0].message.content

def parse_transcription_response(response_text):
    """Parse the response using delimiters instead of JSON"""
    sections = response_text.split("---------------------------------------")
    
    parsed = {
        "transcription": "",
        "paralinguistic_analysis": "",
        "user_insights": ""
    }
    
    if len(sections) >= 3:
        parsed["transcription"] = sections[0].strip()
        parsed["paralinguistic_analysis"] = sections[1].strip()
        parsed["user_insights"] = sections[2].strip()
    else:
        # Fallback - return full response
        parsed["transcription"] = response_text
    
    return parsed


def main1():
    st.set_page_config(page_title="OAuth Debug Tool", page_icon="🔧")
    
    st.title("🔧 OAuth Debug Tool")
    
    # Show current configuration
    st.markdown("### 📋 Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**App Configuration:**")
        st.code(f"""
Client ID: {CONFIG.get('GOOGLE_CLIENT_ID', 'MISSING')[:30]}...
App Domain: {CONFIG.get('APP_DOMAIN', 'MISSING')}
Redirect URI: {REDIRECT_URI}
Authorized Email: {CONFIG.get('AUTHORIZED_EMAIL', 'MISSING')}
        """)
    
    with col2:
        st.markdown("**Current Request Info:**")
        
        # Get current URL info
        query_params = dict(st.query_params)
        
        st.code(f"""
Current URL: {st.context.request.url if hasattr(st.context, 'request') else 'Unknown'}
Query Params: {query_params}
        """)
    
    # Show what to add to Google OAuth
    st.markdown("### 🔗 Google OAuth Configuration")
    st.info("Add BOTH of these URLs to your Google OAuth Client:")
    
    st.code(f"""
Authorized redirect URIs:
1. {REDIRECT_URI}
2. {REDIRECT_URI}/
    """)
    
    # Test OAuth flow
    st.markdown("### 🧪 Test OAuth Flow")
    
    # Check for OAuth response
    if 'code' in st.query_params:
        st.success("✅ OAuth callback received!")
        code = st.query_params['code']
        st.write(f"**Authorization Code**: {code[:50]}...")
        
        # Test token exchange
        if st.button("🔄 Test Token Exchange"):
            with st.spinner("Testing token exchange..."):
                token_data = {
                    'client_id': CONFIG['GOOGLE_CLIENT_ID'],
                    'client_secret': CONFIG['GOOGLE_CLIENT_SECRET'],
                    'code': code,
                    'grant_type': 'authorization_code',
                    'redirect_uri': REDIRECT_URI,
                }
                
                try:
                    response = requests.post('https://oauth2.googleapis.com/token', data=token_data)
                    st.write(f"**Response Status**: {response.status_code}")
                    
                    if response.status_code == 200:
                        token_response = response.json()
                        st.success("✅ Token exchange successful!")
                        
                        if 'id_token' in token_response:
                            # Decode ID token
                            id_token = token_response['id_token']
                            payload_part = id_token.split('.')[1]
                            payload_part += '=' * (4 - len(payload_part) % 4)
                            payload = json.loads(base64.urlsafe_b64decode(payload_part))
                            
                            st.write(f"**User Email**: {payload.get('email')}")
                            st.write(f"**Authorized Email**: {CONFIG['AUTHORIZED_EMAIL']}")
                            
                            if payload.get('email') == CONFIG['AUTHORIZED_EMAIL']:
                                st.success("✅ Email authorization successful!")
                            else:
                                st.error("❌ Email not authorized")
                        else:
                            st.error("❌ No ID token in response")
                            st.json(token_response)
                    else:
                        st.error(f"❌ Token exchange failed: {response.status_code}")
                        st.json(response.json())
                        
                except Exception as e:
                    st.error(f"❌ Token exchange error: {e}")
    
    elif 'error' in st.query_params:
        st.error(f"❌ OAuth Error: {st.query_params['error']}")
        if 'error_description' in st.query_params:
            st.write(f"**Description**: {st.query_params['error_description']}")
        
        st.markdown("### 🔍 Common Solutions:")
        st.markdown("""
        - **403 Forbidden**: Check OAuth consent screen test users
        - **redirect_uri_mismatch**: Update Google OAuth redirect URIs
        - **unauthorized_client**: Check client ID and secret
        """)
    
    else:
        # Show OAuth URL
        if CONFIG.get('GOOGLE_CLIENT_ID'):
            params = {
                'client_id': CONFIG['GOOGLE_CLIENT_ID'],
                'redirect_uri': REDIRECT_URI,
                'scope': 'openid email profile',
                'response_type': 'code',
                'access_type': 'offline',
                'prompt': 'consent',
                'include_granted_scopes': 'true'
            }
            
            auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
            
            st.markdown("### 🚀 Test OAuth Flow")
            
            # Show the auth URL for manual inspection
            with st.expander("🔍 Inspect OAuth URL"):
                st.code(auth_url)
            
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
                    🔐 Test Google OAuth
                </button>
            </a>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Google Client ID not configured")
    
    # Configuration checklist
    st.markdown("### ✅ Configuration Checklist")
    
    checklist = [
        ("Google Client ID configured", bool(CONFIG.get('GOOGLE_CLIENT_ID'))),
        ("Google Client Secret configured", bool(CONFIG.get('GOOGLE_CLIENT_SECRET'))),
        ("Authorized email configured", bool(CONFIG.get('AUTHORIZED_EMAIL'))),
        ("App domain configured", bool(CONFIG.get('APP_DOMAIN'))),
    ]
    
    for item, status in checklist:
        if status:
            st.success(f"✅ {item}")
        else:
            st.error(f"❌ {item}")


def main():
    """Main application"""

    # -------------------auth

    st.set_page_config(page_title="Secure Transcription App", page_icon="🔐")

    # Choose authentication method
    # Show logout option
    if authenticate_with_gmail():
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

        # ------------------- auth end

        st.title("Test Assistant")


        if 'transcription_segments' not in st.session_state:
            st.session_state.transcription_segments = []

        if 'messages' not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "system", "content": system_prompt})

        # NEW: Display accumulated context before new recording
        if st.session_state.transcription_segments:
            st.subheader(f"📝 Previous Context ({len(st.session_state.transcription_segments)} segments)")
            
            # Show context summary
        #    with st.expander("📋 View Previous Segments", expanded=False):
        #        for i, segment in enumerate(st.session_state.transcription_segments, 1):
        #            st.markdown(f"**--- Segment {i} ---**")
        #            st.write(segment)
        #            st.markdown("---")
            
            # Clear button
            if st.button("🗑️ Clear All Context"):
                st.session_state.transcription_segments = []
                st.session_state.messages = [{"role": "system", "content": system_prompt}]
                st.rerun()
            
            st.markdown("---")

        audio_value = st.audio_input("Ask your question!")

        if audio_value:
            audio_data = audio_value.read()
            encoded_audio_string = base64.b64encode(audio_data).decode("utf-8")
            
            # NEW: Enhanced prompt with context information
            current_segment = len(st.session_state.transcription_segments) + 1
            
            # Add context to the prompt if there are previous segments
            if st.session_state.transcription_segments:
                context_info = f"\n\n**CONTEXT**: This is segment #{current_segment}. Previous segments contain:\n"
                for i, segment in enumerate(st.session_state.transcription_segments, 1):
                    # Include a summary of each previous segment
                    preview = segment[:200] + "..." if len(segment) > 200 else segment
                    context_info += f"Segment {i}: {preview}\n"
                
                enhanced_prompt = user_prompt + context_info
            else:
                enhanced_prompt = f"**SEGMENT #1 (First segment)**\n\n{user_prompt}"
            
            # MODIFIED: Use enhanced prompt instead of basic user_prompt
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt},  # Changed from user_prompt
                        {
                            "type": "input_audio",
                            "input_audio": {"data": encoded_audio_string, "format": "wav"},
                        }
                    ],
                }
            )
            
            completion = None

            try:
                # Show which segment we're processing
                st.info(f"🎯 Processing Segment #{current_segment}" + 
                        (f" (with context from {len(st.session_state.transcription_segments)} previous segments)" 
                         if st.session_state.transcription_segments else " (first segment)"))
                
                completion = openai_client_us2.chat.completions.create(
                    model=DEPLOYMENT_ID,
                    messages=st.session_state.messages,  # This includes all previous context
                )
            
                # Display the response
                if completion and completion.choices:
                    response = completion.choices[0].message
                    

                    
                    # Store the new segment
                    st.session_state.transcription_segments.append(response.content)
                    
                    # Show success with segment number
                    st.success(f"✅ Segment #{current_segment} completed!")
                    
                    # Display current transcription
                    st.subheader(f"📝 Latest Transcription (Segment #{current_segment})")
                    st.write(response.content)
                    # parsed_result = parse_transcription_response(response.content)
                    # st.write (parsed_result["transcription"])
                    # st.write (parsed_result["paralinguistic_analysis"])
                    # st.write (parsed_result["user_insights"])
                                
                    # NEW: Show context summary
                    if len(st.session_state.transcription_segments) > 1:
                        with st.expander("🔍 Context Used"):
                            st.write(f"**Previous segments**: {len(st.session_state.transcription_segments) - 1}")
                            st.write("**Context included**: Summaries of all previous segments")

                    # Add assistant response to conversation
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.content  # Changed from list format to simple string
                    })
                  
                    
            except Exception as e:
                print("Error in completion", e)
                st.write("Error in completion", e)
                st.stop()


        # NEW: Show accumulated conversation at the bottom
        if len(st.session_state.transcription_segments) > 1:
            st.markdown("---")
        #    st.subheader("📜 Full Conversation")
            
            combined_transcript = ""
            for i, segment in enumerate(st.session_state.transcription_segments, 1):
                combined_transcript += f"\n\n--- Segment {i} ---\n\n{segment}"
            
        #    st.text_area("Complete Transcript", combined_transcript, height=300)

        #    # Add consolidation section

        #    st.markdown("---")
            
            # Consolidation expander
            with st.expander("🧠 Consolidated Analysis", expanded=False):
                if st.button("🔄 Generate Consolidated Analysis"):
                    with st.spinner("Analyzing complete conversation..."):
                        consolidated = create_consolidated_analysis()
                        st.session_state.consolidated_analysis = consolidated
                
                if hasattr(st.session_state, 'consolidated_analysis'):
                    st.write(st.session_state.consolidated_analysis)
            
            # Individual segments expander  
            with st.expander("📋 Individual Segments", expanded=False):
                for i, segment in enumerate(st.session_state.transcription_segments, 1):
                    st.markdown(f"**--- Segment {i} ---**")
                    st.write(segment)
           
            # Download option
            st.download_button(
                "📥 Download Full Transcript",
                combined_transcript,
                file_name="full_transcript.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()