# Streamlit GPT-4o Audio Transcription Application (v1)

Production Streamlit application for GPT-4o audio transcription with Google OAuth authentication.

## Features

- Real-time audio recording and transcription
- Multi-segment conversation support
- Consolidated analysis generation
- Performance metrics tracking
- Secure OAuth authentication

## Setup

1. Configure environment variables or Streamlit secrets:
   ```
   GOOGLE_CLIENT_ID=your_client_id
   GOOGLE_CLIENT_SECRET=your_client_secret
   AZURE_OPENAI_API_KEY_US2=your_api_key
   AZURE_OPENAI_API_ENDPOINT_US2=your_endpoint
   AUTHORIZED_EMAIL=your_email@example.com
   APP_DOMAIN=your_domain
   ```

2. Run the application:
   ```bash
   uv run streamlit run app.py
   ```

## Configuration

- `.streamlit/config.toml` - Streamlit configuration
- `.env` - Local environment variables (not committed)
- `.streamlit/secrets.toml` - Production secrets (not committed)

## Dependencies

See `pyproject.toml` for complete dependency list. Key dependencies:
- streamlit - Web UI framework
- openai - Azure OpenAI API client
- PyJWT - Token validation
- librosa - Audio processing

## Deployment

Supports deployment to:
- Render (using render.yaml)
- Docker (using .dockerfile)
- Any platform supporting Streamlit apps