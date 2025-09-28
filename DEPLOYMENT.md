# Deployment Guide for GPT-4o Audio Transcription App

## Render Deployment (Docker)

This application is configured to deploy on Render using Docker for better control over the environment and dependencies.

### Prerequisites

1. A Render account (https://render.com)
2. The following environment variables configured in Render Dashboard:
   - `GOOGLE_CLIENT_ID` - Your Google OAuth Client ID
   - `GOOGLE_CLIENT_SECRET` - Your Google OAuth Client Secret
   - `AZURE_OPENAI_API_ENDPOINT_US2` - Azure OpenAI endpoint
   - `AZURE_OPENAI_API_KEY_US2` - Azure OpenAI API key
   - `AUTHORIZED_EMAIL` - Email authorized to access the app
   - `APP_DOMAIN` - Your Render app domain (e.g., `fouro-audio-streamlit.onrender.com`)

### Deployment Steps

1. **Connect GitHub Repository**
   - Log into Render Dashboard
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   - Name: `transcription-app` (or your preferred name)
   - Environment: Docker (automatically detected from render.yaml)
   - Region: Choose closest to your users

3. **Set Environment Variables**
   - In the Render Dashboard, go to Environment
   - Add all required environment variables listed above
   - Make sure sensitive values are marked as secret

4. **Deploy**
   - Render will automatically deploy from your main/master branch
   - First deployment may take 10-15 minutes due to Docker image building

### File Structure for Deployment

```
Repository Root/
├── render.yaml           # Render configuration (Docker mode)
├── Dockerfile           # Docker container definition
├── .dockerignore        # Files to exclude from Docker build
├── v1-streamlit-app/    # Application directory
│   ├── app.py
│   ├── requirements.txt
│   └── .streamlit/
└── shared_components/   # Shared utilities
```

### Key Configuration Files

#### render.yaml
- Located at repository root
- Configures Render to use Docker deployment
- Defines environment variables

#### Dockerfile
- Located at repository root
- Installs system dependencies (ffmpeg, libsndfile1)
- Installs Python dependencies
- Sets up proper working directory and PYTHONPATH

#### .dockerignore
- Excludes unnecessary files from Docker build
- Reduces image size and build time

### Google OAuth Setup

1. **Configure OAuth Redirect URI**
   - In Google Cloud Console, add your Render URL as authorized redirect:
   - `https://your-app-name.onrender.com`

2. **Update APP_DOMAIN**
   - Set `APP_DOMAIN` in Render environment to match your app URL
   - Example: `fouro-audio-streamlit.onrender.com`

### Troubleshooting

#### Import Errors
- The Dockerfile sets `PYTHONPATH=/app` to ensure shared_components can be imported
- The app changes to `v1-streamlit-app` directory before running

#### Audio Processing Issues
- FFmpeg and libsndfile1 are installed in the Docker image
- These are required for audio resampling functionality

#### OAuth Redirect Issues
- Ensure `APP_DOMAIN` matches your actual Render URL
- Check that the redirect URI is added in Google Cloud Console
- Verify `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are correct

### Updating the Deployment

1. **Automatic Deploys**
   - Push changes to your main branch
   - Render will automatically rebuild and deploy

2. **Manual Deploy**
   - Use Render Dashboard to trigger manual deploy
   - Useful for deploying from specific branches

### Performance Optimization

- Docker image is optimized with multi-stage build considerations
- .dockerignore reduces build context size
- System dependencies are installed in a single layer

### Alternative: Deploy Without Docker

If you prefer not to use Docker, you can modify render.yaml:

```yaml
services:
  - type: web
    name: transcription-app
    env: python
    buildCommand: |
      apt-get update && apt-get install -y ffmpeg libsndfile1
      pip install -r v1-streamlit-app/requirements.txt
    startCommand: |
      cd v1-streamlit-app && streamlit run app.py
```

Note: This requires Render to install system dependencies which may not always be supported.

## Local Testing with Docker

To test the Docker deployment locally:

```bash
# Build the Docker image
docker build -t transcription-app .

# Run locally with environment variables
docker run -p 8080:8080 \
  -e GOOGLE_CLIENT_ID=your_client_id \
  -e GOOGLE_CLIENT_SECRET=your_secret \
  -e AZURE_OPENAI_API_ENDPOINT_US2=your_endpoint \
  -e AZURE_OPENAI_API_KEY_US2=your_key \
  -e AUTHORIZED_EMAIL=your_email \
  -e APP_DOMAIN=localhost:8080 \
  transcription-app
```

Access the app at `http://localhost:8080`