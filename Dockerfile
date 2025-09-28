FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire repository structure
COPY . .

# Install shared_components first
RUN pip install --no-cache-dir -e ./shared_components

# Install v1-streamlit-app dependencies
RUN pip install --no-cache-dir -r ./v1-streamlit-app/requirements.txt

# Set working directory to v1-streamlit-app
WORKDIR /app/v1-streamlit-app

# Expose Streamlit's default port
EXPOSE 8080

# Configure Streamlit
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONPATH=/app

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]