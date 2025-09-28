# GPT-4o Audio Transcription System

A production-ready audio transcription system using OpenAI's GPT-4o audio preview model with both Streamlit (v1) and FastAPI (v2) implementations.

## Repository Structure

```
4o-audio-streamlit/
├── v1-streamlit-app/     # Current production Streamlit application
├── v2-fastapi-app/       # Future FastAPI implementation (in development)
├── shared_components/    # Shared business logic and utilities
├── docs-shared/          # Common documentation
├── tests/               # Test suites for all versions
└── scripts/             # Utility and migration scripts
```

## Quick Start

### Using UV (Recommended)

```bash
# Install UV if not already installed
pip install uv

# Sync dependencies for all workspace members
uv sync

# Run Streamlit app (v1)
cd v1-streamlit-app
uv run streamlit run app.py

# Future: Run FastAPI app (v2)
cd v2-fastapi-app
uv run uvicorn main:app --reload
```

### Development

This repository uses UV workspace management for modern dependency handling:

- **Workspace Configuration**: Root `pyproject.toml` defines workspace members
- **Shared Components**: Reusable utilities in `shared_components/`
- **Version-specific Dependencies**: Each app version has its own `pyproject.toml`

## Features

- **Audio Processing**: Real-time audio recording and transcription via web interface
- **Multi-segment Support**: Contextual transcription across multiple audio segments
- **Consolidated Analysis**: Intelligent summarization of complete conversations
- **Language Detection**: Automatic language identification
- **Performance Monitoring**: Real-time metrics tracking with cost estimation
- **Secure Access**: Google OAuth 2.0 authentication

## Architecture

The system follows a modular architecture with shared components between v1 (Streamlit) and v2 (FastAPI) implementations:

- **v1-streamlit-app**: Current production application
- **v2-fastapi-app**: Next-generation implementation (in development)
- **shared_components**: Common business logic, error handling, and metrics

## Documentation

- [Project Documentation](docs-shared/project_doc.md) - Detailed system documentation
- [API Documentation](v1-streamlit-app/docs/api.md) - API reference
- [Migration Guide](docs-shared/migration.md) - v1 to v2 migration guide

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run v1 specific tests
uv run pytest tests/v1-streamlit/

# Run performance benchmarks
uv run pytest tests/performance/
```

## License

Copyright © 2024. All rights reserved.