# FastAPI GPT-4o Audio Transcription Application (v2)

Next-generation FastAPI implementation of the GPT-4o audio transcription system.

## Status

🚧 **In Development** - This version is currently being implemented as part of the migration from Streamlit to FastAPI.

## Planned Features

- RESTful API endpoints for audio transcription
- WebSocket support for real-time transcription
- Enhanced performance and scalability
- Jinja2 template-based UI
- Vanilla JavaScript frontend
- Full feature parity with v1 Streamlit application

## Architecture

- FastAPI backend for API endpoints
- Uvicorn ASGI server
- Jinja2 templates for server-side rendering
- Vanilla JavaScript for client-side interactivity
- Shared components from `shared-components` package

## Development

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn main:app --reload
```

## Migration from v1

This application maintains feature parity with the v1 Streamlit application while providing:
- Better performance
- More flexible API design
- Enhanced scalability
- Improved testing capabilities

See [Migration Guide](../docs-shared/migration.md) for details.