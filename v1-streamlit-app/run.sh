#!/bin/bash
# Run script for v1 Streamlit app with proper paths

# Add parent directory to PYTHONPATH for shared components
export PYTHONPATH="../:$PYTHONPATH"

# Run streamlit app
streamlit run app.py "$@"