#!/bin/bash
# Launch Streamlit app
# This avoids Gradio's file upload issues

cd "$(dirname "$0")"
streamlit run app_streamlit.py
