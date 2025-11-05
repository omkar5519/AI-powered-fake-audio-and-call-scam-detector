#!/bin/bash

# Use $PORT if set (Render), otherwise default to 8501 for local testing
PORT=${PORT:=8501}

# Start Streamlit
streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
